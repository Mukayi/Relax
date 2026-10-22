# Straggler（慢 rank）分析

Relax 的 Megatron 后端内置一个常驻、低开销的慢 rank 分析器：每个训练 rank 用计算流上的 CUDA event 记录自己每一步的 forward / backward / optimizer / 梯度同步 / 参数 all-gather 等段的 GPU 时间（区间内 CPU 侧的 launch 空隙也算在里面），每隔 K 个 rollout 把一个固定长度的统计向量 Gloo all-gather 到 primary rank，由 primary 上的一个后台线程判定哪个 rank 拖慢了整组、为什么，并把结论写进指标、timeline 和日志。

它不是 `torch.profiler`：不采 kernel、不做 `cuda.synchronize()`、不改变通算 overlap，可以在生产训练里一直开着。分析器自身的任何异常都不会抛进训练循环：它会计数，屡次失败后在所有 rank 上一起自动关闭。

## 开启

分析器默认关闭。通过 Ray 运行时环境把环境变量传给训练 Actor：

```yaml
# configs/env.yaml
env_vars:
  RELAX_STRAGGLER_PROFILER: "1"
  RELAX_STRAGGLER_REPORT_INTERVAL: "10"
```

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `RELAX_STRAGGLER_PROFILER` | bool | false | 开关。 |
| `RELAX_STRAGGLER_REPORT_INTERVAL` | int | 10 | 每多少个 rollout 做一次跨 rank 汇聚与判定（一个"窗口"）。一个 rollout 可能包含多个 optimizer step；下文的 ms/step 都是每个 rollout 的累计值。 |
| `RELAX_STRAGGLER_Z_THRESHOLD` | float | 3.0 | 候选 rank 的 robust z 分数（基于组内中位数 / MAD）门限。 |
| `RELAX_STRAGGLER_REL_THRESHOLD` | float | 0.10 | 候选 rank 相对组内中位数的最小超出比例。 |
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | 连续多少个窗口满足条件才告警（所有 reason 都适用），用来过滤 checkpoint、GC 峰等一次性抖动；必须 ≥ 1。 |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | 已告警的 rank 要连续多少个干净窗口才解除告警，避免告警反复开关；「不确定」的窗口不算干净；必须 ≥ 1。 |
| `RELAX_STRAGGLER_GATHER_TIMEOUT_S` | float | 60.0 | profiler 专用 Gloo 组的超时（秒）。单次 gather 是几毫秒的 CPU 小张量汇聚；这个值只限制「有 rank 已经消失」时其余 rank 阻塞多久。 |

开启后每个 actor 角色的进程启动时打一行 `Straggler profiler enabled: report_interval=… primary=… level=… control=… trace_dir=… meta=RankMeta(rank=…, dp=…, tp=…, pp=…, cp=…, ep=…, host=…, device=…, primary=…)`（`trace_dir=-` 表示不采样明细，见下文「采样步明细」）；critic / reference / `actor_fwd` 不跑训练步，不会被采样。

## 看什么

### 日志（primary rank）

日志由 primary 上的分析线程在判定一结束就打出，`step` 是该窗口最后一个 rollout 的 step，不等下一个训练步。没有告警的窗口打一行 INFO：

```text
[straggler] step=29 no straggler (uncertain 0); self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

有告警时，某 rank 的 reason 变化的那个窗口打一条 WARNING，之后每个仍有告警的窗口打一张 INFO per-rank 表。每种 reason 都要连续 `PERSIST_WINDOWS` 个窗口才成立，要连续 `RECOVER_WINDOWS` 个干净窗口才解除（解除时再打一条 WARNING：`rank 5 (…) recovered from slow_device after 2 clean windows`）：

```text
[straggler] step=29 rank 3 (rank3_dp3_tp0_pp0, host=…, gpu=3) reaches the DP grad-sync 197.4 ms/step
after its DP peers (peers idle 203.3 ms/step = 22% of their compute, z=41.0) for 3 windows;
its own GPU forward is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.56x -> late_arrival (host-side stall)
[straggler] step=29 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | steady_ms | mb_fwd_max | reason       | uncertain
    0 | rank0_dp0_tp0_pp0 | …    |      0 |   933.7 |     6.4 |     6.1 | 14802.1 |        63.1 |   1.8 |       0.0 |      157.8 | none         |
    3 | rank3_dp3_tp0_pp0 | …    |      3 |   737.1 |     5.9 |   197.4 | 14592.7 |        50.5 |   1.8 |       0.0 |      148.0 | late_arrival |
```

表里只列出 rank 0 和 rank 3，实际每个 rank 一行。`steady_ms` 是稳态 PP 等待（`pp_wait_up + pp_wait_down`，这个例子没开 PP，所以为 0），`mb_fwd_max` 是本窗口最慢的单个前向（ms）。

某个 rank 的窗口不足以下结论时，它被标为「不确定」并给出原因（原因变化时打一条 WARNING），而不是默默当作正常：

```text
[straggler] step=19 rank 3 (rank3_dp3_tp0_pp0) is uncertain: 2 event pairs were still in flight when the window
closed; their time lands in the next window
```

| 原因 | 含义 |
|------|------|
| `dropped_events` / `unread_events` / `profiler_errors` | 该 rank 这个窗口的计时不完整：有事件对因队列满被丢弃、窗口结束时还没读到（会计入下一个窗口）、或分析器在该 rank 上捕获了异常。未读到的 `dp_grad_sync` 会让该 rank 看起来像晚到，所以不下结论。 |
| `no_peers` | 所在 PP stage 没有其他 rank 可比。 |
| `missing_tokens` | 看起来算得慢，但该 stage 没有 token 数，分不清是数据不均还是设备慢。 |
| `no_wait_reference` | 它在等 PP，但同 stage 的其他 rank 都被判为慢、晚到或不确定，没有可以对比的等待时间。 |

「不确定」的窗口两头都不算：它打断候选的连续计数，也不算作告警解除的干净窗口；已有的告警保持不变。

### 指标（与 `perf/*` 同一 step key，进 TensorBoard / WandB / ClearML）

这些标量随训练线程下一次 `log_perf_data` 的 `perf/*` 在同一次 `tracking_utils.log` 里发出，不额外发请求：开启 `--use-metrics-service` 时，每次 log 都是训练线程上的一次同步 HTTP 请求。判定在后台线程上跑，所以一个窗口的标量通常随下一个 rollout 的 `perf/*` 发出；`straggler/window/{first,last}_rollout` 标明它属于哪个窗口。运行的最后一个 rollout 会等最后一个窗口判定完再发。如果后台判定落后，同一步里有多个窗口同时判定完成（正常运行中不会出现），每个窗口照常写 timeline 和日志，但只有其中最新一个上报成功的窗口的标量进指标（同一次 log 每个 key 只能带一个值）；被它覆盖的每个窗口打一条 WARNING `… windows ready at once; only rollouts …-… reach the metrics`，并计一次分析器自身异常；上报失败的窗口只按这次失败计一次（见下文「分析器自身出错时」）。

| 指标 | 含义 |
|------|------|
| `straggler/{fwd,bwd,optim,pp_recv,pp_send,dp_grad_sync,dp_param_gather,lp_fwd,lp_pp_recv,lp_pp_send}/{median_ms,max_ms,max_rank,spread}` | 每段 GPU 时间（ms/step）的全局中位数、最大值；`max_rank` / `spread` 是相对**同 PP stage 其他 rank** 超出最多的那个 rank 及其超出比例（`value/peer_median − 1`）。`lp_*` 是 log-prob / forward-only 阶段的同一组段。`dp_param_gather` 只在关闭 `--overlap-param-gather` 时有值（Megatron 不给 overlap 路径计时）。 |
| `straggler/self/*`、`straggler/self_per_ktok/*` | 自身计算时间（`fwd + bwd + optim`）及其按 token 归一化后的版本；后者与 `tokens/*` 一样，任一 rank token 数为 0 时不输出。 |
| `straggler/wait/{median_ms,max_ms,max_rank,spread}` | 等待别人的时间（`pp_recv + dp_grad_sync + dp_param_gather`）。 |
| `straggler/late/max_ms`、`straggler/late/max_rank`、`straggler/late/peer_idle_ms` | 最晚到达 DP 梯度同步的 rank、晚了多少（按 `bwd + dp_grad_sync` 比较，见下文 `late_arrival`）、同组其他 rank 因此每步空转多久。 |
| `straggler/tokens/{median,max,spread}` | 各 rank 每步 token 数的不均程度：`median` / `max` 为全局值，`spread` 为同 stage 内最大超出；任一 rank token 数为 0 时不输出。 |
| `straggler/pp_stage_imbalance` | 各 PP stage 计算时间中位数的 `max/min`，与是否有慢 rank 无关。 |
| `straggler/{pp_wait_fill,pp_wait_up,pp_wait_down,pp_wait_drain}/{max_ms,max_rank}` | 训练阶段的 `pp_recv` 按非交错 1F1B 的阶段拆开（ms/step）后的最大值及其 rank：`fill` 是填充阶段等上游激活，`up` / `down` 是稳态等上游激活 / 下游梯度，`drain` 是排空阶段等下游梯度。均衡的流水线在稳态不空闲，所以 `up + down` 超出传输时间的部分就是额外的气泡。上游 rank 在一步开头停顿，会把它所在的整条流水线往后推，体现在下游 stage 的 `fill` 上，稳态不变。只在开 PP 时输出；交错调度（VPP）的收发只计入 `pp_recv`。 |
| `straggler/pp_bubble/ratio` | `pp_recv` 占该 rank 已计时训练时间（自身 + 等待 + `pp_send`）的比例，取各 PP stage 中位数里的最大值；只在开 PP 时输出。 |
| `straggler/micro_batch/{fwd_max_ms,fwd_max_rank,fwd_argmax,fwd_argmax_tokens,fwd_min_ms,bwd_max_ms,bwd_max_rank,tokens_max,tokens_min}` | 本窗口内单个训练 micro-batch 的极值（不按步平均）：最慢的前向及其 rank、它是所在 rollout 的第几个前向（从 0 起）和它的 token 数（没拿到时为 −1）、最快的前向、最慢的反向及其 rank，以及单个 micro-batch 的最大 / 最小 token 数（没有 token 数时不输出）。 |
| `straggler/gc/{median_ms,max_ms}` | Python GC 停顿。 |
| `straggler/flagged/count`、`straggler/flagged/rank`（无则 −1）、`straggler/flagged/reason` | 告警状态。reason 编码：0 none、1 slow_device、2 data_imbalance、3 upstream_wait、4 cpu_bound、5 late_arrival。 |
| `straggler/waiting/count` | 本窗口因上游 PP stage 慢而在等的 rank 数（被上游拖慢，不是慢 rank；不计入 `flagged`）。 |
| `straggler/uncertain/count`、`straggler/uncertain/rank`（无则 −1）、`straggler/uncertain/cause` | 「不确定」的 rank 数、第一个这样的 rank 及其原因。原因编码：0 none、1 dropped_events、2 unread_events、3 profiler_errors、4 no_peers、5 missing_tokens、6 no_wait_reference。 |
| `straggler/recovered/count` | 本窗口解除告警的 rank 数。 |
| `straggler/window/first_rollout`、`straggler/window/last_rollout` | 这组标量属于哪个窗口（它们通常随下一个 rollout 发出）。 |
| `straggler/control/level` | 这个窗口的级别：0 paused、1 coarse、2 detail（见下文「运行中切换级别」）。 |
| `straggler/latency/analyzed_ms` | 从窗口结束（gather 之前）到判定完成（告警日志此刻打出）的毫秒数。 |
| `straggler/health/state`、`straggler/health/errors`、`straggler/health/degraded_ranks` | 分析器自身健康：各 rank 中最差的状态（0 active、1 degraded、2 disabled）、本窗口捕获的异常数、处于 degraded 的 rank 数。 |
| `straggler/self_overhead_ms`、`straggler/gather_ms`、`straggler/analyze_ms`、`straggler/dropped_events`、`straggler/unread_events` | 工具自身开销：每步 drain 读事件的 CPU 时间；primary 上每窗口 gather 的耗时（首个窗口另含一次元数据 gather）；后台线程每窗口判定的耗时；因队列满被丢弃的事件对数（正常为 0）；窗口结束时尚未读到的事件对数（正常为 0）。 |

### Timeline

已开 `--timeline-dump-dir` 时（timeline 经 metrics-service adapter 落盘，所以还需要 `--use-metrics-service`），每个窗口会给每个 rank 的每个非零段追加一条 `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}` 事件（`pid` = 2³⁰ + 全局 rank，高于任何真实 pid；`tid` = 段序号），在 Perfetto 里一行一个 rank，肉眼就能看出谁的 `bwd` 条更长、谁的 `dp_grad_sync` 条最短。

### 采样步明细（Perfetto）

设置了 `RELAX_STRAGGLER_TRACE_DIR`，或已开 `--timeline-dump-dir`（默认写到它下面的 `straggler/`）时，从第二个窗口起，每个窗口的第一个 rollout 会放到真实时间轴上：这一步开头同步一次 GPU 并记一个锚点事件，之后每个计时区间相对锚点的起止时间都被保留；窗口结束时各 rank 把这一步交给 primary（只发给 primary 的 `gather_object`），primary 在后台线程写一个 `straggler_trace_rollout{N}.json`（Chrome trace 格式，用 [ui.perfetto.dev](https://ui.perfetto.dev) 打开），只保留最新的 `RELAX_STRAGGLER_TRACE_KEEP`（默认 50）个。这个功能不需要 metrics service。

每个 rank 一行，按 (dp, pp, tp) 排序，同一条流水线的各 stage 相邻。行名里带上该 rank 的实测 PP 等待，以及按它自己实测的前向 / 反向推算的均衡 1F1B 气泡（`(PP − 1) × (平均 fwd + 平均 bwd)` × 优化器步数），两者之差就是额外的气泡；最慢的那个 stage 推算值会大于实测（差为负），这时看其他 stage 的差值。行内按时间嵌套：

| 层 | 内容 |
| --- | --- |
| 第 1 层（PP） | `busy`：连续的计算和通信；`bubble fill` / `up` / `down` / `drain`：PP 等待，按 1F1B 阶段命名，`waits_for` 写明在等上一个还是下一个 stage；`untimed`：超过 0.5 ms、没有任何计时区间覆盖的时间（CPU 侧停顿） |
| 第 2 层（fwd / bwd） | `busy` 里面的 `F{i}` / `B{i}`（`i` 是 micro-batch 序号，前向带 token 数）、`send`、`grad sync`、`param gather`、`optimizer`；log-prob 阶段是 `lp F{i}` 等 |

primary 自己的 CPU 侧计时（`Timer` 事件，例如 `actor_train` 和每个 micro-batch 的 `get_data_batch_*`，去掉调用栈）另起一行 `primary CPU`；采样步结束时仍在计时的不在其中。

| 指标 | 含义 |
|------|------|
| `straggler/trace/ranks`、`straggler/trace/lost` | 交来明细的 rank 数；丢失的区间数（队列满被丢弃，或超过每步 4096 个区间的上限） |
| `straggler/trace/align_spread_ms` | 同一 DP 组的梯度同步结束时刻在各 rank 墙钟上的最大差，也就是跨卡对齐误差 |
| `straggler/trace/bubble_excess_ms_max`、`straggler/trace/bubble_excess_rank` | 实测 PP 等待超出均衡 1F1B 气泡最多的量和它的 rank |

开销与限制：

- 每个窗口多一次 GPU 同步（采样步开头）和一次只发往 primary 的 gather（每个区间 24 B）；生成和写文件在 primary 的后台线程。8 卡 A100、Qwen3-0.6B TP1×PP4×DP2（步时约 1.6 s）实测：训练线程上每个明细窗口多约 2 ms（同步约 0.4 ms、gather 约 1.3 ms、读偏移约 0.3 ms），默认窗口 10 个 rollout 时折合每步约 0.01%。
- 跨卡对齐用的是各 rank 的墙钟（`time.time()`），跨机器时受 NTP 精度影响，误差看 `align_spread_ms`。
- 阶段命名和气泡推算只适用于非交错 1F1B；VPP 和 `combined_1f1b` 下只看第 2 层。窗口只有 1 个 rollout 时，采样步可能在窗口结束时还没读完，这一窗口就没有明细。

### 运行中切换级别

| 级别 | 含义 |
| --- | --- |
| 0 `paused` | 不记录任何计时区间；窗口照常 gather（只带状态），但不做判定，告警的连续计数冻结，标量只有 `straggler/control/level` |
| 1 `coarse` | 上面的窗口统计和判定 |
| 2 `detail` | 再加采样步明细；需要明细的输出目录（见上一节），没有时按 `coarse` 运行 |

启动级别由 `RELAX_STRAGGLER_LEVEL` 给出，默认有明细目录时为 2，否则为 1。开启 `--use-metrics-service` 时，primary 在一个守护线程上每 `RELAX_STRAGGLER_CONTROL_POLL_S`（默认 5）秒向 metrics service 询问一次。训练中途可以在浏览器打开 `http://<head 节点>:<Serve HTTP 端口>/metrics/straggler` 点按钮切换，也可以直接：

```bash
curl -X POST http://<head>:<port>/metrics/straggler/control -H 'Content-Type: application/json' -d '{"level": 0}'
```

请求的级别写进 primary 那一行的 `control` 字段，随下一次窗口 gather 发给所有 rank，所有 rank 从下一个窗口起一起切换，所以最多晚一个窗口加一次轮询间隔生效；页面上能看到当前生效的级别和最近一次轮询的时间。分析器因异常关闭后（见「分析器自身出错时」）不再响应切换；没开 metrics service 时级别固定为启动值。这些接口没有鉴权，和 metrics service 的其他接口一样只应在集群内访问。训练进程自己不开端口：页面和接口挂在 Relax 启动的 Ray Serve HTTP 服务上（head 节点，端口 8000），由 primary 主动轮询；从集群外访问需要自行转发这个端口。

## 覆盖范围

计时复用 Megatron 已有的 22 个计时点，覆盖**调度面**：

| 覆盖 | 内容 |
| --- | --- |
| `fwd` / `bwd` | 每个 micro-batch 的前向 / 反向（整块，不含层内拆分） |
| `pp_recv` / `pp_send` | PP 点对点通信（9 个上游名字并入 2 段，不区分级次；非交错 1F1B 训练时的 4 个接收名字另按阶段和方向计入 `pp_wait_*`） |
| `dp_grad_sync` | DP 梯度同步（4 个上游名字并入 1 段，CP 组内与 DP 组内不可分） |
| `dp_param_gather` | 分布式优化器的参数 all-gather |
| `optim` | 优化器各阶段（6 个上游名字） |

**不覆盖**（上游没有对应计时点，需要新增挂点）：

- **attention、MoE、head/logits、MTP 的内部**。上游 Megatron 在这些模块里没有任何计时点，
  所以现在只能看到它们被包含在 `fwd` / `bwd` 里，无法单独归因。要在这些粒度上定位，
  需要新增挂点——这属于后续工作，当前版本不提供。
- **EP all-to-all**（MoE 专家并行的通信）。它属于「等 peer」类，混在计算段里。
- **真实的通算重叠比例**。`dp_grad_sync` 变小**不能**直接读成「重叠更好」——它是「等待段之和」，
  是代理量，不是重叠率本身。

**已知粒度限制**（详见 `reason` 一节）：同一 TP 组的 rank 在前向 / 反向里互相等，其中一个变慢时
整组都会被报出，分不出是哪一个；所有 rank 同时变慢时（整机降频、共享存储抖动）发现不了。

## reason 怎么读

| reason | 判定依据 | 排查方向 |
|--------|----------|----------|
| `slow_device` | 自身 GPU 时间或 forward 时间显著高于同 PP stage 的其他 rank，token 数正常 | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`、ECC、同机邻居、NVLink 拓扑 |
| `data_imbalance` | 自身时间或 forward 时间高，但按 token 归一化后正常，token 数明显偏多 | `--balance-data`、动态 batch 切分、超长样本 |
| `cpu_bound` | 自身时间或 forward 时间偏高，**且**训练步内的 Python GC 停顿占自身时间 ≥ 5% | `gc.freeze()`、数据处理线程与训练线程抢 GIL、Python 侧 per-token 循环 |
| `upstream_wait` | 自身正常，但 `pp_recv` 显著高于同 stage 中未被判为慢或晚到的 rank，且超出量 ≥ 该 stage 中位自身时间的 5%；因为等 PP 而晚到梯度同步的 rank 也归为这一类 | 看上游 stage 被 flag 的 rank，或 `pp_stage_imbalance`（层划分不均） |
| `late_arrival` | 自身 GPU 时间正常，但它的 `bwd + dp_grad_sync` 显著**短于**同 DP 组其他 rank——一次集合通信对所有人同时结束，最后到的那个 rank 等得最少，其他人的这段时间里都含着等它的时间 | GPU 之间的 CPU 侧空隙：数据取用、同步 I/O / HTTP、GIL、launch gap；对该 rank `py-spy dump --pid <pid>` 最直接 |

`late_arrival` 容易被漏掉：只看 GPU 时间的规则抓不到它。该 rank 的 GPU 时间和大家一样，只是每步晚到梯度同步；常见原因是训练线程上的同步阻塞，例如 primary rank 在训练线程上用同步 HTTP 请求上报指标。

开启 `--overlap-grad-reduce` 时，梯度 reduce-scatter 在 backward 中按 bucket 发出，同 DP 组其他 rank 等最慢 rank 的时间落在各自的 `bwd` 里，`dp_grad_sync` 只剩很短的尾段。所以晚到按 `bwd + dp_grad_sync` 比较（开不开 overlap，这段时间都在梯度同步完成时结束），慢 rank 除了自身时间还单独比较 forward（forward 里没有 DP 集合通信）。在等的 rank 自身时间也会变长，所以 `late_arrival` 告警报的是该 rank 的 forward 与其他 rank 之比，而不是自身时间之比。

## 分析器自身出错时

计时调用、读事件、窗口 gather、判定、日志回调里的异常都被捕获并计数，不会抛进训练循环。某个 rank 累计 3 次异常，或连续 3 个窗口有丢弃的事件对或异常，就请求关闭分析器。这个请求放在该 rank 的窗口向量里（`health` 字段）一起 gather，所以所有 rank 看到同一张表，在同一个窗口一起关闭，不会有 rank 单独退出而让其他 rank 卡在下一次 gather 里。关闭后计时调用直接跳过，但该 rank **仍然参与每次窗口 gather**：它发一个长度为 `NUM_FIELDS` 的全零行，只把 `health` 写成 2，因此集合通信的形状不变，关闭状态也会持续传给还没看到的 rank。

gather 本身失败时只有本 rank 知道。处理方式是**不退出集合通信**：失败被计数，本次窗口照常重置，下一次窗口仍发一行（正常行或上面的空行）。持续失败达到上限后该 rank 本地关闭并开始发空行，其余 rank 在下一个能完成的 gather 里读到 `health = 2`，在同一个窗口一起关闭。首个窗口的元数据汇聚只做一次、不重试：其余 rank 拿到元数据后每个窗口只做数值 gather，失败的 rank 单独重试会和它们的集合通信错位，所以它照常发这一窗口的行并把 `health` 写成 2，所有 rank 在这个窗口一起关闭。元数据汇聚本身只有一次集合通信，每个 rank 发一个 1 KiB 的定长字节行：本地序列化元数据失败、或元数据超长的 rank 照常参与，只是发一个空行，所有 rank 收到后看到同一个缺失的条目，同样在这个窗口一起关闭。为了让「某 rank 进程真的消失」这种发不出任何东西的情形有界，profiler 在安装时给这类 gather 单独建一个 Gloo 组，超时由 `RELAX_STRAGGLER_GATHER_TIMEOUT_S`（默认 60 s）控制，不影响通用 Gloo 组和训练路径的超时。**已知限制**：单次 gather 会因为组超时而阻塞到该超时；如果整个组都失联，所有 rank 都只能各自本地关闭，没有跨 rank 的传播。

## 开销

- 每个 Megatron 计时点一对非阻塞的 `event.record()`（事件从池里复用）；每步末尾用 `event.query()` 惰性读取已完成的事件，不调用 `cuda.synchronize()`，耗时记在 `straggler/self_overhead_ms`。
- 每 `REPORT_INTERVAL` 个 rollout 一次 Gloo `all_gather`，耗时记在 `straggler/gather_ms`。每个 rank 每窗口贡献 272 B（34 个 float64），与 micro-batch 数无关；由于是 `all_gather`，每个 rank 都会拿到 `N × 272 B` 的完整表，全作业每窗口合计接收 `N × (N−1) × 272 B`（不含 Gloo 协议开销）。首个窗口另有一次元数据汇聚，每个 rank 1 KiB。
- 判定在 primary 的后台线程上跑，耗时记在 `straggler/analyze_ms`，复杂度 O(n log n)（n = 同 stage rank 数）：8 个 rank 每窗口几毫秒，2048 个 rank 的 CPU 基准约 35 ms。
- 在 8×A100、Qwen3-0.6B DP8（step 约 1 s）上，逐项相加的最坏上界约 1.9 ms/step（约 0.19%），端到端开 / 关对比与 0 无法区分；开启后不新增同步调用，NCCL 与计算 kernel 的重叠比例不变。测量方法和各组数据见 [RFC #362](https://github.com/redai-studio/Relax/issues/362)。

## 告警时延

判定只在窗口关闭时执行，并且要连续 `PERSIST_WINDOWS` 个窗口成为候选才告警。默认参数（`REPORT_INTERVAL=10`、`PERSIST_WINDOWS=3`）下，从开始变慢到打出告警约 21–30 个 rollout；如果变慢开始得太晚、被所在窗口的累计值稀释而没过门限，还要再多一个窗口。调小这两个参数能更早告警，代价是更容易被一次性抖动触发。

告警日志在判定完成时立即打出（`straggler/latency/analyzed_ms`，通常几毫秒）；标量随下一次 `perf/*` 上报，再由所用的输出后端（metrics service、W&B、TensorBoard 或 ClearML）展示。

## 实现位置

- `relax/utils/straggler/timers.py`：替代 Megatron `config.timers` 的非阻塞计时器（Megatron 自带的 `Timer.start/stop` 会 `cuda.synchronize()`，所以不开 profiler 时 Relax 把它设为 `None`）。
- `relax/utils/straggler/stats.py`：统计向量布局、Megatron timer 名 → 段的映射。
- `relax/utils/straggler/collector.py`：每 rank 一个，事件池、惰性读取、GC 回调、每个窗口的 gather；不依赖 torch，事件、gather 等由外部传入。
- `relax/utils/straggler/worker.py`：primary 上按顺序执行判定的后台线程。
- `relax/utils/straggler/detector.py`：纯 numpy 的窗口分析与判定（含「不确定」与告警解除），可在无 GPU 环境单测。
- `relax/utils/straggler/health.py`：分析器自身的健康状态（active → degraded → disabled）。
- `relax/utils/straggler/reporter.py`：指标 / timeline 输出（训练线程）与日志输出（后台线程）。
- `relax/utils/straggler/trace.py`：采样步明细：把各 rank 的区间排成嵌套切片，写出 Perfetto 能打开的 JSON（在 primary 的后台线程）。
- `relax/utils/straggler/control.py`：运行中切换级别：metrics service 一侧的状态、primary 上的轮询线程和控制页。
- `relax/utils/straggler/runtime.py`：唯一接触加速器（经 `relax.utils.device`，CUDA / NPU 通用）、进程组和 Megatron 并行状态的模块；`StragglerProfiler` 负责每步在 `log_perf_data` 前后的上报。
- 接线：`relax/backends/megatron/model.py`（`config.timers = straggler_timers(...)` 三处）、`relax/backends/megatron/actor.py`（`install_straggler_profiler`，每步 `_log_perf_data`）、`relax/utils/metrics/service.py`（控制页和 `/straggler/*` 接口）。
