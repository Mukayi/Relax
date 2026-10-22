# Straggler（慢 rank）分析

Relax 的 Megatron 后端内置一个低开销的慢 rank 分析器，可以在生产训练中一直开着。每个训练 rank 在计算流上用 CUDA event 给每一步的 forward、backward、optimizer、梯度同步、参数 all-gather 等段计时，两个 event 之间 CPU 侧的 launch 空隙也算在内。每隔 K 个 rollout，所有 rank 把一个定长的统计向量 Gloo all-gather 到 primary rank；primary 的后台线程判断有没有 rank 拖慢整组、原因是什么，并把结论写进指标、timeline 和日志。

它不是 `torch.profiler`：不采集 kernel，不调用 `cuda.synchronize()`，不影响通信与计算的重叠。分析器自身的异常不会抛进训练循环：异常会被计数，多次出错后所有 rank 一起关闭分析器。

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
| `RELAX_STRAGGLER_PROFILER` | bool | false | 总开关。 |
| `RELAX_STRAGGLER_REPORT_INTERVAL` | int | 10 | 每多少个 rollout 汇总并判定一次，这段时间称为一个窗口。一个 rollout 可能包含多个 optimizer step，下文的 ms/step 都是每个 rollout 的累计值。 |
| `RELAX_STRAGGLER_Z_THRESHOLD` | float | 3.0 | 判为异常的 robust z 分数门限（基于组内中位数和 MAD）。 |
| `RELAX_STRAGGLER_REL_THRESHOLD` | float | 0.10 | 判为异常所需的最小超出比例（相对组内中位数）。 |
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | 连续多少个窗口异常才告警，用来过滤 checkpoint、GC 峰值等一次性抖动；对所有 reason 都适用，必须 ≥ 1。 |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | 已告警的 rank 要连续多少个正常窗口才解除告警，避免告警反复开关；「不确定」的窗口不算正常；必须 ≥ 1。 |
| `RELAX_STRAGGLER_GATHER_TIMEOUT_S` | float | 60.0 | 分析器专用 Gloo 组的超时（秒）。一次 gather 正常只需几毫秒，这个值只决定某个 rank 进程已经消失时，其余 rank 最多等多久。 |
| `RELAX_STRAGGLER_TRACE_DIR` | str | 空 | 采样步 trace 的输出目录；不设时用 `--timeline-dump-dir` 下的 `straggler/`，两者都没有就不写 trace。 |
| `RELAX_STRAGGLER_TRACE_KEEP` | int | 50 | 保留最新的多少个 trace 文件。 |
| `RELAX_STRAGGLER_LEVEL` | int | −1 | 启动级别（见「运行中切换级别」）；−1 表示有 trace 目录时为 2，否则为 1。 |
| `RELAX_STRAGGLER_CONTROL_POLL_S` | float | 5.0 | primary 向 metrics service 查询级别的间隔（秒）。 |

开启后，每个 actor 进程启动时打印一行 `Straggler profiler enabled: report_interval=… primary=… level=… control=… trace_dir=… meta=RankMeta(rank=…, dp=…, tp=…, pp=…, cp=…, ep=…, host=…, device=…, primary=…)`，其中 `trace_dir=-` 表示不写 trace。critic、reference 和 `actor_fwd` 不跑训练步，不做分析。

## 看什么

### 日志（primary rank）

日志由 primary 的分析线程在判定结束后立即打印，`step` 是该窗口最后一个 rollout 的 step。没有告警的窗口打印一行 INFO：

```text
[straggler] step=29 no straggler (uncertain 0); self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

有告警时，rank 的 reason 发生变化的窗口打印一条 WARNING，之后每个仍有告警的窗口打印一张逐 rank 的 INFO 表。任何 reason 都要连续 `PERSIST_WINDOWS` 个窗口才成立，连续 `RECOVER_WINDOWS` 个正常窗口才解除；解除时再打印一条 WARNING，例如 `rank 5 (…) recovered from slow_device after 2 clean windows`：

```text
[straggler] step=29 rank 3 (rank3_dp3_tp0_pp0, host=…, gpu=3) reaches the DP grad-sync 197.4 ms/step
after its DP peers (peers idle 203.3 ms/step = 22% of their compute, z=41.0) for 3 windows;
its own GPU forward is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.56x -> late_arrival (host-side stall)
[straggler] step=29 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | steady_ms | mb_fwd_max | reason       | uncertain
    0 | rank0_dp0_tp0_pp0 | …    |      0 |   933.7 |     6.4 |     6.1 | 14802.1 |        63.1 |   1.8 |       0.0 |      157.8 | none         |
    3 | rank3_dp3_tp0_pp0 | …    |      3 |   737.1 |     5.9 |   197.4 | 14592.7 |        50.5 |   1.8 |       0.0 |      148.0 | late_arrival |
```

表里只列出 rank 0 和 rank 3，实际每个 rank 一行。`steady_ms` 是稳态 PP 等待（`pp_wait_up + pp_wait_down`，这个例子没有 PP，所以为 0），`mb_fwd_max` 是本窗口最慢的单个前向（ms）。

如果某个 rank 本窗口的数据不足以下结论，它会被标为「不确定」并给出原因（原因变化时打印一条 WARNING），而不是被当作正常：

```text
[straggler] step=19 rank 3 (rank3_dp3_tp0_pp0) is uncertain: 2 event pairs were still in flight when the window
closed; their time lands in the next window
```

| 原因 | 含义 |
|------|------|
| `dropped_events` / `unread_events` / `profiler_errors` | 该 rank 本窗口的计时不完整：有计时因队列满被丢弃，有计时在窗口结束时还没读到（计入下一个窗口），或分析器在该 rank 上捕获了异常。没读到的 `dp_grad_sync` 会让它看起来像晚到，所以不下结论。 |
| `no_peers` | 所在 PP stage 没有其他 rank 可比。 |
| `missing_tokens` | 看起来算得慢，但该 stage 没有 token 数，分不清是数据不均还是设备慢。 |
| `no_wait_reference` | 它在等 PP，但同 stage 的其他 rank 都被判为慢、晚到或不确定，没有可对比的等待时间。 |

「不确定」的窗口会打断告警的连续计数，也不算解除告警所需的正常窗口；已有的告警保持不变。

### 指标

这些标量和 `perf/*` 使用同一个 step，写入 TensorBoard / WandB / ClearML。它们和训练线程下一次 `log_perf_data` 的 `perf/*` 在同一次 `tracking_utils.log` 里发出，不单独发请求（开启 `--use-metrics-service` 时，每次 log 都是训练线程上的一次同步 HTTP 请求）。因为判定在后台线程上进行，一个窗口的标量通常随下一个 rollout 的 `perf/*` 发出，由 `straggler/window/{first,last}_rollout` 标明所属窗口；最后一个 rollout 会等最后一个窗口判定完再发。

如果后台判定落后，同一步里会有多个窗口判定完成（正常运行不会出现）。这时每个窗口照常写 timeline 和日志，但一次 log 里每个 key 只有一个值，所以只有最新一个上报成功的窗口的标量进入指标。被覆盖的每个窗口打印一条 WARNING `… windows ready at once; only rollouts …-… reach the metrics`，并计一次分析器异常；上报失败的窗口只按这次失败计一次（见「分析器自身出错时」）。

| 指标 | 含义 |
|------|------|
| `straggler/{fwd,bwd,optim,pp_recv,pp_send,dp_grad_sync,dp_param_gather,lp_fwd,lp_pp_recv,lp_pp_send}/{median_ms,max_ms,max_rank,spread}` | 每段 GPU 时间（ms/step）的全局中位数和最大值；`max_rank` / `spread` 是相对**同 PP stage 其他 rank** 超出最多的 rank 及超出比例（`value/peer_median − 1`）。`lp_*` 是 log-prob / forward-only 阶段的同名段。`dp_param_gather` 只在关闭 `--overlap-param-gather` 时有值（Megatron 不给 overlap 路径计时）。 |
| `straggler/self/*`、`straggler/self_per_ktok/*` | 自身计算时间（`fwd + bwd + optim`）及按 token 归一化的值；后者在任一 rank 的 token 数为 0 时不输出。 |
| `straggler/wait/{median_ms,max_ms,max_rank,spread}` | 等待其他 rank 的时间（`pp_recv + dp_grad_sync + dp_param_gather`）。 |
| `straggler/late/max_ms`、`straggler/late/max_rank`、`straggler/late/peer_idle_ms` | 最晚到达 DP 梯度同步的 rank、晚了多少（按 `bwd + dp_grad_sync` 比较，见 `late_arrival`），以及同组其他 rank 因此每步空等的时间。 |
| `straggler/tokens/{median,max,spread}` | 各 rank 每步 token 数的不均衡程度：`median` / `max` 是全局值，`spread` 是同 stage 内最大的超出比例；任一 rank 的 token 数为 0 时不输出。 |
| `straggler/pp_stage_imbalance` | 各 PP stage 计算时间中位数的 `max/min`，与是否有慢 rank 无关。 |
| `straggler/{pp_wait_fill,pp_wait_up,pp_wait_down,pp_wait_drain}/{max_ms,max_rank}` | 训练阶段的 `pp_recv` 按非交错 1F1B 的阶段拆开（ms/step）后的最大值和对应 rank：`fill` 是填充阶段等上游激活，`up` / `down` 是稳态阶段等上游激活 / 下游梯度，`drain` 是排空阶段等下游梯度。均衡的流水线在稳态不空等，所以 `up + down` 超出传输时间的部分就是多出来的气泡。上游 rank 在一步开头停顿时，整条流水线被往后推，体现在下游 stage 的 `fill` 上，稳态不变。只在开 PP 时输出；交错调度（VPP）的收发只计入 `pp_recv`。 |
| `straggler/pp_bubble/ratio` | `pp_recv` 占该 rank 已计时训练时间（自身计算 + 等待 + `pp_send`）的比例，取各 PP stage 中位数的最大值；只在开 PP 时输出。 |
| `straggler/micro_batch/{fwd_max_ms,fwd_max_rank,fwd_argmax,fwd_argmax_tokens,fwd_min_ms,bwd_max_ms,bwd_max_rank,tokens_max,tokens_min}` | 本窗口内单个训练 micro-batch 的极值（不按步平均）：最慢的前向及其 rank、它在所在 rollout 里的 micro-batch 序号（从 0 起；VPP 与 `combined_1f1b` 同时开启时为 −1）及其 token 数（拿不到时为 −1）、最快的前向、最慢的反向及其 rank，以及单个 micro-batch 的最大 / 最小 token 数（没有 token 数时不输出）。 |
| `straggler/gc/{median_ms,max_ms}` | Python GC 停顿。 |
| `straggler/flagged/count`、`straggler/flagged/rank`（无则 −1）、`straggler/flagged/reason` | 告警状态。reason 编码：0 none、1 slow_device、2 data_imbalance、3 upstream_wait、4 cpu_bound、5 late_arrival。 |
| `straggler/waiting/count` | 本窗口因上游 PP stage 慢而在等待的 rank 数。它们是被拖慢的，不是慢 rank，不计入 `flagged`。 |
| `straggler/uncertain/count`、`straggler/uncertain/rank`（无则 −1）、`straggler/uncertain/cause` | 「不确定」的 rank 数、第一个这样的 rank 及其原因。原因编码：0 none、1 dropped_events、2 unread_events、3 profiler_errors、4 no_peers、5 missing_tokens、6 no_wait_reference。 |
| `straggler/recovered/count` | 本窗口解除告警的 rank 数。 |
| `straggler/window/first_rollout`、`straggler/window/last_rollout` | 这组标量所属的窗口。 |
| `straggler/control/level` | 本窗口的级别：0 paused、1 coarse、2 detail、3 modules（见「运行中切换级别」）。 |
| `straggler/latency/analyzed_ms` | 从窗口结束（gather 之前）到判定完成（即告警日志打印时）的毫秒数。 |
| `straggler/health/state`、`straggler/health/errors`、`straggler/health/degraded_ranks` | 分析器自身的健康状态：各 rank 中最差的状态（0 active、1 degraded、2 disabled）、本窗口捕获的异常数、处于 degraded 的 rank 数。 |
| `straggler/self_overhead_ms`、`straggler/gather_ms`、`straggler/analyze_ms`、`straggler/dropped_events`、`straggler/unread_events` | 分析器自身的开销：每步读取计时的 CPU 时间；primary 上每个窗口 gather 的耗时（首个窗口还包括一次元数据 gather）；后台线程每个窗口的判定耗时；因队列满被丢弃的计时数；窗口结束时还没读到的计时数（后两项正常为 0）。 |

### Timeline

开启 `--timeline-dump-dir` 时（timeline 由 metrics service 的 adapter 写盘，所以还需要 `--use-metrics-service`），每个窗口为每个 rank 的每个非零段追加一条事件 `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}`（`pid` = 2³⁰ + 全局 rank，高于任何真实 pid；`tid` = 段序号）。在 Perfetto 里每个 rank 一行，一眼就能看出谁的 `bwd` 更长、谁的 `dp_grad_sync` 最短。

### 采样步 trace（Perfetto）

有 trace 目录时（见「开启」中的 `RELAX_STRAGGLER_TRACE_DIR`），从第二个窗口起，每个窗口的第一个 rollout（下称采样步）会被记录在真实时间轴上：这一步开头同步一次 GPU 并记录一个锚点事件，这一步里的每次计时都保留相对锚点的起止时间。窗口结束时，各 rank 把采样步的计时发给 primary（只发往 primary 的 `gather_object`），primary 在后台线程写出 `straggler_trace_rollout{N}.json`。有 trace 目录时，所有 rank 每个窗口都参加这次 gather，没有采样步的 rank（第一个窗口、`paused` / `coarse` 级别、分析器关闭后）发送一个空条目，所以某个 rank 本地出错不会让各 rank 的集合通信错开。文件是 Chrome trace 格式，用 [ui.perfetto.dev](https://ui.perfetto.dev) 打开。这个功能不需要 metrics service。

每个 rank 一行，按 (dp, pp, tp) 排序，同一条流水线的各 stage 相邻。行名包含该 rank 实测的 PP 等待，以及用它自己实测的前向、反向耗时推算的均衡 1F1B 气泡（`(PP − 1) × (平均 fwd + 平均 bwd)` × 优化器步数），两者之差就是多出来的气泡。最慢的 stage 推算值会大于实测值（差为负），这时看其他 stage 的差值。行内的切片按时间嵌套：

| 层 | 内容 |
| --- | --- |
| 第 1 层（PP） | `busy`：连续的计算和通信；`bubble fill` / `up` / `down` / `drain`：PP 等待，按 1F1B 阶段命名，`waits_for` 写明在等上一个还是下一个 stage；`untimed`：超过 0.5 ms 且没有任何计时覆盖的时间（CPU 侧停顿） |
| 第 2 层（fwd / bwd） | `busy` 里的 `F{i}` / `B{i}`（`i` 是 micro-batch 序号，VPP 下另带模型块，记为 `F{i} chunk {c}`；前向带 token 数）、`send`、`grad sync`（DP 梯度同步，即 Megatron 的 `all-grads-sync`；其他梯度同步用 Megatron 的计时器名，例如 `embedding-grads-all-reduce`）、`param gather`、`optimizer`；log-prob 阶段为 `lp F{i}` 等 |
| 第 3 层（模块） | 只在 `modules` 级别出现：`F{i}` / `B{i}` 里每个 transformer 层的 attention 和 MLP（MoE 层记为 `moe`）的前向、反向，例如 `attn fwd L3`、`moe bwd L3`，参数带 `layer` 和 `micro_batch`（VPP 下另带 `model_chunk`）；MTP 层的层号从 1 重新开始，记为 `attn fwd MTP L1` 等，参数另带 `mtp`；出现在 `B{i}` 里的前向是激活重计算，记为 `… (recompute)`；log-prob 阶段为 `lp attn fwd L3` 等 |

primary 自身的 CPU 侧计时（`Timer` 事件，例如 `actor_train` 和每个 micro-batch 的 `get_data_batch_*`，不含调用栈）单独一行 `primary CPU`；采样步结束时还没结束的计时不包括在内。

| 指标 | 含义 |
|------|------|
| `straggler/trace/ranks`、`straggler/trace/lost` | 提供了 trace 的 rank 数；丢失的计时数（队列满被丢弃，或超出每步 4096 条普通计时、8192 条模块计时的上限；模块计时单独排队，最多占队列容量的一半，队列满时先被挤掉；被挤掉的模块计时只计在这里，不计入 `dropped_events`） |
| `straggler/trace/module_brackets` | trace 中模块计时的条数，各 rank 求和（仅 `modules` 级别） |
| `straggler/trace/align_spread_ms` | 同一 DP 组 DP 梯度同步（`all-grads-sync`）的结束时刻在各 rank 墙钟上的最大差值，即跨卡对齐误差 |
| `straggler/trace/bubble_excess_ms_max`、`straggler/trace/bubble_excess_rank` | 实测 PP 等待超出均衡 1F1B 气泡的最大值及其 rank |

开销与限制：

- 每个窗口多一次 GPU 同步（采样步开头）和一次只发往 primary 的 gather（每条计时 24 B）；生成和写文件在 primary 的后台线程完成。在 8 卡 A100、Qwen3-0.6B TP1×PP4×DP2（步时约 1.6 s）上实测，训练线程每个窗口多约 2 ms（同步约 0.4 ms、gather 约 1.3 ms、读取时间偏移约 0.3 ms），按默认 10 个 rollout 一个窗口折算，每步约 0.01%。有 trace 目录、但级别为 `coarse` 或 `paused` 时，每个窗口仍有这次 gather，每个 rank 只发一个空条目。
- `modules` 级别只在采样步挂 hook，所有 rank 相同。同一配置下实测（每个 stage 7 层，每个 rank 每步约 250 条模块计时）：采样步的前向长 3–10%（小模型受 kernel 发射限制，hook 的 CPU 开销会显现出来），反向不变，步时仍在步间波动范围内；每个窗口读取模块计时多约 3.5 ms。窗口统计里包含这些略长的前向。CUDA graph 捕获期间不计时，被捕获的层没有模块切片。
- 跨卡对齐依赖各 rank 的墙钟（`time.time()`），跨机器时受 NTP 精度影响，误差见 `align_spread_ms`。
- 阶段命名和气泡推算只适用于非交错 1F1B。VPP 下的 micro-batch 序号和模型块按 Megatron 的交错调度表（`get_schedule_table`）由调用顺序换算：一次 forward-backward 里的第 k 次前向对应表中第 k 项，第 k 次反向对应同一项、模型块倒序。`combined_1f1b` 把前向和反向合并执行，VPP 下不编号。窗口只有 1 个 rollout 时，采样步可能在窗口结束时还没读完，这个窗口就没有 trace。

### 运行中切换级别

| 级别 | 含义 |
| --- | --- |
| 0 `paused` | 不计时；窗口照常 gather（只带状态），但不做判定，告警的连续计数冻结，标量只有 `straggler/control/level` |
| 1 `coarse` | 上文的窗口统计和判定 |
| 2 `detail` | 再加采样步 trace；需要 trace 目录，没有时按 `coarse` 运行 |
| 3 `modules` | 在 trace 里再加第 3 层。hook 只在采样步存在，只有这一步有额外开销；模块计时只写进 trace，不进窗口统计，也不参与判定。和 `detail` 一样需要 trace 目录，且不会是默认级别 |

启动级别由 `RELAX_STRAGGLER_LEVEL` 决定，默认有 trace 目录时为 2，否则为 1；3 只能显式设置。开启 `--use-metrics-service` 时，primary 的一个守护线程每 `RELAX_STRAGGLER_CONTROL_POLL_S` 秒向 metrics service 查询一次级别。训练中途可以在浏览器打开 `http://<head 节点>:8000/metrics/straggler` 点按钮切换，也可以直接发请求：

```bash
curl -X POST http://<head>:8000/metrics/straggler/control -H 'Content-Type: application/json' -d '{"level": 0}'
```

请求的级别写进 primary 统计向量的 `control` 字段，随下一次窗口 gather 发给所有 rank，所有 rank 从下一个窗口起一起切换，因此最多晚一个窗口加一次轮询间隔生效。页面上显示当前生效的级别和最近一次轮询的时间。分析器因异常关闭后不再响应切换（见「分析器自身出错时」）；没开 metrics service 时级别固定为启动值。

训练进程本身不监听端口：页面和接口由 Relax 启动的 Ray Serve HTTP 服务提供（head 节点的 8000 端口），primary 主动轮询它。这些接口没有鉴权，不要暴露到集群外；需要在本地浏览器打开时，自行转发 8000 端口。

## 覆盖范围

计时复用 Megatron 已有的 22 个计时点，覆盖以下各段：

| 段 | 内容 |
| --- | --- |
| `fwd` / `bwd` | 每个 micro-batch 的前向 / 反向（整体计时，不拆分到层） |
| `pp_recv` / `pp_send` | PP 点对点通信（Megatron 的 9 个计时器名合并为 2 段，不区分是哪两个 rank 之间；非交错 1F1B 训练时，4 个接收计时器另按阶段和方向计入 `pp_wait_*`） |
| `dp_grad_sync` | DP 梯度同步（4 个计时器名合并为 1 段，CP 组内和 DP 组内的部分分不开；采样步 trace 里 4 个计时器分开显示） |
| `dp_param_gather` | 分布式优化器的参数 all-gather |
| `optim` | 优化器各阶段（6 个计时器名） |

**不覆盖**（Megatron 没有对应的计时点）：

- **attention、MoE、head/logits、MTP 的内部**。窗口统计里它们都包含在 `fwd` / `bwd` 中。`modules` 级别把每层的 attention 和 MLP / MoE 块作为整体计时，但只写进采样步 trace；块内部（例如 MoE 的 dispatch / experts / combine）和输出层仍然拆不开。
- **EP all-to-all**（MoE 专家并行的通信）。它本质上是等待其他 rank，但计入了计算时间。
- **真实的通信计算重叠比例**。`dp_grad_sync` 变小**不能**直接理解为重叠变好：它是等待时间之和，只能间接反映重叠。

**已知的粒度限制**（详见「reason 怎么读」）：同一 TP 组的 rank 在前向 / 反向里互相等待，其中一个变慢时整组都会被报出，分不出是哪一个；所有 rank 同时变慢时（整机降频、共享存储抖动）无法发现。

## reason 怎么读

| reason | 判定依据 | 排查方向 |
|--------|----------|----------|
| `slow_device` | 自身计算时间或前向时间显著高于同 PP stage 的其他 rank，token 数正常 | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`、ECC、同机邻居、NVLink 拓扑 |
| `data_imbalance` | 自身计算时间或前向时间偏高，但按 token 归一化后正常，token 数明显偏多 | `--balance-data`、动态 batch 切分、超长样本 |
| `cpu_bound` | 自身计算时间或前向时间偏高，**且**训练步内的 Python GC 停顿占自身计算时间 ≥ 5% | `gc.freeze()`、数据处理线程与训练线程争抢 GIL、Python 侧逐 token 的循环 |
| `upstream_wait` | 自身计算时间正常，但 `pp_recv` 显著高于同 stage 中没被判为慢或晚到的 rank，且超出量 ≥ 该 stage 自身计算时间中位数的 5%；因为等 PP 而晚到梯度同步的 rank 也归为这一类 | 看上游 stage 被告警的 rank，或 `pp_stage_imbalance`（层划分不均） |
| `late_arrival` | 自身 GPU 时间正常，但 `bwd + dp_grad_sync` 显著**短于**同 DP 组其他 rank：集合通信对所有 rank 同时结束，最后到的 rank 等得最少，其他 rank 的这段时间里都包含了等它的时间 | GPU 之间的 CPU 侧空隙：取数据、同步 I/O / HTTP、GIL、launch 间隙；对该 rank 执行 `py-spy dump --pid <pid>` 最直接 |

`late_arrival` 容易被漏掉，只看 GPU 时间的规则发现不了它：该 rank 的 GPU 时间和其他 rank 一样，只是每步都晚到梯度同步。常见原因是训练线程上的同步阻塞，例如 primary rank 在训练线程上用同步 HTTP 请求上报指标。

开启 `--overlap-grad-reduce` 时，梯度 reduce-scatter 在 backward 中按 bucket 发出，同 DP 组其他 rank 等最慢 rank 的时间落在各自的 `bwd` 里，`dp_grad_sync` 只剩很短的尾段。因此晚到按 `bwd + dp_grad_sync` 比较，无论是否开启 overlap，这段时间都在梯度同步完成时结束。判断慢 rank 时，除了自身计算时间还单独比较前向时间，因为前向里没有 DP 集合通信。在等待的 rank 自身计算时间也会变长，所以 `late_arrival` 告警给出的是该 rank 的前向时间与其他 rank 之比，而不是自身计算时间之比。

## 分析器自身出错时

计时调用、读取计时、窗口 gather、判定和日志回调中的异常都会被捕获并计数，不会抛进训练循环。某个 rank 累计 3 次异常，或连续 3 个窗口有被丢弃的计时或异常，就会请求关闭分析器。这个请求写在该 rank 统计向量的 `health` 字段里随 gather 发出，所以所有 rank 看到同一张表，在同一个窗口一起关闭，不会出现某个 rank 单独退出、其他 rank 卡在下一次 gather 的情况。关闭后计时调用直接跳过，但该 rank **仍然参与每次窗口 gather**：它发送一个长度为 `NUM_FIELDS`、只有 `health` 为 2 的全零向量，所以集合通信的形状不变，关闭状态也会持续传给还没收到的 rank。

gather 本身失败时只有失败的 rank 知道。它**不退出集合通信**：失败被计数，本窗口照常重置，下一个窗口照常发送统计向量（或关闭后的全零向量）。连续失败达到上限后，该 rank 在本地关闭并开始发送全零向量，其余 rank 在下一次成功的 gather 中读到 `health = 2`，在同一个窗口一起关闭。

本地打包统计向量失败的 rank 照常参加 gather，只是发送一行 NaN，所有 rank 都跳过这个窗口。gather 成功、但某个 rank 在本地读表时出错，只有它跳过这个窗口：有 trace 目录时它照常参加随后的采样 gather（见「采样步 trace」），错过的级别切换在下一个窗口补上（primary 每个窗口都重复发送请求的级别）。两种情况都按一次异常计数。

首个窗口的元数据汇聚只做一次、不重试：其余 rank 拿到元数据后每个窗口只做数值 gather，失败的 rank 单独重试会与它们的集合通信错位。所以失败的 rank 照常发送这个窗口的统计向量并把 `health` 设为 2，所有 rank 在这个窗口一起关闭。元数据汇聚只有一次集合通信，每个 rank 发送一个 1 KiB 的定长字节串；本地序列化失败或元数据超长的 rank 照常参与，只是发送一个空条目，所有 rank 都看到同一个缺失的条目，同样在这个窗口一起关闭。

为了在某个 rank 进程彻底消失、什么都发不出来时也能结束等待，分析器在安装时为这些 gather 单独创建一个 Gloo 组，超时由 `RELAX_STRAGGLER_GATHER_TIMEOUT_S` 控制，不影响通用 Gloo 组和训练路径的超时。**已知限制**：这种情况下单次 gather 会一直阻塞到超时；如果整个组都失联，各 rank 只能各自在本地关闭，关闭状态不会在 rank 之间传播。

## 开销

- 每个 Megatron 计时点一对非阻塞的 `event.record()`（event 从池中复用）；每步结束时用 `event.query()` 读取已完成的计时，不调用 `cuda.synchronize()`，耗时计入 `straggler/self_overhead_ms`。
- 每 `REPORT_INTERVAL` 个 rollout 做一次 Gloo `all_gather`，耗时计入 `straggler/gather_ms`。每个 rank 每个窗口发送 272 B（34 个 float64），与 micro-batch 数无关；由于是 all_gather，每个 rank 都会收到 `N × 272 B` 的完整表，整个作业每个窗口共接收 `N × (N−1) × 272 B`（不含 Gloo 协议开销）。首个窗口还有一次元数据汇聚，每个 rank 1 KiB。
- 判定在 primary 的后台线程上进行，耗时计入 `straggler/analyze_ms`，复杂度 O(n log n)（n 为同 stage 的 rank 数）：8 个 rank 每个窗口几毫秒，2048 个 rank 的 CPU 基准测试约 35 ms。
- 在 8×A100、Qwen3-0.6B DP8（每步约 1 s）上，各项开销相加的最坏上界约 1.9 ms/step（约 0.19%），端到端开 / 关对比测不出差异；分析器不新增同步调用，不改变 NCCL 与计算 kernel 的重叠。测量方法和数据见 [RFC #362](https://github.com/redai-studio/Relax/issues/362)。
- 采样步 trace 和 `modules` 级别的开销见「采样步 trace」一节。

## 告警时延

判定只在窗口结束时进行，并且要连续 `PERSIST_WINDOWS` 个窗口异常才告警。默认参数（`REPORT_INTERVAL=10`、`PERSIST_WINDOWS=3`）下，从开始变慢到告警约 21–30 个 rollout；如果变慢发生在窗口后段，被窗口累计值稀释而没超过门限，还要再多一个窗口。调小这两个参数可以更早告警，代价是更容易被一次性抖动触发。

告警日志在判定完成时打印（`straggler/latency/analyzed_ms`，通常几毫秒）；标量随下一次 `perf/*` 上报，再由所用的输出后端（metrics service、W&B、TensorBoard 或 ClearML）展示。

## 实现位置

- `relax/utils/straggler/timers.py`：替代 Megatron `config.timers` 的非阻塞计时器。Megatron 自带的 `Timer.start/stop` 会调用 `cuda.synchronize()`，所以不开分析器时 Relax 把 `config.timers` 设为 `None`。
- `relax/utils/straggler/stats.py`：统计向量的布局，以及 Megatron 计时器名到段的映射。
- `relax/utils/straggler/collector.py`：每个 rank 一个，负责 event 池、惰性读取、GC 回调和每个窗口的 gather；不依赖 torch，event 和 gather 由外部传入。
- `relax/utils/straggler/worker.py`：primary 上按顺序执行判定的后台线程。
- `relax/utils/straggler/detector.py`：纯 numpy 的窗口分析和判定（包括「不确定」和解除告警），可以在没有 GPU 的环境中单测。
- `relax/utils/straggler/health.py`：分析器自身的健康状态（active → degraded → disabled）。
- `relax/utils/straggler/reporter.py`：指标 / timeline 输出（训练线程）和日志输出（后台线程）。
- `relax/utils/straggler/trace.py`：把各 rank 采样步的计时排成嵌套切片，写出 Perfetto JSON（在 primary 的后台线程）。
- `relax/utils/straggler/control.py`：运行中切换级别：metrics service 侧的状态、primary 的轮询线程和控制页。
- `relax/utils/straggler/modules.py`：`modules` 级别在采样步挂到每层 attention 和 MLP / MoE 块上的前向 hook 和梯度 hook。
- `relax/utils/straggler/runtime.py`：唯一接触加速器（通过 `relax.utils.device`，CUDA / NPU 通用）、进程组和 Megatron 并行状态的模块；`StragglerProfiler` 负责每步在 `log_perf_data` 前后上报。
- 接入点：`relax/backends/megatron/model.py`（三处 `config.timers = straggler_timers(...)`）、`relax/backends/megatron/actor.py`（`install_straggler_profiler`、`register_straggler_model`，以及每步的 `_log_perf_data`）、`relax/utils/metrics/service.py`（控制页和 `/straggler/*` 接口）。
