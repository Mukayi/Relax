#!/usr/bin/env python3
"""#463 材料的图（中文标注），数据来自三个实验写出的 CSV。

用法：figs_463.py <数据目录> <输出目录>
数据目录下是 a_out/ 和 logs/。需要一种中文字体，例如 Noto Sans CJK SC 或文泉驿微米黑。
"""

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402


CJK_FONTS = ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "PingFang SC",
             "Microsoft YaHei", "SimHei"]
BEFORE, AFTER = "#c0392b", "#2471a3"
WAIT, OTHER = "#e67e22", "#b8c4cf"
COMBOS = [("proxy", "before"), ("proxy", "after"), ("noproxy", "before"), ("noproxy", "after")]
LABEL = {("proxy", "before"): "修复前，有代理变量", ("proxy", "after"): "修复后，有代理变量",
         ("noproxy", "before"): "修复前，无代理变量", ("noproxy", "after"): "修复后，无代理变量"}
B_LEVELS = ["tb", "tb+timeline", "tb+100ms", "tb+300ms", "tb+1000ms"]
B_NAMES = ["只有\nTensorBoard", "TensorBoard\n+ timeline", "TensorBoard\n+ 100 ms", "TensorBoard\n+ 300 ms",
           "TensorBoard\n+ 1000 ms"]
FIRST = 20


def use_cjk_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in names:
            plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [name, "DejaVu Sans"],
                                 "axes.unicode_minus": False})
            return
    sys.exit("找不到中文字体，请安装其中一种：" + "、".join(CJK_FONTS))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def color(arm, cond="proxy"):
    return (BEFORE if arm == "before" else AFTER), (1.0 if cond == "proxy" else 0.45)


def c_medians(d):
    """Per combination: mean over the two runs of the median rollout time and of the median wait, rollouts 20+."""
    iv = [r for r in read_csv(d / "logs" / "c_intervals.csv") if int(r["rollout"]) >= FIRST]
    bl = [r for r in read_csv(d / "logs" / "c_blocked.csv") if r["interval_s"] and int(r["rollout"]) >= FIRST]
    out = {}
    for cond, arm in COMBOS:
        def per_run(rows, key, scale=1.0):
            return [st.median(float(r[key]) * scale for r in rows if (r["cond"], r["arm"], r["rep"]) == (cond, arm, rep))
                    for rep in ("r1", "r2")]
        total, wait = per_run(iv, "interval_s"), per_run(bl, "blocked_ms", 1e-3)
        out[(cond, arm)] = {"total": st.fmean(total), "wait": st.fmean(wait), "runs": total, "wait_runs": wait}
    return out


def fig1_rollout_time(d, out):
    m = c_medians(d)
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ys = list(range(len(COMBOS)))[::-1]
    for y, key in zip(ys, COMBOS):
        total, wait = m[key]["total"], m[key]["wait"]
        ax.barh(y, total - wait, color=OTHER, height=0.56)
        ax.barh(y, wait, left=total - wait, color=WAIT, height=0.56)
        ax.text(total + 0.03, y, f"{total:.3f} s", va="center", fontsize=12, fontweight="bold")
        if wait > 0.3:
            ax.text(total - wait / 2, y, f"等待指标请求 {wait:.3f} s", va="center", ha="center", color="white",
                    fontsize=11)
        else:
            ax.text(total + 0.36, y, f"其中等待指标请求 {wait * 1e3:.0f} ms", va="center", fontsize=10, color="0.35")
    ax.set_yticks(ys, [LABEL[k] for k in COMBOS], fontsize=11)
    ax.set_xlim(0, 2.75)
    ax.set_xlabel("每个 rollout 的耗时（秒，中位数）")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(color=OTHER, label="训练和其他工作"), Patch(color=WAIT, label="rank 0 等待指标请求返回")],
              loc="lower right", fontsize=10, frameon=False)
    ax.set_title("有代理变量时，修复前每个 rollout 慢了一倍多，多出的时间都花在等待指标请求上", fontsize=12, loc="left")
    fig.text(0.01, 0.01, "Qwen3-0.6B SFT，8×A100，DP8。每种组合跑 2 次，每次统计第 20 个 rollout 之后的部分，图中是两次的平均。",
             fontsize=9, color="0.4")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig1_rollout_time.png", dpi=150)
    plt.close(fig)
    return m


def box(ax, x, y, text, fc, w=0.2, h=0.17):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                                fc=fc, ec="0.35", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=11)


def arrow(ax, x0, x1, y, c):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18, color=c, lw=2.2))


def fig2_client(d, out):
    rows = read_csv(d / "a_out" / "a_calls.csv")
    blocks = json.loads((d / "a_out" / "a_blocks.json").read_text())
    stats = {}
    for cond, arm in COMBOS:
        acond = "proxy" if cond == "proxy" else "no proxy"
        ms = [float(r["ms"]) for r in rows if r["cond"] == acond and r["arm"] == arm]
        bl = [b for b in blocks if b["cond"] == acond and b["arm"] == arm]
        stats[(cond, arm)] = {"n": len(ms), "p50": pct(ms, .5), "p99": pct(ms, .99), "max": max(ms),
                              "failed": sum(1 for r in rows if r["cond"] == acond and r["arm"] == arm and r["ok"] == "0"),
                              "conn_service": sum(b["conn_service"] for b in bl),
                              "conn_other": sum(b["conn_other"] for b in bl)}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    b, a = stats[("proxy", "before")], stats[("proxy", "after")]
    ax0.text(0.0, 0.95, "修复前（有代理变量，容器默认）", fontsize=12, fontweight="bold", color=BEFORE)
    box(ax0, 0.12, 0.74, "训练进程", "#f4f6f7")
    box(ax0, 0.5, 0.74, "代理", "#fdebd0")
    box(ax0, 0.88, 0.74, "指标服务", "#f4f6f7")
    arrow(ax0, 0.225, 0.395, 0.74, BEFORE)
    arrow(ax0, 0.605, 0.775, 0.74, BEFORE)
    ax0.text(0.5, 0.57, f"每次调用都新建一个连接，经代理转发\n每次调用耗时中位数 {b['p50']:.0f} ms", ha="center", va="center",
             fontsize=10.5)
    ax0.text(0.0, 0.42, "修复后", fontsize=12, fontweight="bold", color=AFTER)
    box(ax0, 0.12, 0.22, "训练进程", "#f4f6f7")
    box(ax0, 0.88, 0.22, "指标服务", "#f4f6f7")
    arrow(ax0, 0.225, 0.775, 0.22, AFTER)
    ax0.text(0.5, 0.05, f"复用同一个连接，直接连到指标服务\n每次调用耗时中位数 {a['p50']:.1f} ms", ha="center", va="center",
             fontsize=10.5)
    ys = list(range(len(COMBOS)))[::-1]
    for y, key in zip(ys, COMBOS):
        s = stats[key]
        c, alpha = color(key[1], key[0])
        ax1.barh(y, s["p50"], color=c, alpha=alpha, height=0.56)
        conns = s["conn_service"] + s["conn_other"]
        where = "全部连到代理" if s["conn_other"] else "直接连到指标服务"
        ax1.text(s["p50"] + 4, y + 0.11, f"{s['p50']:.1f} ms（p99 {s['p99']:.0f} ms）", va="center", fontsize=11)
        ax1.text(s["p50"] + 4, y - 0.17, f"新建 {conns} 个连接，{where}", va="center", fontsize=9.5, color="0.35")
    ax1.set_yticks(ys, [LABEL[k] for k in COMBOS], fontsize=11)
    ax1.set_xlim(0, 380)
    ax1.set_xlabel("每次调用的耗时（毫秒，中位数；括号里是 p99）")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title(f"每种组合调用 {stats[COMBOS[0]]['n']} 次，都没有失败", fontsize=11, loc="left")
    fig.suptitle("客户端：修复前，每个指标请求都经过代理，每次多等两百多毫秒", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_client.png", dpi=150)
    plt.close(fig)
    return stats


def fig3_service(d, out):
    rows = []
    for path in sorted((d / "logs").glob("b_*_r*.csv")):
        rows += read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1.1]})
    titles = {"before": "修复前：上报写后端时，其他请求都在排队", "after": "修复后：其他请求不受影响"}
    for ax, arm in ((axes[0], "before"), (axes[1], "after")):
        c, _ = color(arm)
        sel = [r for r in rows if r["arm"] == arm and r["rep"] == "1" and r["level"] == "tb+300ms"]
        for r in sel:
            if r["who"] == "reporter report_step":
                t = float(r["t"])
                ax.axvspan(t, t + float(r["ms"]) / 1e3, color="0.86", lw=0)
        for who, marker, alpha in (("neighbour batch", "o", 0.85), ("health", "^", 0.4)):
            pts = [(float(r["t"]), float(r["ms"])) for r in sel if r["who"] == who]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=9, marker=marker, color=c, alpha=alpha)
        ax.set_xlim(5, 15)
        ax.set_ylim(0, 350)
        ax.set_xlabel("时间（秒）；这 10 秒里写后端耗时为 TensorBoard + 300 ms")
        ax.set_title(titles[arm], fontsize=11.5, loc="left", color=c)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("其他请求的耗时（毫秒）")
    axes[1].legend(handles=[Patch(color="0.86", label="上报（report_step）正在写后端"),
                            Line2D([], [], ls="", marker="o", color="0.3", label="另一个客户端写指标"),
                            Line2D([], [], ls="", marker="^", color="0.3", alpha=0.5, label="/health 检查")],
                   loc="upper right", fontsize=9.5, framealpha=0.95)
    levels = {}
    ax = axes[2]
    for arm in ("before", "after"):
        c, _ = color(arm)
        p99s = []
        for level in B_LEVELS:
            nb = [float(r["ms"]) for r in rows if r["arm"] == arm and r["level"] == level and r["who"] == "neighbour batch"]
            hl = [float(r["ms"]) for r in rows if r["arm"] == arm and r["level"] == level and r["who"] == "health"]
            rs = [float(r["ms"]) for r in rows if r["arm"] == arm and r["level"] == level
                  and r["who"] == "reporter report_step"]
            reps = len({r["rep"] for r in rows if r["arm"] == arm and r["level"] == level})
            levels[(arm, level)] = {"done": len(nb) / reps, "p50": pct(nb, .5), "p99": pct(nb, .99), "max": max(nb),
                                    "health_p99": pct(hl, .99), "health_max": max(hl), "report": st.median(rs)}
            p99s.append(pct(nb, .99))
        x = range(len(B_LEVELS))
        ax.plot(x, p99s, "o-", color=c, lw=2, label="修复前" if arm == "before" else "修复后")
        for i, v in enumerate(p99s):
            if arm == "before" and i >= 2:
                ax.annotate(f"{v:.0f} ms", (i, v), textcoords="offset points", xytext=(-8, 4), ha="right",
                            va="bottom", fontsize=9.5, color=c)
    done = levels[("before", "tb+1000ms")]["done"]
    ax.annotate(f"后端加 1000 ms 时，30 秒只完成\n{done:.0f} 个请求（本应约 600 个）", (4, levels[("before", "tb+1000ms")]["p99"]),
                xytext=(3.95, 45), ha="right", va="center", fontsize=9.5, color=BEFORE,
                arrowprops={"arrowstyle": "->", "color": BEFORE, "lw": 0.8})
    ax.text(2, 15, "修复后都在 10–12 ms", ha="center", va="bottom", fontsize=9.5, color=AFTER)
    ax.set_xticks(range(len(B_LEVELS)), B_NAMES, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(5, 3000)
    ax.set_xlabel("上报时写后端的耗时")
    ax.set_ylabel("其他请求的 p99 耗时（毫秒，对数刻度）")
    ax.set_title("写后端越慢，修复前其他请求等得越久", fontsize=11.5, loc="left")
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.grid(alpha=0.3, which="major")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("服务端：同一个副本上，每秒上报一次，另一个客户端每 50 ms 写一次指标", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig3_service.png", dpi=150)
    plt.close(fig)
    return levels


def fig4_rollouts(d, out, m):
    iv = read_csv(d / "logs" / "c_intervals.csv")
    bl = [r for r in read_csv(d / "logs" / "c_blocked.csv") if r["interval_s"] and int(r["rollout"]) >= FIRST]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.3))
    for cond, arm in COMBOS:
        c, _ = color(arm)
        pts = sorted((int(r["rollout"]), float(r["interval_s"])) for r in iv
                     if r["cond"] == cond and r["arm"] == arm and r["rep"] == "r1" and int(r["rollout"]) < 119)
        ax0.plot([p[0] for p in pts], [p[1] for p in pts], "-" if cond == "proxy" else ":", color=c, lw=1.2,
                 label=LABEL[(cond, arm)])
    ax0.axvline(FIRST, color="0.6", lw=0.8)
    ax0.text(FIRST - 1, 3.75, "左侧是预热，\n不计入统计", ha="right", va="top", fontsize=9, color="0.4")
    ax0.set_ylim(0, 4)
    ax0.set_xlabel("rollout 序号（第 1 次运行；最后一个 rollout 含训练收尾，未画出）")
    ax0.set_ylabel("每个 rollout 的耗时（秒）")
    ax0.set_title("逐个 rollout 的耗时", fontsize=11.5, loc="left")
    ax0.legend(fontsize=9, loc="upper right", ncol=2, frameon=False)
    ax0.spines[["top", "right"]].set_visible(False)
    for cond, arm in COMBOS:
        c, _ = color(arm)
        pts = [(float(r["blocked_ms"]) / 1e3, float(r["interval_s"])) for r in bl if r["cond"] == cond and r["arm"] == arm]
        ax1.scatter([p[0] for p in pts], [p[1] for p in pts], s=10, color=c, marker="o" if cond == "proxy" else "x",
                    alpha=0.6, label=LABEL[(cond, arm)])
    base = m[("noproxy", "after")]["total"]
    xs = [0, max(float(r["blocked_ms"]) / 1e3 for r in bl)]
    ax1.plot(xs, [base + x for x in xs], color="0.35", lw=1, ls="--", label=f"{base:.2f} s + 等待时间")
    ax1.set_xlabel("rank 0 在这个 rollout 里等待指标请求的时间（秒）")
    ax1.set_ylabel("这个 rollout 的耗时（秒）")
    ax1.set_title("几乎每个点都落在虚线上：多出的时间就是等待指标请求的时间", fontsize=11.5, loc="left")
    ax1.legend(fontsize=9, loc="lower right", markerscale=1.5, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    fig.suptitle("端到端：Qwen3-0.6B SFT，8×A100，DP8，每次运行 120 个 rollout", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig4_rollouts.png", dpi=150)
    plt.close(fig)


def main():
    d, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    use_cjk_font()
    m = fig1_rollout_time(d, out)
    a = fig2_client(d, out)
    b = fig3_service(d, out)
    fig4_rollouts(d, out, m)
    print("| 组合 | 每个 rollout 的耗时（两次运行） | rank 0 等待指标请求 |")
    print("| --- | --- | --- |")
    for key in COMBOS:
        r = m[key]
        print(f"| {LABEL[key]} | {r['total']:.3f} s（{r['runs'][0]:.3f}、{r['runs'][1]:.3f}） | {r['wait'] * 1e3:.0f} ms |")
    print()
    for key in COMBOS:
        s = a[key]
        print(LABEL[key], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()})
    print()
    for (arm, level), v in b.items():
        print(arm, level, {k: round(x, 1) for k, x in v.items()})


if __name__ == "__main__":
    main()
