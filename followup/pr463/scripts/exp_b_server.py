"""Experiment B: requests that arrive while /report_step talks to a slow backend.

One real MetricsService from one code tree (Ray Serve, CPU only), three clients
on their own connections, without proxies:
  reporter: once a second, three log_metrics_batch calls and one report_step,
            as the reporting rank does;
  neighbour: a log_metrics_batch every 50 ms on its own steps, standing for
            another process that logs while a report runs;
  health:   GET /health every 100 ms.
Levels, 30 s each: the real backends only (TensorBoard on the shared file
system), the same plus the timeline backend (300 trace events per step in the
reporter's first batch), and TensorBoard followed by a 100 / 300 / 1000 ms wait
injected by b_hook/sitecustomize.py.

Usage: python exp_b_server.py <before|after> <tree> <rep> <backend dir> <out csv>
"""

import csv
import os
import sys
import threading
import time
from argparse import Namespace

import requests

HERE = os.path.dirname(os.path.abspath(__file__))


PORT = 18200
LEVEL_S = float(os.environ.get("B_LEVEL_S", 30))
LEVELS = [("tb", 0, False), ("tb+timeline", 0, True), ("tb+100ms", 100, False), ("tb+300ms", 300, False),
          ("tb+1000ms", 1000, False)]
DELAY_FILE = os.path.join(HERE, "b_delay_ms")


def head_ip():
    import ray

    for node in ray.nodes():
        if node["Alive"] and "node:__internal_head__" in node.get("Resources", {}):
            return node["NodeManagerAddress"]
    return ray.nodes()[0]["NodeManagerAddress"]


def session():
    s = requests.Session()
    s.trust_env = False
    return s


def main():
    arm, tree, rep, backend_dir, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    sys.path.insert(0, tree)
    with open(DELAY_FILE, "w") as f:
        f.write("0")

    import ray
    from ray import serve

    ray.init(address="local", num_cpus=4, include_dashboard=False, _temp_dir="/tmp/ray_ms463_b", log_to_driver=False)
    serve.start(http_options={"host": "0.0.0.0", "port": PORT})
    from relax.utils.metrics.service import MetricsService

    cfg = Namespace(
        use_tensorboard=True, tb_project_name="ms463b", tb_experiment_name="b", save=None, use_wandb=False,
        use_clearml=False, notify_urls=None, timeline_dump_dir=os.path.join(backend_dir, "timeline"),
    )
    env = {"TENSORBOARD_DIR": os.path.join(backend_dir, "tb"), "PYTHONPATH": f"{HERE}/b_hook:{tree}",
           "MS463_DELAY_FILE": DELAY_FILE}
    serve.run(
        MetricsService.options(ray_actor_options={"num_cpus": 1, "runtime_env": {"env_vars": env}}).bind(
            healthy=None, pg=None, config=cfg, role="metrics"
        ),
        name="metrics", route_prefix="/metrics",
    )
    url = f"http://{head_ip()}:{PORT}/metrics"
    rows = []
    lock = threading.Lock()

    def record(level, who, t0_level, t0, ms, ok):
        with lock:
            rows.append({"arm": arm, "rep": rep, "level": level, "who": who, "t": round(t0 - t0_level, 4),
                         "ms": round(ms, 3), "ok": int(ok), "wall": round(time.time() - ms / 1e3, 3)})

    def timed(level, who, t0_level, fn):
        t0 = time.perf_counter()
        try:
            r = fn()
            ok = r.status_code == 200 and r.json().get("status") in ("success", "healthy")
        except Exception:
            ok = False
        record(level, who, t0_level, t0, (time.perf_counter() - t0) * 1e3, ok)

    scalars = {f"train/m{i:02d}": 0.5 + i for i in range(40)}
    step = [0]
    for level, delay_ms, timeline in LEVELS:
        with open(DELAY_FILE, "w") as f:
            f.write(str(delay_ms))
        stop = threading.Event()
        t0_level = time.perf_counter()

        def reporter():
            s = session()
            tick = t0_level
            while not stop.is_set():
                step[0] += 1
                n = step[0]
                for i in range(3):
                    batch = dict(scalars)
                    if timeline and i == 0:
                        now_us = time.time() * 1e6
                        batch["__timeline_events__"] = [
                            {"name": f"op{k}", "ph": "X", "ts": now_us + k, "dur": 5.0, "pid": 0, "tid": 0}
                            for k in range(300)
                        ]
                    timed(level, "reporter batch", t0_level,
                          lambda: s.post(f"{url}/log_metrics_batch", json={"step": n, "metrics": batch}, timeout=30))
                timed(level, "reporter report_step", t0_level,
                      lambda: s.post(f"{url}/report_step", json={"step": n}, timeout=30))
                tick += 1.0
                stop.wait(max(0.0, tick - time.perf_counter()))

        def periodic(who, period, call):
            s = session()
            tick = t0_level
            i = 0
            while not stop.is_set():
                i += 1
                timed(level, who, t0_level, lambda: call(s, i))
                tick += period
                stop.wait(max(0.0, tick - time.perf_counter()))

        threads = [
            threading.Thread(target=reporter),
            threading.Thread(target=periodic, args=("neighbour batch", 0.05, lambda s, i: s.post(
                f"{url}/log_metrics_batch", json={"step": 10_000_000 + i, "metrics": {"rollout/x": 1.0}},
                timeout=30))),
            threading.Thread(target=periodic, args=("health", 0.1, lambda s, i: s.get(f"{url}/health", timeout=30))),
        ]
        for t in threads:
            t.start()
        time.sleep(LEVEL_S)
        stop.set()
        for t in threads:
            t.join()
        print(f"{arm} rep {rep} {level}: {sum(r['level'] == level for r in rows)} requests", flush=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    serve.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()
