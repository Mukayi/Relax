"""Experiment A: the metrics client's transport, before and after #463.

One real MetricsService (Ray Serve, CPU only) and the two real MetricsClient
implementations, loaded side by side from the two code trees. Each step sends what
the reporting rank sends: three log_metrics_batch calls and one report_step. The
clients alternate in blocks, with the environment's proxy variables kept and with
them removed. Recorded per call: the time the caller is blocked, success, the
exception or HTTP status, and every new TCP connection with whether it went to
the service or to another host.

Usage: PYTHONPATH=<post tree> python exp_a_client.py <pre tree> <post tree> <out dir>
"""

import argparse
import csv
import importlib.util
import json
import os
import time
from argparse import Namespace

import requests.adapters
import urllib3.connection


PORT = 18100
STEPS_PER_BLOCK = int(os.environ.get("A_STEPS", 50))
REPS = int(os.environ.get("A_REPS", 3))
GAP_S = 0.2
PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")


def load_client(tree, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(tree, "relax/utils/metrics/client.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MetricsClient


def head_ip():
    import ray

    for node in ray.nodes():
        if node["Alive"] and "node:__internal_head__" in node.get("Resources", {}):
            return node["NodeManagerAddress"]
    return ray.nodes()[0]["NodeManagerAddress"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pre")
    ap.add_argument("post")
    ap.add_argument("out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    saved_proxy = {k: os.environ[k] for k in PROXY_VARS if k in os.environ}

    import ray
    from ray import serve

    ray.init(address="local", num_cpus=4, include_dashboard=False, _temp_dir="/tmp/ray_ms463_a", log_to_driver=False)
    serve.start(http_options={"host": "0.0.0.0", "port": PORT})
    from relax.utils.metrics.service import MetricsService

    cfg = Namespace(
        use_tensorboard=True, tb_project_name="ms463a", tb_experiment_name="a", save=None, use_wandb=False,
        use_clearml=False, notify_urls=None, timeline_dump_dir=None,
    )
    serve.run(
        MetricsService.options(
            ray_actor_options={"num_cpus": 1, "runtime_env": {"env_vars": {"TENSORBOARD_DIR": a.out + "/tb"}}}
        ).bind(healthy=None, pg=None, config=cfg, role="metrics"),
        name="metrics", route_prefix="/metrics",
    )
    ip = head_ip()
    url = f"http://{ip}:{PORT}/metrics"
    service_hosts = {ip}

    clients = {"before": load_client(a.pre, "client_before"), "after": load_client(a.post, "client_after")}

    # Every request and every new connection, from the two chokepoints both clients share.
    sends, conns = [], []
    orig_send, orig_connect = requests.adapters.HTTPAdapter.send, urllib3.connection.HTTPConnection.connect

    def send(self, request, **kw):
        t0 = time.perf_counter()
        err = status = None
        try:
            r = orig_send(self, request, **kw)
            status = r.status_code
            return r
        except BaseException as e:
            err = type(e).__name__
            raise
        finally:
            sends.append({"ep": request.path_url.rsplit("/", 1)[-1], "ms": (time.perf_counter() - t0) * 1e3,
                          "status": status, "err": err})

    def connect(self):
        conns.append("service" if self.host in service_hosts else "other host")
        return orig_connect(self)

    requests.adapters.HTTPAdapter.send = send
    urllib3.connection.HTTPConnection.connect = connect

    metrics = {f"train/m{i:02d}": 0.5 + i for i in range(40)}
    rows, blocks = [], []
    step = 0
    for cond in ("proxy", "no proxy"):
        for k in PROXY_VARS:
            os.environ.pop(k, None)
        if cond == "proxy":
            os.environ.update(saved_proxy)
        for rep in range(REPS):
            order = ("before", "after") if rep % 2 == 0 else ("after", "before")
            for arm in order:
                client = clients[arm](url)
                n_send, n_conn = len(sends), len(conns)
                t_block = time.time()
                for _ in range(STEPS_PER_BLOCK):
                    step += 1
                    step_ms = 0.0
                    for kind in ("batch", "batch", "batch", "report"):
                        t0 = time.perf_counter()
                        if kind == "batch":
                            ok = client.log_metrics_batch(step, metrics, immediate=True)
                        else:
                            ok = client.report_step(step).get("status") == "success"
                        ms = (time.perf_counter() - t0) * 1e3
                        step_ms += ms
                        rows.append({"cond": cond, "rep": rep, "arm": arm, "step": step, "kind": kind,
                                     "ms": round(ms, 3), "ok": int(bool(ok)), "wall": round(time.time(), 3)})
                    rows[-1]["step_ms"] = round(step_ms, 3)
                    time.sleep(GAP_S)
                errs = {}
                for s in sends[n_send:]:
                    key = s["err"] or (None if s["status"] == 200 else f"HTTP {s['status']}")
                    if key:
                        errs[key] = errs.get(key, 0) + 1
                c = conns[n_conn:]
                blocks.append({"cond": cond, "rep": rep, "arm": arm, "start": round(t_block, 1),
                               "requests": len(sends) - n_send, "errors": errs,
                               "conn_service": c.count("service"), "conn_other": c.count("other host")})
                print(json.dumps(blocks[-1]), flush=True)

    with open(os.path.join(a.out, "a_calls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cond", "rep", "arm", "step", "kind", "ms", "ok", "wall", "step_ms"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "a_blocks.json"), "w") as f:
        json.dump(blocks, f, indent=1)
    serve.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()
