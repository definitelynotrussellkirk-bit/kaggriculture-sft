"""Old-architecture runs on a Colab VM from the uploaded step cache, ALL owned by one foreground cell (one or more arms
co-located on the same GPU, each its own trainer process).

  colab exec -s SESSION -f worker-<tag>.py      (launched locally under nohup; never exec a second cell into this kernel)

Expects: /content/parts/code.tar (the base-v1 trainer tree); per arm a parts directory with SEED_NAME ("<ckpt name> <sha256>")
and seed.part.* (the checkpoint to start or resume from); /content/cache/*.stp + rows.tsv + CACHE.json. setup_vm.py must have
pinned torch 2.10.0 / numpy 2.4.3 (the cache identity records them).

Per arm, /content/results/<NAME>/: HEARTBEAT.json (every 30 s), metrics.jsonl, train.log, COMMAND.json, pull/ (newest checkpoint
in 60 MB parts + SHA256, swapped atomically), pull1k/<ckpt>/ (every 1,000th update, until the puller removes it), DONE.json.
The launcher replaces the RUNS line: [(name, absolute end update, [extra train.py args], data workers, parts dir), ...].
"""
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import tarfile
import time

RUNS = [("n1-beven-colab", 1000000, ['--v6-data', 'true', '--pack-steps', '256', '--batch-windows', '1024', '--compile-model', 'true', '--grad-checkpoint', 'false', '--cache-pool-seats', '24', '--ckpt-every', '40', '--keep-ckpts', '6', '--eval-every', '0', '--log-every', '10', '--ema-decay', '0.992', '--lr', '3e-5', '--lr-schedule', 'wsd', '--decay-updates', '0', '--warmup', '300', '--grow-lr-mult', '1', '--mask-unit-noops', 'true', '--aux-scale', '2.8', '--max-seconds', '6852'], 16, "/content/parts/n1b")]
CODE = "/content/bcode"
CACHE = "/content/bcache"
PART = 60 * 1024 * 1024
ENV = {'PBC_DRAW_LOG': '/content/results/n1-beven-colab/draws', 'PBC_DEALIAS_GRADS': '1', 'PBC_EAGER_CHECKS': '1', 'PBC_ASYNC_CKPT': '1', 'PBC_DEFER_ITEMS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTORCH_ALLOC_CONF': 'expandable_segments:True', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PBC_ATTN': 'flex', 'PBC_BUCKET_GROUP': '3', 'PBC_CACHE_ROWS': '/content/bcache/rows-n-d2d605843046.tsv', 'PBC_DEC_TRIM': '1', 'PBC_EMA_EXTRA': '0.984,0.996', 'PBC_GRAD_CLIP': '10', 'PBC_HIST_DROPOUT': '0.5', 'PBC_JOBBIAS': 'fold', 'PBC_SEAT_FRACTION': '0.25', 'PBC_TURN_NORM': '1', 'PBC_TURN_W_MOVE': '0.5', 'PBC_TURN_W_ORDER': '2', 'PBC_UPDATE_STATS': '50'}


def setup_code():
    if not os.path.exists(CODE):
        with tarfile.open("/content/parts/code.tar") as t:
            t.extractall(CODE, filter="data")
    meta = json.load(open(f"{CACHE}/CACHE.json"))
    if meta.get("rows") != f"{CACHE}/rows.tsv":
        meta["rows"] = f"{CACHE}/rows.tsv"
        json.dump(meta, open(f"{CACHE}/CACHE.json.tmp", "w"), indent=1)
        os.replace(f"{CACHE}/CACHE.json.tmp", f"{CACHE}/CACHE.json")


def setup_seed(out, parts_dir):
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/LATEST"):
        return
    name, want = open(f"{parts_dir}/SEED_NAME").read().split()[:2]
    h = hashlib.sha256()
    with open(f"{out}/{name}.tmp", "wb") as o:
        for p in sorted(glob.glob(f"{parts_dir}/seed.part.*")):
            b = open(p, "rb").read()
            h.update(b)
            o.write(b)
    if h.hexdigest() != want:
        raise SystemExit(f"seed sha mismatch {h.hexdigest()} != {want} ({parts_dir})")
    os.replace(f"{out}/{name}.tmp", f"{out}/{name}")
    open(f"{out}/LATEST", "w").write(name + "\n")


def split(ckpt, pull):
    os.makedirs(pull, exist_ok=True)
    tmp = pull + ".new"
    os.makedirs(tmp, exist_ok=True)
    for old in glob.glob(tmp + "/*"):
        os.remove(old)
    h, n = hashlib.sha256(), 0
    with open(ckpt, "rb") as f:
        while chunk := f.read(PART):
            h.update(chunk)
            with open(f"{tmp}/{os.path.basename(ckpt)}.part.{n:03d}", "wb") as o:
                o.write(chunk)
            n += 1
    with open(f"{tmp}/SHA256", "w") as o:
        o.write(f"{h.hexdigest()}  {os.path.basename(ckpt)}  parts={n}\n")
    for old in glob.glob(pull + "/*"):          # swap: SHA256 appears only after all parts of the new checkpoint exist
        os.remove(old)
    for p in sorted(glob.glob(tmp + "/*.part.*")) + [tmp + "/SHA256"]:
        os.replace(p, f"{pull}/{os.path.basename(p)}")


def main():
    setup_code()
    here = f"{CODE}/primitive-bc-v6-20260927/work/runs/maintainer/primitive-bc"
    env = dict(os.environ, **ENV, PYTHONPATH=f"{CODE}/primitive-bc-v6-20260927/src:{here}")
    arms = []
    for name, end, extra, workers, parts_dir in RUNS:
        out = f"/content/results/{name}"
        setup_seed(out, parts_dir)
        latest = open(f"{out}/LATEST").read().strip()
        cur = int(latest.split("-")[1].split(".")[0])
        cmd = ["python3", "-B", f"{here}/train.py", "fit", "--out", out, "--device", "cuda", "--resume",
               "--updates", str(end), "--stop-after", str(max(0, end - cur)), "--workers", str(workers),
               "--step-cache", CACHE, *extra]
        with open(f"{out}/COMMAND.json", "w") as o:
            json.dump({"cmd": cmd, "env": ENV, "start_latest": latest, "co_located": [r[0] for r in RUNS],
                       "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, o, indent=1)
        p = subprocess.Popen(cmd, cwd=here, env=env, stdout=open(f"{out}/train.log", "a"), stderr=subprocess.STDOUT)
        arms.append({"name": name, "out": out, "p": p, "pulled": latest, "start": cur, "rc": None})
    while True:
        for a in arms:
            out = a["out"]
            rc = a["p"].poll()
            latest = open(f"{out}/LATEST").read().strip() if os.path.exists(f"{out}/LATEST") else None
            if latest and latest != a["pulled"] and os.path.exists(f"{out}/{latest}"):
                try:
                    split(f"{out}/{latest}", f"{out}/pull")
                    a["pulled"] = latest
                except FileNotFoundError:    # pruned between LATEST read and split: the next beat takes the newer one
                    pass
            for ck in sorted(glob.glob(f"{out}/ckpt-*.pt")):   # every 1,000th update: parts kept until the puller removes them
                u = int(os.path.basename(ck)[5:12])
                k = f"{out}/pull1k/{os.path.basename(ck)}"
                if u % 1000 == 0 and u != a["start"] and not os.path.exists(k + "/SHA256") \
                        and not os.path.exists(k + ".pulled"):
                    try:
                        split(ck, k)
                    except FileNotFoundError:
                        pass
            beat = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "latest": latest,
                    "pulled": a["pulled"], "returncode": rc, "pid": a["p"].pid}
            with open(f"{out}/HEARTBEAT.json.tmp", "w") as o:
                json.dump(beat, o)
            os.replace(f"{out}/HEARTBEAT.json.tmp", f"{out}/HEARTBEAT.json")
            if rc is not None and a["rc"] is None:
                a["rc"] = rc
                with open(f"{out}/DONE.json", "w") as o:
                    json.dump({**beat, "done": True}, o)
                print("done", a["name"], beat, flush=True)
        if all(a["rc"] is not None for a in arms):
            return max(a["rc"] for a in arms)
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
