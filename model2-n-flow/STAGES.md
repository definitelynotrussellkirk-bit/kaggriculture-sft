# Model 2 (N flow): training stages

Shipped archive `nflow-main-wash8014.7661782af4ca815585945b019963b86586dbae490ba2554b592fdb080f0f4b3e.tar.gz` (13,605,819 B).
7,252,244 parameters, random init, no weights from any other model. Every sha below was recomputed from the files on 2026-10-16 unless marked "recorded".

Chain: random init (seed 17) → main run S0-S9 on a Colab G4, u0 → u56259 → seed transform (raw := EMA) → wash, u56259 → u56559 on an RTX 4090 → raw export (EMA := raw) → EMA export → fp16 archive.

## Layout

| path | what |
|---|---|
| `code/` | trainer as of S7-S9 and the wash (`train.py` `nf_model.py` `nf_data.py` `nf_format.py` `COLUMNS.json`), codec, roster builder, seed/export/release scripts, tests, `DESIGN.md`, `DATA-FORMAT.md` |
| `code-history/<segment>/` | earlier pinned versions, only files that differ from `code/` |
| `rosters/` | every roster the stages trained on, with build receipts |
| `records/colab-main/` | RUN records for S0-S8, the Colab `train.log` (covers S0 to past u56259), Colab checkpoint sha list, keep receipt |
| `records/wash/` | wash CMD, RUN record, EFFECTIVE receipt, train.log/jsonl |
| `records/release/` | archive build, cold verify, parity, release gate receipts |
| `records/runtime/` | pip freeze of the training (py3.12) and hosted-like serving (py3.11) environments |
| `deploy/` | the serving harness the archive was built from (snapshot `repo-snap-bdbdfa70`): `build_archive.py`, agent, feature code, and the `src/farm_neat` files the archive contains |

## Hardware and runtime

| stage | hardware | python / torch |
|---|---|---|
| S0-S9 | Colab G4: NVIDIA RTX PRO 6000 Blackwell Server Edition (RUN `gpu`) | Colab python 3.13 (path in train.log) / torch 2.10.0+cu128 (RUN `torch`) |
| wash | gpu-4090b: RTX 4090 (RUN `gpu`, sacct job 9329) | Python 3.12.3 / torch 2.10.0+cu128; `records/runtime/training-runtime-py312.freeze.txt` |
| export, archive build | gpu-4090b, CPU only | Python 3.12.3; cold verify under the hosted-like py3.11 env (torch 2.6.0+cpu) |

## Stage table

| seg | updates | started (UTC, RUN) | roster (repo file, sha256 prefix) | bucket_probs | code (see pins) | logged wall s | output ckpt sha256 |
|---|---|---|---|---|---|---|---|
| S0 | 0 → 3000 | 2026-09-30 06:55:14 | `rows-nflow-v2.1ad4d8296e84.tsv` 3c711095 | none (row weights) | **not pinned**; candidate `code-history/u0000000-candidate` | 1,848 | not retained |
| S1 | 3000 → 17000 | 07:26:42 | `rows-nflow-v3.d8f474eee00d.tsv` 7a56da1f | A=.35,B=.20,C=.20,D=.15,E=.05 | P-u3000 | 8,574 | not retained |
| S2 | 17000 → 33000 | 09:49:44 | `rows-nflow-v3.b1871ee74465.tsv` 7e4b788a | A=0.3684,BC=0.4211,D=0.1579,E=0.0526 | P-u17000 | 9,980 | not retained |
| S3 | 33000 → 39020 | 12:36:17 | `rows-nflow-v3.0f274b384748.live.tsv` eaa15a27 | A=0.3495,BC=0.3995,D=0.1498,E=0.0499,L=0.0513 | P-u33000 | 3,689 | 5d7ea55c |
| S4 | 39020 → 40010 | 14:07:01 | same as S3 | same as S3 | P-u33000 | 613 | a0038f5b |
| S5 | 40010 → 40690 | 15:36:50 | same as S3 | same as S3 | P-u33000 | 429 | ceab32ca (recorded, Colab SHA256SUMS; file not retained) |
| S6 | 40690 → 48000 | 15:44:23 | same as S3 | same as S3 | P-u33000 | 4,493 | 3568f973 |
| S7 | 48000 → 54500 | 17:08:34 | `rows-nflow-v3.3d8bab05728b.tsv` 81b9ca2d | A=0.5239,BC=0.2116,D=0.1471,E=0.0490,L=0.0552,W_N=0.0131 | P-final | 3,652 | not retained |
| S8 | 54500 → 55188 | 18:11:38 | same as S7 | same as S7 | P-final | 2,969 | 5c1915f7 |
| S9 | 55188 → 56259 | not recorded | `rows-nflow-v3.b1c73e484343.tsv` c9ba2405 | A=0.6897,BC=0.1460,L=0.1143,W_N=0.0500 | **not recorded** (see gaps) | 4,475 | c27324605a88e3b4a754b1f31877041a1c95b5b6ab3740db09bd49ec0f08188e |
| wash | 56259 → 56559 | 22:30:36 (18:30:36 EDT) | `rows-nflow-v3.pool-aa146612b035.tsv` aa146612 | T1W=0.717949,BEAT1=0.082051,T2W=0.140000,T3W=0.060000 | P-final | 304 | 27c37c03b198fb65a2f58d2d1b779abfda4d702c2a38d75ad3d8b7e72ba0b12c |

- Logged wall time S0-S9: 40,722 s = 11.3 h (max `elapsed_s` per segment in `records/colab-main/train.log`). Each restart resumes from the newest checkpoint; updates logged after that checkpoint by the stopped process are discarded (e.g. S0 logged to u3010, S1 resumed at u3000).
- Checkpoint shas at boundaries come from `records/colab-main/checkpoints.sha256` and `KEEP-RECEIPT.tsv`; the u56259 file was rehashed and matches both.
- Fixed in every segment: width 256, 4 relational blocks, 8 heads (4 relation-typed), fast/slow/worker GRU 128/256/64, 2 decoder layers; Muon on weight matrices (`is_muon_matrix`) + AdamW on the rest, lr 3e-4, weight decay 0.05, beta2 0.95, clip 10, chunk 24, wdl_frac 0.1, loss weights wdl 0.2 / next 0.1 / opp 0.1, seed 17.
- LR in S0-S9: 300-update linear warmup, then constant 3e-4. `--decay-start-utc 22:15Z` makes `lr_at` return the base LR until that wall-clock time; u56259 was kept by 20:33:59Z (`KEEP-RECEIPT.tsv`), so the decay never applied inside this lineage. The wash uses wsd with no decay updates: constant 3e-5.
- Pool growth in S0/S1: the Colab step cache was still being filled while training. Rows without a `.stp` wait and join the pool on the next 600 s re-list (`nf_data.py` `_select`). Usable rows went 1,783 → 6,244 of 14,737 during S0 and 3,015 → 13,748 of 13,751 during S1 (`nf_data:` lines in train.log). The exact data exposure of S0/S1 therefore depends on fetch timing.

## Code pins

From the `code_sha256` field of each RUN record (S0 has none). All files below are in the repo byte-identical; shas verified.

| pin set | train.py | nf_model.py | nf_data.py | nf_format.py | COLUMNS.json |
|---|---|---|---|---|---|
| S0 candidate (unpinned) | e2f3e6c4 `code-history/u0000000-candidate` | eb67a63f (= P-u3000) | c741e331 `u0000000-candidate` | e8fcf366 `u0000000-candidate` | fa2c826a |
| P-u3000 | b81a5284 `code-history/u0003000` | eb67a63f `code-history/u0003000` | e1c32cc2 `code-history/u0003000` | 8c0984df `code-history/u0003000` | fa2c826a |
| P-u17000 | 26090e61 `code-history/u0017000` | ff16ef16 `code/` | 15de4b2a `code-history/u0017000` | 8c0984df `code-history/u0003000` | fa2c826a |
| P-u33000 | 9a3b5007 `code-history/u0033000` | ff16ef16 `code/` | 15de4b2a `code-history/u0017000` | c79fc6c5 `code-history/u0033000` | fa2c826a |
| P-final (S7, S8, wash) | c488dbbe `code/` | ff16ef16 `code/` | 53065aa2 `code/` | 70f351af `code/` | fa2c826a `code/` |

- The S0 candidate is the code tarball staged at 06:41Z for the 06:54Z launch; its roster file matches RUN-0000000's roster sha (3c711095), its code is not pinned by any record.
- `codec.py` (a8ecdb49, `code/pbc/work/runs/maintainer/`) is not in the RUN pins. The same bytes are in the S0 candidate tarball, the NFS code dir, and the shipped archive.
- `roster.py` (a2f5c3e0) and the launch scripts are current bytes, unpinned. Rosters are included byte-identical, so a roster builder version is not needed to retrain.

## Main-run commands (S0-S9)

S0 command (from `records/colab-main/RUN-0000000.json` args; cwd = /content/code):

```
python3 -B train.py --roster /content/code/data/rows-nflow-v2.tsv --out /content/results/nflow --stp-dir /content/cache --columns /content/code/COLUMNS.json --codec-dir /content/code/pbc/work/runs/maintainer --max-seconds 60467.749626 --stop-utc 2026-09-30T23:45:00Z --decay-start-utc 2026-09-30T22:15:00Z --updates 200000 --stop-at 0 --streams 2 --workers 32 --chunk 24 --accum 2 --wdl-frac 0.1 --lr 0.0003 --min-lr-frac 0.1 --warmup 300 --schedule cosine --decay-updates 0 --weight-decay 0.05 --beta2 0.95 --grad-clip 10.0 --adamw-lr-mult 1.0 --optimizer muon --muon-momentum 0.95 --muon-ns-steps 5 --ema 0.999 --compile default --ckpt-every 500 --ckpt-minutes 15.0 --keep-ckpts 6 --log-every 10 --profile-at 40 --seed 17 --width 256 --blocks 4 --heads 8 --rel-heads 4 --fast 128 --slow 256 --worker 64 --dec-layers 2 --w-wdl 0.2 --w-next 0.1 --w-opp 0.1
```

Later segments = the S0 command with these flags changed (S1-S8 from their RUN records, S9 from the `args` stored in ckpt-0056259.pt):

| seg | changed flags |
|---|---|
| S1 | `--bucket-probs A=.35,B=.20,C=.20,D=.15,E=.05 --roster /content/code/data/rows-nflow-v3.tsv` (max-seconds 58579) |
| S2 | `--allow-recipe-change bucket_probs,roster_sha256 --bucket-probs A=0.3684,BC=0.4211,D=0.1579,E=0.0526 --profile-at 0 --roster /content/code/data/rows-nflow-v3.b1871ee74465.tsv` (max-seconds 49997) |
| S3 | `--allow-recipe-change bucket_probs,roster_sha256 --bucket-probs A=0.3495,BC=0.3995,D=0.1498,E=0.0499,L=0.0513 --profile-at 0 --roster /content/code/data/rows-nflow-v3.0f274b384748.tsv` (max-seconds 40004) |
| S4 | `--allow-recipe-change bucket_probs,roster_sha256 --bucket-probs A=0.3495,BC=0.3995,D=0.1498,E=0.0499,L=0.0513 --keep-ckpts 12 --profile-at 0 --roster /content/code/data/rows-nflow-v3.0f274b384748.tsv` (max-seconds 34560) |
| S5 | `--allow-recipe-change bucket_probs,roster_sha256 --bucket-probs A=0.3495,BC=0.3995,D=0.1498,E=0.0499,L=0.0513 --keep-ckpts 12 --profile-at 0 --roster /content/code/data/rows-nflow-v3.0f274b384748.tsv` (max-seconds 29171) |
| S6 | `--allow-recipe-change bucket_probs,roster_sha256 --bucket-probs A=0.3495,BC=0.3995,D=0.1498,E=0.0499,L=0.0513 --keep-ckpts 12 --profile-at 0 --roster /content/code/data/rows-nflow-v3.0f274b384748.tsv` (max-seconds 28719) |
| S7 | `--allow-recipe-change bucket_probs,roster_sha256 --bucket-probs A=0.5239,BC=0.2116,D=0.1471,E=0.0490,L=0.0552,W_N=0.0131 --keep-ckpts 12 --profile-at 0 --roster /content/code/data/rows-nflow-v3.3d8bab05728b.tsv` (max-seconds 23667) |
| S8 | `--accum 8 --allow-recipe-change bucket_probs,roster_sha256,accum,streams_total,ema --bucket-probs A=0.5239,BC=0.2116,D=0.1471,E=0.0490,L=0.0552,W_N=0.0131 --ckpt-every 63 --ema 0.996 --keep-ckpts 12 --profile-at 0 --roster /content/code/data/rows-nflow-v3.3d8bab05728b.tsv --shards /content/nfz --streams 4` (max-seconds 19883) |
| S9 | `--accum 8 --allow-recipe-change bucket_probs,roster_sha256,accum,streams_total,ema --bucket-probs A=0.6897,BC=0.1460,L=0.1143,W_N=0.0500 --ckpt-every 63 --ema 0.992 --keep-ckpts 12 --profile-at 0 --roster /content/code/data/rows-nflow-v3.b1c73e484343.tsv --shards /content/nfz --streams 4` (max-seconds 16904) |

## Wash (u56259 → u56559)

Recorded submit line (sacct job 9329, gpu-4090b, 2026-09-30 18:30:28-18:35:41 EDT):

```
sbatch --parsable -w gpu-4090b -J nflow-wash8014 /mnt/ai/scratch/farmNEAT/n-flow/code/arm_skynet.sbatch \
  nflow-u56259-w8014 /mnt/ai/scratch/farmNEAT/n-flow/keeps/ckpt-0056259.pt \
  /mnt/ai/scratch/farmNEAT/broad/t1/wash/rows-t1wash-nflow.aa146612b035.tsv 56559 0.999 3e-5 840
```

`code/arm_skynet.sbatch` then ran (literal, `records/wash/CMD`):

1. `make_t4_seed.py keeps/ckpt-0056259.pt t4f/nflow-u56259-w8014/seed/ckpt-0056259.pt`: raw weights := EMA, EMA kept. Input c2732460…, output b93b5f09….
2. ```
   /mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12 -B train.py --roster /mnt/ai/scratch/farmNEAT/n-flow/t4f/nflow-u56259-w8014/rows.tsv --shards /mnt/ai/scratch/farmNEAT/n-flow/shards-v1 --stp-dir /mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11 --out /mnt/ai/scratch/farmNEAT/n-flow/runs/nflow-u56259-w8014 --max-seconds 840 --updates 200000 --lr 3e-5 --warmup 0 --schedule wsd --decay-updates 0 --workers 4 --streams 5 --chunk 24 --accum 3 --grad-clip 10 --ema 0.999 --ckpt-every 250 --ckpt-minutes 5 --keep-ckpts 3 --log-every 10 --new-seed 41 --stop-at 56559 --bucket-probs T1W=0.717949,BEAT1=0.082051,T2W=0.140000,T3W=0.060000 --allow-recipe-change lr,warmup,schedule,decay_updates,decay_start_utc,stop_utc,accum,streams_total,bucket_probs,roster_sha256,wdl_frac,ema,grad_clip,chunk,min_lr_frac,weight_decay,beta2,optimizer,muon_momentum,muon_ns_steps,adamw_lr_mult,w_wdl,w_next,w_opp
   ```
   env `PYTHONPATH=/mnt/ai/scratch/farmNEAT/runtime/site NF_PREFETCH=4`, cwd `n-flow/code`. 806 of 806 roster rows usable (341 `.nfz`, 465 `.stp`); 404 imitation seats + 402 WDL-only seats; 300 updates in 304 s.

Then, outside SLURM (the command line is not recorded; the transform is recorded in the output checkpoint's `extra.raw_export`):

3. `make_raw_as_ema.py runs/nflow-u56259-w8014/ckpt-0056559.pt runs/nflow-main-wash8014/ckpt-0056559.pt`: EMA := raw weights, so the archive ships the unaveraged end weights. Input 27c37c03…, output 7dacc598….

Re-running steps 1 and 3 with the repo's scripts on the recorded inputs gives checkpoints whose 577 tensor leaves and `extra` dicts all equal the originals (file bytes differ only by serialization).

## Export and archive

Recorded submit line (sacct job 9370, 18:36:06-18:37:45 EDT):

```
sbatch --parsable --export=ALL -J nflow-rel-nflow-main-wash8014 /mnt/ai/scratch/farmNEAT/n-flow/code/release_run.sbatch \
  /mnt/ai/scratch/farmNEAT/n-flow/runs/nflow-main-wash8014 nflow-main-wash8014 \
  /mnt/ai/scratch/farmNEAT/n-flow/t4f/nflow-u56259-w8014/rows.tsv \
  /mnt/ai/scratch/farmNEAT/n-flow/t4f/nflow-u56259-w8014/seed/ckpt-0056259.pt \
  /mnt/ai/scratch/farmNEAT/n-flow/t4f/nflow-u56259-w8014/CMD
```

`release_run.sbatch` → `p2_nflow.sh` → `nflow_deploy.sbatch` with `NFLOW_RS=n-flow/repo-snap-bdbdfa70` (the harness snapshot; all 35 code members of the archive match it byte for byte, and the archive's `builder_sha256` 48e9213a… is that snapshot's `build_archive.py`):

| step | command (from `nflow_deploy.sbatch`) | output |
|---|---|---|
| pin | `pin_checkpoint.py` copies the checkpoint to `release-ckpts/nflow-main-wash8014/` | 7dacc598… |
| EMA export | `ema_export.py CKPT ema.pt` | ema.pt efa5bb25… (= MANIFEST `model_sha256`; re-run here gives 127/127 equal tensors) |
| archive | `build_archive.py build --model ema.pt --out NAME.tar.gz --storage fp16 --temperature 0 --no-verify` (PYTHONPATH `deploy/src:deploy/work/runs/maintainer/n-flow-deploy`) | policy.pt 56fdb450…, archive 7661782a… |
| cold verify | `build_archive.py verify --archive … --opponents PASS --seats 0,1` under the hosted-like py3.11 env | 719 steps per seat, 0 fallbacks (`records/release/VERIFY.json`) |
| parity | `test_nflow_parity.py --steps 30 --ckpt ema.pt` | 248 decisions, max abs logit diff 4.5e-5, tol 2e-3 (`PARITY.json`) |
| checks | `p2_check.py`, `finalize_snapshot.py`, `release_gate.py` | `P2-CHECK.json`, `RELEASE-DONE.json`, `LINKS-*.json` all_pass |

The serving copy of `nf_format.py` in the archive is 8c0984df (the P-u3000/P-u17000 version) and `model.py` is b8d83a86; training at S7-wash used `nf_format.py` 70f351af. The parity row above is the measured train/serve agreement for the shipped weights. `finalize_snapshot.py` lived outside the n-flow tree (`/mnt/ai/scratch/farmNEAT/freshen/guard/`) and is not copied.

## Inputs not in this folder

| input | how it is made | used by |
|---|---|---|
| step cache `.stp` files (`base-v1-stepcache-v11`; on Colab `/content/cache`) | from Kaggle episodes by the step-cache builder (see `data/`); `code/build-own-stp.sbatch` shows the call used for our own losing seats: `step_cache.py build ROSTER OUT --v6 --v8 --procs 4` from `primitive-bc-v6-20260927` | every stage |
| shards `.nfz` (`n-flow/shards-v1`; on Colab `/content/nfz`) | `python3 -B code/nf_format.py build ROSTER OUT --shard i/N [--stp-dir CACHE]` (`code/build-shards.sbatch`). A shard is a projection of one `.stp` by the same encoder the loader applies to a `.stp` (`nf_data.load_seat`), so shards change speed, not data. | S8, S9, wash |
| bucket generations (`broad/buckets/<gen>/A.tsv`) and the wash pool (`broad/build_t1wash.py`) | see `data/`; the rosters here are their outputs, byte-identical, with receipts naming the source file and sha | roster builds |
| `COLUMNS.json` | `nf_format.py derive` on `rows-nflow.1ad4d8296e84.tsv` (recorded in the file); the file itself is included and pinned | every stage |

## Path mapping

| original path | in this repo |
|---|---|
| `/content/code` (Colab VM; synced from `/mnt/ai/scratch/farmNEAT/n-flow/code`) | `code/` (+ `code-history/`) |
| `/content/code/data/rows-nflow-v2.tsv` | `rosters/rows-nflow-v2.1ad4d8296e84.tsv` |
| `/content/code/data/rows-nflow-v3.tsv` | `rosters/rows-nflow-v3.d8f474eee00d.tsv` |
| `/content/code/data/rows-nflow-v3.<gen>.tsv` | `rosters/rows-nflow-v3.<gen>.tsv` (gen 0f274b384748 → `.live.tsv`; the other NFS copy was rewritten by a dry run) |
| `/mnt/ai/scratch/farmNEAT/broad/t1/wash/rows-t1wash-nflow.aa146612b035.tsv`, `n-flow/t4f/nflow-u56259-w8014/rows.tsv` | `rosters/rows-nflow-v3.pool-aa146612b035.tsv` (same sha, same receipt) |
| `/content/cache`, `/mnt/ai/scratch/farmNEAT/base-v1-stepcache-v11` | step cache (`data/`) |
| `/content/nfz`, `/mnt/ai/scratch/farmNEAT/n-flow/shards-v1` | shards, rebuilt with `nf_format.py build` |
| `/content/code/pbc/work/runs/maintainer`, `n-flow/code/pbc/work/runs/maintainer` | `code/pbc/work/runs/maintainer` |
| `/mnt/ai/scratch/farmNEAT/n-flow/repo-snap-bdbdfa70` | `deploy/` |
| `/mnt/ai/scratch/farmNEAT/runtime/python/bin/python3.12` + `runtime/site` | Python 3.12.3 + `records/runtime/training-runtime-py312.freeze.txt` |
| `NFLOW_PBC_DIR` default in `nf_format.py` (`/home/russ/Desktop/farmNEAT/work/runs/maintainer/primitive-bc-v1-deploy-v9-decode`) | only put on `sys.path` by `read_stp`; set `NFLOW_PBC_DIR` to `deploy/work/runs/maintainer/n-flow-deploy` |

The code keeps its original absolute paths (it is byte-identical to what ran); override them with the flags above.

## Not established

- S0 trainer bytes: no record pins them. The candidate tarball is the best match found; its `nf_model.py` equals the S1 pin.
- S9 (u55188 → u56259): no RUN record was recovered, so its start time and code pins are not recorded. Its args come from the checkpoint. The NFS code dir that Colab synced from has held the P-final bytes since 11:55 EDT (15:55Z, file mtimes), before S7 started; that is the only evidence for S9's code.
- Checkpoints at u3000, u17000, u33000, u54500 and the S0 random init were not retained, so those boundaries cannot be checked by hash.
- Bit-exact retraining is not expected: S0/S1 data exposure depended on step-cache fetch timing (above), the main run used `torch.compile` on a Blackwell GPU, and data loading uses 32 worker processes.
- The literal Colab command lines for S0-S9 were not recorded; the commands above are rebuilt from the parsed args the trainer saved.
