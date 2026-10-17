# farmNEAT league service — API and data contract

Contract version **1.0.0** (2026-09-27). Service: CT118 `league` on the R730xd (`pxmx`), **http://league.lan/**
(192.168.88.61, port 80, LAN only). Code: `work/runs/maintainer/pbc-league-service/` in the farmNEAT repo.

The league keeps playing our shippable Kaggle archives against each other, the way Kaggle's episode scheduler does,
so the team decides by the **league leader** (owner, 2026-09-27) and occasionally uploads a non-leader to Kaggle to
see how it does. Scripts and agents may rely on everything in this document; a change that breaks it bumps the major
version (the `contract` field of `/api/v1/health` and `/api/v1/board`).

## 1. Entities

| Entity | Meaning |
|---|---|
| submission | one Kaggle-format agent archive (`.tar.gz` with `main.py` at the root) entered under a unique name |
| match | one fresh world seed and a pair of submissions; always played in **both seat orders** |
| episode | one game: a world seed and a seat order (seat 0, seat 1); 719 steps with the exact engine |
| rating | Bradley-Terry strength on the Elo scale, recomputed from every rated episode |

Identifiers: `submission.id` and `episode.id` are integers assigned by the service. A submission can be addressed by
id or by name everywhere. The replay's Kaggle `EpisodeId` is `7100000000000000 + episode.id` (16 digits: the local
catalog treats ids >= 10^12 as local games); `/api/v1/episodes/{id}` accepts either form.

## 2. Submissions

`POST /api/v1/submissions` (auth). Two forms:

* **Upload**: body = the archive bytes, `Content-Type: application/gzip`, query `name`, optional `note`,
  optional `allow_duplicate=1`. Stored at `/mnt/ai/scratch/farmNEAT/league/archives/<sha256[:12]>-<name>.tar.gz`.
* **Register a NAS path**: `Content-Type: application/json`, body `{"name", "note", "archive_path",
  "allow_duplicate"}`; `archive_path` must be under `/mnt/ai/` (the league host mounts the same NAS). The file is
  not copied; its sha256 is pinned at registration.

Rules: `name` matches `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` and is unique; `note` <= 500 characters; archive <= 100 MB.
An archive whose sha256 equals a non-retired, non-failed submission is refused (409) unless `allow_duplicate`.
Response `201` with the submission (status `pending`), or `422` with status `failed` when the static archive check
fails (not a gzip tar, unsafe or non-regular members, no `main.py`).

**Validation** (worker, one at a time, before any league game):

1. sha256 re-checked against the registered value; static archive check; `MANIFEST.json` summary recorded
   (schema, declared temperature/seed/threads, weight storage, model and builder sha256) when present.
2. **Cold verify on the hosted-like runtime**:
   `/mnt/ai/scratch/farmNEAT/runtime-hosted-py311/venv/bin/python -B <engine>/work/runs/maintainer/primitive-bc-v1-deploy-v4land-a1/build_archive.py verify --archive A --opponents PASS --seats 0,1 --out verify.json`
   (the same command `pbc-selfplay/gate_candidates.py` runs; environment `CUDA_VISIBLE_DEVICES=` and
   `OMP_NUM_THREADS=1`). **Pass = `complete` is true and `fallbacks` is 0** (both seats).
3. **Latency is recorded, not gating**: p50/p95/max step milliseconds after the first step, per seat, plus the
   machine load it was measured under (`load1` at start and end, games running at start and end).

Pass → `provisional`; fail → `failed` with the reason. A validation that was stopped by a thermal pause and did not
pass is re-queued (`pending`, up to 3 attempts) instead of failing, because the verifier's per-request deadline runs
through the pause. Validation records are kept under
`/mnt/ai/scratch/farmNEAT/league/validation/<id>-<name>/<UTC stamp>/` (`verify.json`, `verify.log`).

**Desktop receipt instead of the league's cold verify** (`[validation] accept_receipts = true`; default `false`;
added 2026-09-28). For a pending submission registered by `archive_path` (not uploads; never for the re-verification
of imports), after step 1 the worker reads `VERIFY.json` (`validation.receipt_name`) in the archive's folder: the
receipt `primitive-bc-v1-deploy-v8/deploy_checkpoint.py` writes before it registers the archive. It stands in for
step 2 only when every check holds:

* `schema` is in `validation.receipt_schemas` (`primitive-bc-cold-archive-verify-v2`);
* the verifier build is in `validation.receipt_verifiers` (sha256 list). The identity is the receipt's
  `verifier_sha256` when present, else the archive MANIFEST's `builder_sha256` (the v8 receipt does not name its
  verifier; `deploy_checkpoint.py` builds and verifies with the same `build_archive.py`). Listed:
  `c56789e636824866f5a24f705fbe6cdda2c18f1e32d9c155f44ceea930a394ed` = `primitive-bc-v1-deploy-v8/build_archive.py`,
  whose `verify` is the league verifier's (`primitive-bc-v1-deploy-v4land-a1/build_archive.py`, same `play.py` and
  cold host) plus the time-bank gate (`timing_budget.within_bank` in each game's `complete`);
* `archive_sha256` and `archive_bytes` equal the registered values (the worker has just re-hashed the archive);
  `complete` is true, `fallbacks` 0, `threads` 1;
* exactly one game per seat in `validation.seats` against each of `validation.opponents`, each: done, not censored,
  719 steps, statuses `DONE`/`DONE`, 0 fallbacks, world seed 20260923 and sampler seed 7100 + seat (the verifier
  defaults the league's command uses), 719 timed decisions; its cold host: `complete`, `engine_complete`, 0 hard
  errors, 719 contiguous step-log lines with 0 fallback lines, 1 thread, ready, the closed-archive import check
  (`closed`, no module of ours imported from outside the archive, clean close), `timing_budget` valid and within the
  60 s bank at 1 s per action, and torch/numpy imported from the `games.python` environment.

Accepted → `provisional` with `status_reason` naming the receipt path and its sha256. The validation record has
`method: "desktop-receipt"`, `receipt` (`path`, `sha256` and `bytes` of the exact bytes judged, `schema`,
`verifier_sha256`, `verifier_source`, `copy`, `receipt_archive`), the receipt's per-seat latency
(`latency_measured_on: "desktop (receipt)"`) and no `exit` (no league verify process ran); those bytes are kept as
`receipt.json` in the validation folder; event `receipt_accepted`. Anything missing or mismatched → the normal cold
verify runs: `status_reason` reads `cold verify started (desktop receipt not used: <first problem>)`, event
`receipt_not_used`, and the final validation record carries `receipt` with `accepted: false` and `problems`. A
receipt never fails a submission; only the cold verify does. Trust: whoever can write the archive's NAS folder can
write its receipt, as with the archive itself.

### Status

| Status | Meaning | Plays |
|---|---|---|
| `pending` | accepted, waiting for validation | no |
| `validating` | cold verify running (or a desktop receipt being checked) | no |
| `provisional` | validated; fewer than 20 rated games; scheduled with priority | yes |
| `active` | at least 20 rated games | yes |
| `retired` | by the retire rule or an admin; keeps its rating and history; can be reactivated | no |
| `failed` | validation failed or the archive changed on disk after registration | no |

Retire rule (as in the prototype): after each rating update, active entries ranked below the best 16 active entries
are retired, except the anchor. Every status change is logged (`status_log` in the submission detail).

## 3. Matchmaking (Kaggle-like)

The worker keeps `max_concurrent_games` episodes running (default 6; a validation run takes one slot). When the
queue is empty it creates a match:

* **New submissions get more games early**: while an entry has fewer than 20 rated games, 60% of new matches pick
  such an entry first (fewest games plus in-flight episodes first).
* Otherwise player A is drawn with weight `(sigma/100)^2 + 0.25` (uncertain ratings play more).
* The opponent B is drawn with weight `4 p (1-p) * (1 + sigma_B/200) / (1 + pair_games/8)`, `p` = the BT win
  probability of A over B: close and uncertain ratings are favoured and opponents are spread.
* Both weights are divided by `(1 + queued/running episodes of that entry)^2`, so matches created back to back do not
  pile onto one entry.
* **Every match uses a fresh world seed and is played in both seat orders** (two episodes). Seeds count up from
  32000000 (the desktop prototype used 31000000+); a seed is never reused.
* Only `provisional` and `active` entries play. Queued episodes of an entry that stops being playable are cancelled.

## 4. Games

Each episode runs `league_game.py` with the hosted-like runtime above: it calls `pbc-ladder/ladder.py`'s `play()`
unchanged from an immutable engine snapshot (`/mnt/ai/scratch/farmNEAT/league/engine/<id>/`, `SNAPSHOT.json` lists
every file's sha256): cold archive hosts (each archive's own `main.py` in a fresh process, one torch thread, its
MANIFEST's declared temperature and seed), the exact engine (`farm_neat.game.exact_engine`), 719 steps, sampler seed
7100, no hosted time bank (`games.hosted_rules = false`; step latency is recorded instead).

**Dusk RNG rule.** With `games.crn_rng = true` (owner 2026-09-27) the worker passes `--crn-rng` and the game and its
replay use the opt-in CRN dusk of the snapshot's `src/farm_neat/rollout/crn_rng.py` (configuration `crnRng: true`): each
tile's weed draw is keyed by (seed, day, farm, x, y) and each shop unlock by (seed, day), so both seat orders of a match
see the same shops and weeds whatever the players do (same per-tile and per-unlock marginals; a declared rules variant,
not the original engine). Such episodes record `runtime.crn_rng = true`, the replay's `configuration.crnRng = true` and
`info.LeagueEpisode.crn_rng = true`, and the export manifest row `crn_rng: true`; re-simulate them with
`crn_rng.environment()`. The snapshot must contain that module (`make_crn_snapshot.py`), else the runner fails the episode.

Result: **win = higher final cash, equal cash = draw**. An episode is **rated** unless a seat had a fallback, a hard
error, fewer than 719 steps, or the replay rebuild disagreed with the game's cash (`unrated_reason` says which). A
runner failure is retried up to 3 attempts, then the episode is `error`. Episode states: `queued`, `running`,
`done`, `error`, `cancelled`.

**Replays** are Kaggle episode JSON (the layout of `pbc-seed-action-replays/seed_action_replays.py`
`reconstruct_episode`): header fields from an official 2026-09-26 replay (`replay-template.json`, configuration with
`seed: null` as on Kaggle), `info.seed` = world seed, `info.TeamNames` = `SELF:<entry name>` for both seats (the
catalog's local-game convention), 720 boundaries, step-0 action PASS, no `remainingOverageTime`. The runner rebuilds
the replay from (world seed, recorded actions) with the exact engine and checks the final cash
(`replay_verified`). Files: `/mnt/ai/scratch/farmNEAT/league/episodes/<UTC day>/<episode id>/` with `game.json`
(the ladder's game record + league metadata), `game.log` and `replay.json.gz` until the hourly export below moves the
replay into the catalog zip (then the episode's replay reference is `zip:<zip>#<member>`; the API serves it from
there, so every replay is stored once).

**Catalog export**: every 15 min (`league-backup.timer`) episodes that finished before the current UTC hour are packed
into the local replay catalog's fetch-folder format: `/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/league/`
`archives/league-<UTC hour>Z-<stamp>.zip` (members `<EpisodeId>.json`) and `MANIFEST.jsonl` (episode_id, archive,
member, sha256, bytes, day, team_names, submission_ids, rewards = final cash, winner_seat, versions = entry names,
world_seed, league_episode, rated). Zips are written as `.tmp`, read back and hash-checked, renamed, then the manifest
is appended, then the episodes point at their zip member and the loose `.json.gz` files are deleted; nothing is
rewritten. `catalog.py update` picks the folder up as source `league` (local self-play, so training queries exclude it
unless asked).

## 5. Ratings

Bradley-Terry MAP estimate (MM iterations to convergence) over all rated episodes, with **one virtual draw of every
entry against the anchor** (the prototype's prior), on the Elo scale **anchored at `fwd1-46236` = 2880**
(Kaggle 56561674, hosted peak 2899.2). `sigma` is the standard error from the observed Fisher information of the
same likelihood (anchor fixed, so the anchor's sigma is 0 and every other sigma is relative to it); the 95% interval
is `rating ± 1.96 sigma`. An entry with no games shows 2880 ± ~680. Ratings are recomputed at most every 30 s after
new results. Retired entries stay in the fit (their games still inform everyone else) and are hidden from the
default board.

**Leader** = the top-rated playable entry, once it is `active` (at least 20 rated games). While the top entry is
still provisional, `/board` returns `leader: null` and names it in `top`; `best_active` is the highest-rated active
entry (for information, not the leader).

## 6. Endpoints (`/api/v1`, JSON unless noted)

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/health` | — | `ok` (worker heartbeat < 120 s; HTTP 503 otherwise), worker state (running/max games, queue, validation, pause flags), thermal state, episode/submission counts, games per hour (1 h, 24 h), disk |
| GET | `/metrics` | — | the same as JSON; `GET /metrics` (no prefix) is Prometheus text |
| GET | `/thermal` | — | the thermal guard's last reading and decision |
| GET | `/board` | — | leaderboard: `leader` (null until the top entry is active), `top`, `best_active`, `rows` (rank, id, name, status, rating, sigma, lo, hi, games, wins, draws, losses, win_share, mean_margin, note), `anchor`, `method`, `computed_utc`; `?all=1` includes retired/failed; `?format=md` markdown |
| GET | `/submissions` | — | all submissions with status, validation summary and rating; `?status=` filter |
| GET | `/submissions/{id or name}` | — | detail: archive path/sha256, manifest, full validation record, rating, status log |
| GET | `/submissions/{id or name}/episodes` | — | that submission's episodes, newest first (`limit` <= 1000, `before`, `since`, `state`) |
| GET | `/submissions/{id or name}/opponents` | — | head-to-head over rated episodes: per opponent games, W/D/L, score, mean margin, opponent rating |
| GET | `/episodes` | — | episodes, newest first; `submission=`, `state=`, `limit=`, `before=`, `since=` (ascending) |
| GET | `/episodes/{id}` | — | one episode: seats (name, cash, status, fallbacks, p50 ms), score0, rated/unrated_reason, timing, load at start, thermal pause flag, runtime; `?record=1` adds `game.json` |
| GET | `/episodes/{id}/replay` | — | the Kaggle replay JSON (served gzip-encoded when the client accepts gzip); `?raw=gz` downloads the `.json.gz` |
| POST | `/submissions` | token | see section 2 |
| POST | `/submissions/{id or name}/retire` | token | body `{"reason"}`; cancels its queued episodes |
| POST | `/submissions/{id or name}/reactivate` | token | retired → provisional or active (by rated games) |
| POST | `/admin/pause`, `/admin/resume` | token | stop / restart launching games and validations; running games finish (drain); persists across restarts |
| GET | `/contract` | — | this document |
| GET | `/` | — | HTML dashboard (leaderboard + state), refreshes every 60 s |

Errors: JSON `{"error": "..."}` with 400 (bad input), 401 (token), 404, 409 (name taken, duplicate archive, invalid
transition), 411/413 (upload size), 422 (archive check failed), 500, 503 (health not ok).

Client: `league_client.py submit|status|board|episodes|h2h|replay|health|retire|reactivate|pause|resume|sync-gates`
(standard library; `$LEAGUE_URL`, default `http://league.lan`).

## 7. Auth

Read endpoints are open on the LAN (like home.lan). Every POST needs `Authorization: Bearer <upload token>`. The
token lives in `/home/russ/Desktop/LAN/secrets.txt` (section **farmNEAT league API**, key `upload_token`; pointer row in
`LAN/credentials.md`) and in `/etc/league/api-token` on the container (root:league, 0640). The client reads it at run
time from `$LEAGUE_TOKEN`, `--token-file` or that secrets section; it is never logged. Rotate: write a new value to
both places; no restart needed (read per request).

## 8. Thermal guard

The R730xd runs hot (inlet ~30-33 °C, fans near max). A read-only host timer (`league-thermal.timer` on pxmx, every
30 s) writes the BMC's sensors (`ipmitool sdr type temperature/fan`) to `/var/lib/league-thermal/thermal.json`,
bind-mounted read-only into the container at `/mnt/host-thermal/`. The worker checks every 15 s:

* **Primary rule: inlet temperature** — pause at **>= 34 °C**, resume **below 32 °C**.
* **Fallback when the feed is missing or older than 180 s**: the host CPU package temperature (coretemp, visible in
  the container) — pause at >= 80 °C (the sensor's `max`), resume below 72 °C.
* Neither readable: pause (fail closed).

Pause = every running game and validation process group gets SIGSTOP and no new work starts; resume = SIGCONT (the
pause time does not count against the game timeout). Episodes that were paused are flagged `thermal_paused`
(their step latencies include the pause). Pauses and resumes are recorded as events; `/health` shows the state.

## 9. Resources, storage, retention, backups, failure behaviour

* Container CT118: 24 cores (16 until 2026-09-27 20:20 EDT), 32 GB RAM, 32 GB rootfs on `local-lvm`; NAS `/mnt/ai` over NFS inside the container.
* Worker unit caps: `CPUQuota=2200%` (drop-in cpuquota.conf; was 1400%), `MemoryMax=26G`, games at nice 10. Each game ≈ 1 busy core (the two cold
  hosts alternate) and ≈ 0.9 GB RAM; `max_concurrent_games` = 12 since 2026-09-27 20:20 EDT (was 6; owner: raise it while the R730 is under 50% busy). No new game starts while `MemAvailable` < 4 GB. Measured at 6 concurrent on 2026-09-27
  15:28-16:02 UTC: 33 episodes in 0.56 h = 59 episodes/hour, median 336 s per 719-step episode (235-546 s), median
  step p50 165 ms per seat; `league_stats.py` on the container gives the current numbers.
* Deploys: `deploy/deploy.sh --drain` pauses scheduling, waits for running games to finish, switches the release and
  resumes; a plain deploy or a worker restart re-queues running episodes instead.
* Live state: SQLite (WAL) `/var/lib/league/league.sqlite3` on the container. Backups: every 15 min an online backup to
  `/mnt/ai/scratch/farmNEAT/league/backups/db/league-latest.sqlite3` plus an hourly gzip snapshot kept 7 days; the
  container itself is in the nightly Proxmox vzdump job (02:30, `nas-ai-backup`).
* NAS data (`/mnt/ai/scratch/farmNEAT/league/`): `archives/` (uploads, kept), `episodes/<day>/<id>/` (`game.json`,
  `game.log` kept), `validation/`, `engine/` (immutable snapshots), `releases/`, `import/` (the prototype state as
  imported), `backups/`, `tests/` (e2e smoke results). Replays are kept in the catalog zips
  (`primitive-bc-data/extra/league/`, ≈ 0.65 MB per episode compressed, ≈ 36 MB as JSON); a loose `replay.json.gz`
  that was never exported is pruned after 30 days.
* Failure behaviour: the worker and API restart under systemd (`Restart=always`). Episodes running when the worker
  stops are re-queued (same seed and seats, attempt + 1, at most 3); a validation interrupted by a restart goes back
  to `pending`. A game over 90 min wall (excluding thermal pauses) is killed and retried. If the NAS is unreachable,
  NFS I/O blocks (hard mount) and games resume when it returns. A submission whose archive changes on disk is
  `failed` before its next game.

## 10. Kaggle replay-download worker (designed, **disabled**)

Not implemented as running code and `kaggle.replay_download_enabled = false`; the league makes **no Kaggle API calls**.
When the maintainer enables it:

* **Shared call counter on the NAS**: `/mnt/ai/scratch/farmNEAT/kaggle-budget/KAGGLE-REQUESTS-<UTC day>.txt` (one
  integer), incremented under an exclusive `flock` on `KAGGLE-REQUESTS.lock` **before** each call; every Kaggle client
  (desktop `pbc-autosubmit` tools included) must use the same file so the ~3,000 calls/UTC-day total is enforced in
  one place (today the desktop keeps its counter in `pbc-autosubmit/KAGGLE-REQUESTS-<day>.txt`; migrating that
  counter is the prerequisite).
* Cadence: hourly (owner: hourly polling for external downloads); cap per run 100 calls and per UTC day 400 calls for
  the league downloader; stop at a global 2,900 so interactive work keeps a reserve; never retry a failed call in the
  same hour.
* Scope: skip episodes already held locally (`catalog.py have`), download only replays of our submissions and of
  the teams named by the maintainer, store them in the catalog's fetch-folder format
  (`extra/<name>/archives/*.zip` + `MANIFEST.jsonl`), then `catalog.py update`.

## 11. Versioning and compatibility

* Contract `1.x`: additive changes only (new endpoints, new fields). Clients must ignore unknown fields.
* Engine snapshots are immutable and named `engine-<date>-<digest8>`; a new snapshot is a config change recorded in
  every episode's `runtime.engine`.
* The prototype league's state (15 entries, 30 games on seeds 31000000-31000014) was imported with
  `source = "import:pbc-league"`; those episodes have no replay (the prototype did not record actions) and ran on the
  desktop runtime (Python 3.12). Imported entries carry `validation.method = "imported"` (with the
  `pbc-selfplay/GATES.jsonl` cold-verify record where one exists); `validation.reverify_imports = true` would
  re-verify them on the league host in the validation slot (off since 2026-09-27 so every slot plays games).
