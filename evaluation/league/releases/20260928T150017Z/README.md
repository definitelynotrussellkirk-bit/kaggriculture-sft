# farmNEAT league service

Always-on local league of our shippable Kaggle archives on the R730xd (CT118 `league`, **http://league.lan/**).
The contract (API, statuses, matchmaking, ratings, thermal guard, retention) is [CONTRACT.md](CONTRACT.md).

## Use

```bash
C=/home/russ/Desktop/farmNEAT/work/runs/maintainer/pbc-league-service/league_client.py
python3 $C board                       # leaderboard (leader = best active entry)
python3 $C submit v5-g1-51236 /mnt/ai/scratch/farmNEAT/pbc-deploy/<dir>/<archive>.tar.gz --note "..." --wait
python3 $C status v5-g1-51236          # validation (cold verify both seats, 0 fallbacks) + rating + status log
python3 $C episodes --submission v5-g1-51236
python3 $C h2h v5-g1-51236            # head-to-head record against each opponent
python3 $C replay 1234 --out 1234.json # Kaggle-format replay
python3 $C health                      # worker, thermal, games/hour
python3 $C sync-gates                  # register gated candidates from pbc-selfplay/GATES.jsonl
```

Write commands read the upload token from `$LEAGUE_TOKEN` or the **farmNEAT league** section of
`LAN/secrets.txt` at run time.

## Files

| File | Role |
|---|---|
| `league_core.py` | config, SQLite schema, archive checks, Bradley-Terry + Fisher intervals, matchmaking, thermal rule |
| `league_api.py` | HTTP API + dashboard (stdlib `ThreadingHTTPServer`), `league-api.service` |
| `league_worker.py` | validation queue, match scheduling, game processes, result ingestion, ratings, thermal guard, `league-worker.service` |
| `league_game.py` | one episode: `pbc-ladder/ladder.py` `play()` from the engine snapshot + action recording + Kaggle replay rebuild (runs on the py3.11 runtime) |
| `league_client.py` | CLI |
| `league_import.py` | import of the desktop prototype (`pbc-league/`) state |
| `league_backup.py` | 15-minute SQLite backup to the NAS, hourly snapshots, replay retention (`league-backup.timer`) |
| `league_export.py` | hourly zips + `MANIFEST.jsonl` of finished replays for the local catalog (`primitive-bc-data/extra/league`), run by the backup timer |
| `league_stats.py` | episodes/hour, episode wall time and step latency from the live database (run on the container) |
| `build_snapshot.py` | immutable engine snapshot on the NAS (`league/engine/<id>`) |
| `replay-template.json` | Kaggle replay header (official 2026-09-26 replay) |
| `deploy/` | systemd units, `league.toml`, `deploy.sh` |
| `host/` | the read-only IPMI thermal feed installed on pxmx (`league-thermal.timer`) |
| `tests/` | unit tests and the end-to-end smoke test |

## Deploy, rollback, operate

```bash
deploy/deploy.sh --drain             # pause, let running games finish, new release, switch, restart, resume
deploy/deploy.sh                     # same without draining (running episodes are re-queued and replayed)
deploy/deploy.sh --api-only          # docs/API-only change: restart only league-api (refused if worker files changed)
deploy/deploy.sh --list              # releases on the container
deploy/deploy.sh --rollback <stamp>  # previous code; the database is not touched
```

* Config: `/etc/league/league.toml` on the container (copy of `deploy/league.toml`); `systemctl restart league-worker`
  after a change. Concurrency: `games.max_concurrent_games`. `games.crn_rng = true` plays new episodes with the opt-in
  CRN dusk (CONTRACT.md section 4; needs an engine snapshot with `src/farm_neat/rollout/crn_rng.py`).
* New engine (game code) snapshot: `python3 build_snapshot.py` on the desktop (writes the NAS), then set
  `games.engine` and restart the worker. Old snapshots stay for the record.
* Logs: `journalctl -u league-worker -u league-api` on the container (`ssh root@league.lan`); game logs in each
  episode folder on the NAS.
* Pause/drain: `league_client.py pause` / `resume`; stop everything: `systemctl stop league-worker` (running episodes
  are re-queued on the next start).
* Restore the database: `systemctl stop league-worker league-api`; copy
  `/mnt/ai/scratch/farmNEAT/league/backups/db/league-latest.sqlite3` (or an hourly `.gz` snapshot, gunzipped) to
  `/var/lib/league/league.sqlite3` (owner league); start both.
* Tests: `python3 -m pytest -q tests/test_core.py tests/test_api.py` (17 unit/API tests, no games, about 3 s);
  on the container `runuser -u league -- python3 /opt/league/current/tests/e2e_smoke.py --archives A.tar.gz B.tar.gz`
  runs a throwaway instance (port 8099, own DB): upload + path registration, the real cold-verify validation, a
  thermal pause/resume on live game processes, 2 matches of 40-step games, a replay check and the board.

## Host pieces (recorded in the LAN repo)

CT118 on pxmx (privileged, `nesting=1,mount=nfs`, 16 cores, 32 GB, 32 GB rootfs on `local-lvm`, `onboot=1`,
static lease 192.168.88.61 on the CCR2004, `league.lan` in AdGuard + router). Host timer `league-thermal.timer`
(`/usr/local/sbin/league-thermal-read`) writes `/var/lib/league-thermal/thermal.json` (mp0, read-only in the CT).
Removal: stop the CT, `pct destroy 118`, disable and remove the timer files, remove the lease and DNS name
(`dns_names.py rm league.lan`).
