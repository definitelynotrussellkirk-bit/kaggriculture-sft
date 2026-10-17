#!/usr/bin/env python3
"""Current Kaggle leaderboard -> top30-losses/LEADERBOARD-<UTC>.csv, one call on the shared counter. Owner 2026-09-27 (pasted
the live top 12): the puller must follow the CURRENT top teams; fetch_top_slow2.sh runs this at most hourly.
  python3 refresh_lb.py --counter .../KAGGLE-REQUESTS-<day>.txt [--cap N]"""
import argparse
import datetime as dt
import glob
import os
import shutil
import tempfile
import zipfile

import fetch_topteam_replays as F
from farm_top30_losses import Budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counter", required=True)
    ap.add_argument("--cap", type=int, default=100000)
    a = ap.parse_args()
    b = Budget(a.counter, a.cap)
    b.take()   # the leaderboard download
    b.take()   # `import kaggle` authenticates at import: one IntrospectToken request when an access token is configured
    from kaggle import api   # already authenticated; a second authenticate() would be a second IntrospectToken request
    tmpd = tempfile.mkdtemp()
    api.competition_leaderboard_download(F.COMP, tmpd)
    for z in glob.glob(os.path.join(tmpd, "*.zip")):
        zipfile.ZipFile(z).extractall(tmpd)
    src = glob.glob(os.path.join(tmpd, "*.csv"))[0]
    out = f"top30-losses/LEADERBOARD-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M}.csv"
    shutil.copy(src, out)
    print(out)


if __name__ == "__main__":
    main()
