"""Beat-archive option (owner 2026-09-25: train on the winners of games that current top-30 policies lost)."""
import glob, importlib.util, json, os, sys, tempfile, zipfile
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

SRC = sorted(glob.glob('/mnt/ai/scratch/farmNEAT/primitive-bc-data/extra/topteam-kigasu/archives/*.zip'))


def _orig():
    f = HERE / '_data_orig_for_test.py'   # same directory: data.py resolves repo paths from __file__
    f.write_text((HERE / 'data.py.orig').read_text())
    try:
        spec = importlib.util.spec_from_file_location('data_orig', f)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        f.unlink()
    return m


@pytest.fixture(scope='module')
def zips():
    d = Path(tempfile.mkdtemp())
    z = zipfile.ZipFile(SRC[0]); names = sorted(z.namelist())[:3]
    a = d / 'kaggriculture-topteams-2026-09-10-x.zip'   # an old day: recent_days must not drop it
    b = d / 'kaggriculture-episodes-2026-09-24.zip'      # a "normal" archive holding one of the same games
    with zipfile.ZipFile(a, 'w') as out:
        for n in names:
            out.writestr(n, z.read(n))
    with zipfile.ZipFile(b, 'w') as out:
        out.writestr(names[0], z.read(names[0]))
    return str(a), str(b), names


def test_beat_seats_winner_not_us(zips):
    a, _, names = zips
    for n in names:
        h = data.read_header(a, n)
        s = data.beat_seats(h)
        r = h['rewards']
        assert s == [0 if r[0] > r[1] else 1]
        h2 = json.loads(json.dumps(h)); h2['info']['TeamNames'][s[0]] = data.OUR_TEAM
        assert data.beat_seats(h2) == s                     # our own policy's win over a strong policy is kept
        h2['info']['TeamNames'][1 - s[0]] = data.OUR_TEAM
        assert data.beat_seats(h2) == []                    # our games against ourselves are not


def test_tasks_dedup_and_no_recent_filter(zips):
    a, b, names = zips
    t = data._tasks([b], 'all', None, None, recent_days=1, beat_archives=(a,))
    assert sorted(x[3] for x in t) == sorted(int(n[:-5]) for n in names)
    assert all(x[0] == a for x in t)


def test_beat_path_bypasses_rank_filter(zips):
    a, _, names = zips
    task = next(x for x in data.list_members([a]) if x[1] == names[0])
    kw = dict(rank_max=1, seats='both', leaderboard={}, windows={'window': 24, 'burn_in': 24, 'stride': None})
    assert list(data._episode_task(task, **kw)) == []
    out = list(data._episode_task(task, beat_archives=frozenset([a]), **kw))
    h = data.read_header(a, names[0])
    assert len(out) == 1 and {w['meta']['seat'] for w in out[0]} == set(data.beat_seats(h))


def test_flag_off_identical_to_original(zips):
    _, b, _ = zips
    o = _orig()
    kw = dict(split='all', rank_max=None, seats='both', workers=0, window=24, burn_in=24, shuffle_seed=3, leaderboard={})
    new = list(data.iter_windows([b], **kw)); old = list(o.iter_windows([b], **kw))
    assert len(new) == len(old) > 0
    for x, y in zip(new, old):
        assert x['start_step'] == y['start_step'] and x['meta'] == y['meta']
        assert all((p['rows'] == q['rows']).all() and p['unit_tok'] == q['unit_tok'] for p, q in zip(x['steps'], y['steps']))


def test_beat_ids_recent_exempt_and_lookup(zips):
    a, b, names = zips
    ids = [int(n[:-5]) for n in names]
    # b holds names[0] (day 09-24); a (day 09-10) is used only as a lookup archive for ids[1:]
    t = data._tasks([b], 'all', None, None, recent_days=1, beat_ids=frozenset(ids), beat_lookup=(a,))
    assert sorted(x[3] for x in t) == sorted(ids)
    assert [x[0] for x in t if x[3] == ids[0]] == [b]
    none = data._tasks([b], 'all', None, None, recent_days=1, beat_ids=frozenset([ids[1]]), beat_lookup=(a,))
    assert sorted(x[3] for x in none) == sorted([ids[0], ids[1]])


def test_beat_ids_path(zips):
    _, b, names = zips
    task = next(x for x in data.list_members([b]))
    kw = dict(rank_max=1, seats='both', leaderboard={}, windows={'window': 24, 'burn_in': 24, 'stride': None})
    assert list(data._episode_task(task, **kw)) == []
    out = list(data._episode_task(task, beat_ids=frozenset([task[3]]), **kw))
    assert len(out) == 1 and {w['meta']['seat'] for w in out[0]} == set(data.beat_seats(data.read_header(b, names[0])))


def test_force_teams_keeps_winner_without_rank(zips):
    a, _, names = zips
    h = data.read_header(a, names[0])
    w = data.beat_seats(h)[0]
    team = h['info']['TeamNames'][w]
    _, _, chosen = data.select_seats(h, '2026-09-10', rank_max=1, seats='both', leaderboard={})
    assert chosen == []
    _, _, chosen = data.select_seats(h, '2026-09-10', rank_max=1, seats='both', leaderboard={}, force_teams=frozenset([team]))
    assert chosen == [w]
    loser = h['info']['TeamNames'][1 - w]
    _, _, chosen = data.select_seats(h, '2026-09-10', rank_max=1, seats='both', leaderboard={}, force_teams=frozenset([loser]))
    assert chosen == []
