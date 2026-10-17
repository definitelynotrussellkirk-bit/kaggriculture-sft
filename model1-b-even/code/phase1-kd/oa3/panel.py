"""Complete two-seat world clusters; approximate inference, never auto-promotion."""
from __future__ import annotations
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

BASELINES = ('B-u25000-greedy', 'OA1-u65000-greedy')

def _t_quantile(p: float, df: int) -> float:
    # Cornish-Fisher expansion through O(df^-3), only used for n>=16 worlds.
    z = statistics.NormalDist().inv_cdf(p)
    return z + (z**3+z)/(4*df) + (5*z**5+16*z**3+3*z)/(96*df**2) + (3*z**7+19*z**5+17*z**3-15*z)/(384*df**3)

def reduce_pairs(rows: list[dict], *, expected_worlds: int, manifest: dict | None = None) -> dict:
    if expected_worlds < 16:
        raise ValueError('At least 16 independent world clusters required')
    groups = defaultdict(dict)
    for r in rows:
        opponent, world, seat = r['opponent'], r['world'], r['seat']
        if opponent not in BASELINES or not isinstance(world, int) or isinstance(world, bool) or seat not in (0, 1):
            raise ValueError('Unknown baseline, noninteger world, or invalid seat')
        score, margin = float(r['score']), float(r['margin'])
        if score not in (0.0, .5, 1.0) or not math.isfinite(margin) or type(r['ok']) is not bool:
            raise ValueError('Need finite margin, terminal win/draw/loss score, boolean ok')
        if r['ok'] and ((margin > 0 and score != 1) or (margin < 0 and score != 0) or (margin == 0 and score != .5)):
            raise ValueError('Terminal score and candidate-minus-opponent margin disagree')
        key = (world, seat)
        if key in groups[opponent]:
            raise ValueError(f'Duplicate panel game: {opponent}/{key}')
        groups[opponent][key] = r
    if set(groups) != set(BASELINES):
        raise ValueError('Exactly both protected baselines are required')
    worlds = None
    stats = {}
    for opponent in BASELINES:
        g = groups[opponent]
        ws = {w for w, seat in g}
        if len(ws) != expected_worlds or set(g) != {(w, s) for w in ws for s in (0, 1)}:
            raise ValueError(f'Missing worlds/seats for {opponent}')
        if worlds is not None and ws != worlds:
            raise ValueError('Baselines must share exactly the same world set')
        worlds = ws
        scores = [sum(float(g[(w,s)]['score']) for s in (0,1))/2 for w in sorted(ws)]
        margins = [sum(float(g[(w,s)]['margin']) for s in (0,1))/2 for w in sorted(ws)]
        n = len(scores)
        # One-sided familywise alpha .05, Bonferroni over TWO baselines.
        t = _t_quantile(.975, n-1)
        lo = max(0., statistics.mean(scores) - t*statistics.stdev(scores)/math.sqrt(n))
        sorted_m = sorted(margins)
        stats[opponent] = {'worlds': n, 'games': 2*n,
            'score': statistics.mean(scores), 'score_lower_approx_adjusted': lo,
            'cash_margin_mean': statistics.mean(margins), 'cash_margin_worst_world': min(margins),
            'cash_margin_bottom_quartile_mean': statistics.mean(sorted_m[:max(1,n//4)]),
            'crashes': sum(not r['ok'] for r in g.values()),
            'world_scores': dict(zip(map(str, sorted(ws)), scores))}
    provenance_verified = False
    if manifest is not None:
        if set(manifest['worlds']) != worlds or manifest.get('phase') not in ('screen','confirm'):
            raise ValueError('Results do not match pre-registered world/phase manifest')
        required = ('candidate_sha256','baseline_sha256','decoder','rules','rng_mode','world_registry_sha256')
        for key in required:
            if key not in manifest or not manifest[key]:
                raise ValueError(f'Manifest missing {key}')
        for r in rows:
            for key in ('candidate_sha256', 'decoder', 'rules', 'rng_mode'):
                if r.get(key) != manifest[key]:
                    raise ValueError(f'Game has wrong or missing {key}')
            if r.get('baseline_sha256') != manifest['baseline_sha256'][r['opponent']]:
                raise ValueError('Game baseline hash differs from manifest')
        provenance_verified = True
    eligible = all(s['score'] >= .60 and s['score_lower_approx_adjusted'] > .50 and not s['crashes'] for s in stats.values())
    return {'opponents': stats, 'eligible_for_owner_review': eligible,
        'provenance_verified': provenance_verified,
        'confirmatory_gate_passed': bool(eligible and provenance_verified and manifest['phase']=='confirm' and expected_worlds>=32),
        'method': 'Two-seat world means; approximate Student t, one-sided Bonferroni alpha=.05 over two baselines.',
        'warning': 'Not exact coverage; constant samples yield zero estimated variance. No rating estimate or automatic promotion. Native RNG/cold archive gates remain.'}

def make_plan(*, candidate, baselines: dict, rules_file, registry: dict,
              phase: str, seed: int=930, screen: dict | None=None) -> dict:
    """Freeze artifact identity and draw worlds without touching the input registry.

    Freshness is relative to the caller's registry only. This is NOT a claim that
    the source registry includes every selection/training/holdout world.
    """
    import hashlib
    import random
    from .source import sha256
    if phase not in ('screen','confirm') or set(baselines)!=set(BASELINES):
        raise ValueError('Need screen/confirm and exactly both baselines')
    if not isinstance(registry.get('used_worlds'),list) or not registry.get('provenance'):
        raise ValueError('Need an owner-provenanced used_worlds registry')
    if any(type(w) is not int for w in registry['used_worlds']):
        raise ValueError('Registry world IDs must be integers')
    result={'phase':phase,'candidate_sha256':sha256(candidate),
        'baseline_sha256':{name:sha256(path) for name,path in baselines.items()},
        'decoder':'greedy-temp0','rules':sha256(rules_file),'rng_mode':'crn',
        'world_registry_sha256':hashlib.sha256(json.dumps(registry,sort_keys=True).encode()).hexdigest(),
        'seed':seed,'worlds':[],'freshness_scope':'Only the supplied owner registry; not independently certified.',
        'run_contract':'Both seats, same world and exact artifacts; terminal cash margin from candidate perspective.'}
    used=set(registry['used_worlds'])
    if phase=='confirm':
        if screen is None or screen.get('phase')!='screen':
            raise ValueError('Confirmation requires the prior screen manifest')
        for key in ('candidate_sha256','baseline_sha256','decoder','rules','rng_mode'):
            if screen.get(key)!=result[key]:raise ValueError(f'Identity changed since screen: {key}')
        used.update(screen['worlds'])
        result['screen_manifest_sha256']=hashlib.sha256(json.dumps(screen,sort_keys=True).encode()).hexdigest()
    rng=random.Random(seed)
    for _ in range(16 if phase=='screen' else 32):
        while True:
            w=rng.randrange(1,2**31)
            if w not in used:break
        used.add(w);result['worlds'].append(w)
    result['games']=len(result['worlds'])*2*len(BASELINES)
    result['jobs']=[{'opponent':name,'world':world,'seat':seat} for name in BASELINES for world in result['worlds'] for seat in (0,1)]
    return result
