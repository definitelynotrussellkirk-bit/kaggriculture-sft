"""Free-running evaluation of a full-economic MoE checkpoint: SAMPLED programs, real games.

The learner seat runs a FullSeasonPlan controller whose programs are SAMPLED from the
checkpoint at the declared decision steps, exactly the way
``farm_neat.training.campaign.collect`` samples them:

    policy.load_model -> fast.FastEncoder -> pack_plan (full-plan v4 adapter)
    -> campaign.collate_bucket -> encoder(rows, token_mask, knownness)
    -> fast.GraphedSampler.sample(context, streams, max_commands, as_of_step=step)
    -> controller.propose(observation, program) -> controller.publish(...)

Worlds run in lockstep inside one process so every decision step is one batched encode
and one batched sample (lane count == worlds per batch), like the collector.

Opponent: PASS (issues PASS_ACTION every turn) or the frozen ShopRouterAdapter.
Modes:  opening  = publication_steps (0,)                 -- one program at step 0
        five     = publication_steps (0,144,288,432,576)  -- the campaign clock

This is measurement only: no optimizer, no publication, nothing written into any
checkpoint directory. Terminal own cash on unseen worlds is the reported metric.

Usage:
  python3 eval_policy.py --checkpoint DIR --out OUT.json [--games 200] [--mode opening|five]
     [--opponent pass|router] [--seed-base 7000000] [--sampler-base 7500000]
     --max-commands 64 [--worlds-per-batch 20] [--temperature 1.0] [--gpu-fraction 0.25]
"""
from __future__ import annotations

import os  # repository-root resolution below
import argparse
import collections
import copy
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the src/ directory, so farm_neat imports work when run as a script

ROOT = Path(_SRC).parent
sys.path.insert(0, str(ROOT / "src"))

from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, PASS_ACTION
from farm_neat.policy import fast_profile as fast
from farm_neat.policy import plan_observation as plan_features
from farm_neat.policy import network as policy
from farm_neat.economy.orders import FullSeasonPlan
from farm_neat.economy.action_space import MAX_COMMANDS, validate_program
from farm_neat.training import campaign as campaign

FIVE = (0, 144, 288, 432, 576)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    def q(f):
        return float(ordered[min(len(ordered) - 1, int(f * len(ordered)))])
    return {"min": float(ordered[0]), "p25": q(0.25), "median": float(statistics.median(ordered)),
            "p75": q(0.75), "max": float(ordered[-1]), "mean": float(statistics.fmean(ordered))}


def game_key(row):
    return tuple(row[k] for k in ('seed', 'seat', 'sampler_seed', 'opponent'))


def summarize(games, *, requested_games):
    """Version-2 reduction: errors have no outcome and cannot pass a gate."""
    if type(requested_games) is not int or requested_games < 1 or len(games) > requested_games:
        raise ValueError('positive requested count and at most one row per request required')
    if len({game_key(g) for g in games}) != len(games):
        raise ValueError('duplicate evaluation identity')
    good = [g for g in games if 'error' not in g]
    for g in good:
        if any(type(g[k]) not in (int, float) or not math.isfinite(g[k]) or g[k] < 0
               for k in ('own_cash', 'opponent_cash')):
            raise ValueError('finite nonnegative terminal cash required')
        if any(type(g[k]) is not int or g[k] < 0 for k in ('physical_turns', 'rejected_commands')):
            raise ValueError('nonnegative activity/refusal counters required')
        validate_program(g['programs']['0'])
    n = len(good)
    fraction = lambda count: count/n if n else None
    cash = [g['own_cash'] for g in good]
    lengths, ops = [], collections.Counter()
    for g in good:
        body = [c['op'] for c in g['programs']['0'] if c['op'] != 'STOP']
        lengths.append(len(body)); ops.update(body)
    profitable = sum(c > 3000 for c in cash)
    idle = sum(g['physical_turns'] == 0 for g in good)
    stop = sum(length == 0 for length in lengths)
    return {'schema': 'farm-full-economic-free-running-sampled-eval-v2',
            'requested_games': requested_games, 'games': n, 'completed_games': n,
            'error_games': len(games)-n, 'unreported_games': requested_games-len(games),
            'complete': n == requested_games, 'eligible_for_gate': n == requested_games,
            'fraction_denominator': 'COMPLETED_GAMES; INCOMPLETE_PANEL_CANNOT_PASS_GATE',
            'quantile_definition': 'median=MEAN_MIDDLE_PAIR; p25/p75=FLOOR_N_TIMES_P',
            'physical_turns_definition': 'TURNS_WITH_NON_PASS_ACTION; NOT_SIMULATED_STEPS',
            'seeds': [g['seed'] for g in good], 'scored_seeds': [g['seed'] for g in good],
            'profitable_games': profitable, 'profitable_fraction': fraction(profitable),
            'above_10k_fraction': fraction(sum(c > 10000 for c in cash)),
            'own_cash': quantiles(cash), 'opponent_cash': quantiles([g['opponent_cash'] for g in good]),
            'wins': sum(g['own_cash'] > g['opponent_cash'] for g in good),
            'draws': sum(g['own_cash'] == g['opponent_cash'] for g in good),
            'losses': sum(g['own_cash'] < g['opponent_cash'] for g in good),
            'idle_games': idle, 'idle_fraction': fraction(idle),
            'physical_turns': quantiles([g['physical_turns'] for g in good]),
            'stop_only_openings': stop, 'stop_only_fraction': fraction(stop),
            'opening_length': quantiles(lengths), 'opening_op_histogram': dict(ops.most_common()),
            'rejected_commands_total': sum(g['rejected_commands'] for g in good), 'per_game': games}


def compare_panels(baseline, candidate, *, bootstrap_samples=10000, bootstrap_seed=20260913):
    """Complete, one-game-per-world paired comparison; never automatic promotion.

    Exact McNemar tests are exploratory and unadjusted. Percentile bootstrap
    intervals resample whole paired worlds, not individual player outcomes.
    """
    import numpy as np
    common = ('mode', 'decision_steps', 'opponent', 'temperature', 'max_commands',
              'forced_stop', 'adapter_identity_sha256', 'executor_source_sha256',
              'fast_profile_sha256', 'opponent_identity_sha256', 'runtime_profile',
              'evaluator_source_sha256', 'worlds_per_batch')
    for key in common:
        if key not in baseline or baseline[key] != candidate.get(key):
            raise ValueError('paired condition missing or differs: '+key)
    if baseline['opponent'] not in ('pass', 'router'):
        raise ValueError('one declared fixed-opponent stratum required')
    checked = []
    for panel in (baseline, candidate):
        if (panel.get('schema') != 'farm-full-economic-free-running-sampled-eval-v2'
                or panel.get('complete') is not True or panel.get('externally_pinned_model') is not True
                or panel.get('fixed_program') is not None):
            raise ValueError('complete externally pinned neural panel required')
        model_hash = panel.get('model_file_sha256')
        if not isinstance(model_hash, str) or len(model_hash) != 64 or any(c not in '0123456789abcdef' for c in model_hash):
            raise ValueError('exact model identity required')
        reduced = summarize(panel['per_game'], requested_games=panel['requested_games'])
        if not reduced['complete']:
            raise ValueError('incomplete paired rows')
        rows = sorted(reduced['per_game'], key=game_key)
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('one game per independent world required; repeated seats need clustered analysis')
        for r in rows:
            if (any(type(r[k]) is not int or r[k] < 0 for k in ('seed', 'sampler_seed'))
                    or type(r['seat']) is not int or r['seat'] not in (0, 1)
                    or r['opponent'] != panel['opponent'] or r.get('simulated_steps') != 719
                    or set(r['programs']) != {str(s) for s in panel['decision_steps']}):
                raise ValueError('row opponent or complete publication horizon differs')
            for program in r['programs'].values():
                if len(validate_program(program)) > panel['max_commands']:
                    raise ValueError('row exceeds declared command budget')
        checked.append((reduced, rows))
    if [game_key(r) for r in checked[0][1]] != [game_key(r) for r in checked[1][1]]:
        raise ValueError('paired world/seat/sampler/opponent identities differ')
    if baseline['model_file_sha256'] == candidate['model_file_sha256']:
        raise ValueError('comparison requires changed model bytes')
    if type(bootstrap_samples) is not int or not 1000 <= bootstrap_samples <= 100000:
        raise ValueError('bounded explicit bootstrap sample count required')
    n = len(checked[0][1]); rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    def values(row):
        own, other = row['own_cash'], row['opponent_cash']
        return [float(own > 3000), float(row['physical_turns'] == 0), float(own > other),
                float(own > other)+0.5*float(own == other), own-other]
    before, after = [np.asarray([values(r) for r in rows], dtype=np.float64) for _, rows in checked]
    differences = after-before
    replicates = []
    batch_size = max(1, min(256, 262144//n))
    for start in range(0, bootstrap_samples, batch_size):
        indices = rng.integers(n, size=(min(batch_size, bootstrap_samples-start), n))
        replicates.append(differences[indices].mean(axis=1))
    bounds = np.quantile(np.concatenate(replicates), (0.025, 0.975), axis=0)
    metrics = {}
    for i, name in enumerate(('profitable', 'idle', 'win', 'wdl_score', 'cash_margin')):
        item = {'baseline_mean': float(before[:, i].mean()), 'candidate_mean': float(after[:, i].mean()),
                'paired_mean_change': float(differences[:, i].mean()),
                'paired_world_bootstrap_95_interval': bounds[:, i].tolist()}
        if i < 3:
            gained = int(((before[:, i] == 0) & (after[:, i] == 1)).sum())
            lost = int(((before[:, i] == 1) & (after[:, i] == 0)).sum())
            discordant = gained+lost
            p = min(1.0, 2*sum(math.comb(discordant, k) for k in range(min(gained, lost)+1))/2**discordant)
            item.update(gained=gained, lost=lost, exact_paired_p=p, lower_is_better=name == 'idle')
        metrics[name] = item
    return {'schema': 'farmneat-paired-fixed-opponent-comparison-v1', 'complete': True,
            'conditions': {k: baseline[k] for k in common}, 'paired_games': n, 'independent_worlds': n,
            'baseline_model_sha256': baseline['model_file_sha256'],
            'candidate_model_sha256': candidate['model_file_sha256'], 'metrics': metrics,
            'wdl': {role: {k: checked[i][0][k] for k in ('wins', 'draws', 'losses')}
                    for i, role in enumerate(('baseline', 'candidate'))},
            'uncertainty': {'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': bootstrap_seed,
                            'unit': 'WHOLE_PAIRED_WORLD', 'interval': 'PERCENTILE',
                            'tests': 'EXACT_TWO_SIDED_MCNEMAR_EXPLORATORY_UNADJUSTED',
                            'limits': 'No multiplicity or repeated-selection adjustment; degenerate bootstrap intervals do not prove equality'},
            'screening_rollback': metrics['profitable']['paired_mean_change'] <= -0.10
                                  or metrics['idle']['paired_mean_change'] >= 0.10,
            'screening_scope': '10_PERCENTAGE_POINT_OPERATIONAL_SCREEN; NOT_SIGNIFICANCE_OR_CAUSALITY',
            'promotion': False}


def comparison_main(argv):
    parser = argparse.ArgumentParser(description='Compare exact completed paired evaluation files; no gameplay')
    for name in ('baseline', 'candidate'):
        parser.add_argument('--'+name, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists():
        parser.error('comparison output exists; preserve prior evidence')
    panels, identities = [], {}
    for role in ('baseline', 'candidate'):
        path = Path(getattr(args, role)); expected = getattr(args, role+'_sha256')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            parser.error('exact '+role+' panel hash required')
        panels.append(json.loads(payload)); identities[role] = {'path': str(path), 'sha256': expected}
    result = {**compare_panels(*panels), 'panels': identities,
              'retention': {'purpose': 'current prospective fixed-opponent decision',
                            'remove_when': 'no current selection or unresolved regression depends on this comparison'}}
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def evaluate_registered(batch_fn, *, output, metadata, games, lanes, seed_base, sampler_base, opponent):
    """Exact tail sizes, durable attempted/completed batches, atomic final summary.

    On a failed batch every scheduled identity is conservatively marked exposed;
    its exact physical-game count is unknown. Completed earlier batches survive.
    No retries, discarded lanes, or overwritten historical receipts.
    """
    if any(type(v) is not int or v < 1 for v in (games, lanes)):
        raise ValueError('positive games and lanes required')
    output = Path(output)
    journal_path = output.with_suffix(output.suffix + '.batches.jsonl')
    if output.exists() or journal_path.exists():
        raise ValueError('evaluation output already exists; preserve prior evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, exposed = [], []
    started = time.monotonic()
    with journal_path.open('x', encoding='utf-8') as journal:
        def event(kind, **payload):
            journal.write(json.dumps({'event': kind, **payload}, sort_keys=True, allow_nan=False)+'\n')
            journal.flush(); os.fsync(journal.fileno())
        event('REGISTERED', metadata=metadata, requested_games=games, worlds_per_batch=lanes,
              seed_base=seed_base, sampler_base=sampler_base, opponent=opponent,
              retention={'purpose': 'evaluation exposure and completed batch evidence',
                         'remove_when': 'no active selection, reproduction or seed-exposure dependency'})
        def finish():
            result = {**metadata, **summarize(rows, requested_games=games),
                      'worlds_per_batch': lanes, 'attempted_games': len(exposed),
                      'physical_games': len(rows) if all('error' not in r for r in rows) else None,
                      'completed_physical_games': sum('error' not in r for r in rows),
                      'padding_games': 0, 'exposed_seeds': [g['seed'] for g in exposed],
                      'exposed_identities': exposed, 'exposure_policy': 'ATTEMPTED_BATCH_IS_EXPOSED_EVEN_ON_FAILURE',
                      'batch_receipt': journal_path.name, 'seconds': time.monotonic()-started}
            with tempfile.NamedTemporaryFile('w', dir=output.parent, prefix=output.name+'.',
                                             suffix='.tmp', delete=False) as stream:
                staged = Path(stream.name)
                try:
                    json.dump(result, stream, indent=1, sort_keys=True, allow_nan=False)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                except BaseException:
                    staged.unlink(); raise
            try:
                os.replace(staged, output)
            finally:
                staged.unlink(missing_ok=True)
            return result
        for start in range(0, games, lanes):
            slots = [{'seed': seed_base+i, 'seat': i % 2, 'sampler_seed': sampler_base+i,
                      'opponent': opponent} for i in range(start, min(start+lanes, games))]
            exposed.extend(slots)
            event('BATCH_STARTED', identities=slots)
            try:
                batch = batch_fn(slots)
                if len(batch) != len(slots) or [game_key(g) for g in batch] != [game_key(g) for g in slots]:
                    raise ValueError('batch identities differ from registration')
                reduction = summarize(batch, requested_games=len(slots))
                if not reduction['complete']:
                    rows.extend(batch)
                    event('BATCH_INCOMPLETE', per_game=batch)
                    raise RuntimeError('evaluation batch incomplete; failed attempts preserved')
            except BaseException as error:
                if not rows or game_key(rows[-1]) != game_key(slots[-1]):
                    failed = [{**g, 'error': f'{type(error).__name__}: {error}'} for g in slots]
                    rows.extend(failed)
                    event('BATCH_FAILED', per_game=failed)
                finish()
                raise
            rows.extend(batch)
            event('BATCH_COMPLETED', per_game=batch)
            print(f'{len(rows)}/{games} games; median own cash {quantiles([g["own_cash"] for g in rows])["median"]}', flush=True)
        result = finish()
        event('FINALIZED', complete=result['complete'], games=result['games'])
        return result


def run_batch(graph, encoder, sampler, adapter_identity, *, seeds, seats, opponent_kind,
              steps, max_commands, sampler_seeds, fixed_program=None):
    """One lockstep batch of len(seeds) worlds; returns one result dict per world."""
    import torch
    device = next(graph.parameters()).device
    n = len(seeds)
    envs = [ExactKaggricultureEnvironment(seed=s) for s in seeds]
    controllers = [FullSeasonPlan(f"eval-{seeds[i]}-{seats[i]}", publication_steps=steps)
                   for i in range(n)]
    others = []
    if opponent_kind == "router":
        from farm_neat.evaluation.shop_router import ShopRouterAdapter
        others = [ShopRouterAdapter() for _ in range(n)]
    else:
        others = [None] * n
    streams = [fast.LaneStream(int(s)) for s in sampler_seeds]
    programs_by_step = [dict() for _ in range(n)]
    rejected_counts = [0] * n
    physical = [0] * n
    for step in range(719):
        observations = []
        for i, env in enumerate(envs):
            if env.done or env.step_index != step:
                raise ValueError(f"world {i} left the synchronous horizon at step {step}")
            obs = env.agent_observation(seats[i])
            controllers[i].observe(obs)
            observations.append(obs)
        if step in steps:
            packs = [plan_features.pack(observations[i], controllers[i], envs[i].configuration,
                                        expected_identity=adapter_identity) for i in range(n)]
            batch = campaign.collate_bucket(packs)
            with torch.no_grad():
                tensors = tuple(torch.as_tensor(batch[k], device=device)
                                for k in ("rows", "token_mask", "knownness"))
                context, _values, _ = encoder(*tensors)
                torch.cuda.synchronize()
                records, totals, _ = sampler.sample(context, streams, max_commands=max_commands,
                                                    as_of_step=step)
                del context, tensors, totals
            torch.cuda.empty_cache()
            for i in range(n):
                program = records[i]["program"]
                if fixed_program is not None:
                    program = ([*copy.deepcopy(fixed_program), {"op": "STOP"}] if step == 0
                               else [{"op": "STOP"}])
                programs_by_step[i][step] = program
                proposal = controllers[i].propose(observations[i], program)
                rejected_counts[i] += len(proposal["rejected"])
                controllers[i] = controllers[i].publish(observations[i], proposal)
        for i, env in enumerate(envs):
            joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]
            action = controllers[i](observations[i], env.configuration)
            joint[seats[i]] = action
            if action != PASS_ACTION:
                physical[i] += 1
            if others[i] is not None:
                joint[1 - seats[i]] = others[i].act(env.agent_observation(1 - seats[i]),
                                                    env.configuration)
            env.step(joint)
            if others[i] is not None:
                others[i].observe(env.agent_observation(1 - seats[i]))
    out = []
    for i, env in enumerate(envs):
        farms = env.agent_observation(seats[i])["farms"]
        out.append({"seed": seeds[i], "seat": seats[i], "opponent": opponent_kind,
                    "sampler_seed": int(sampler_seeds[i]),
                    "own_cash": float(farms[seats[i]]["money"]),
                    "opponent_cash": float(farms[1 - seats[i]]["money"]),
                    "physical_turns": physical[i], "simulated_steps": 719,
                    "rejected_commands": rejected_counts[i],
                    "programs": {str(k): v for k, v in programs_by_step[i].items()}})
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['compare']:
        return comparison_main(argv[1:])
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-sha256", help="externally pin the selected model before any gameplay")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--mode", choices=("opening", "five"), default="opening")
    p.add_argument("--opponent", choices=("pass", "router"), default="pass")
    p.add_argument("--seed-base", type=int, default=7000000)
    p.add_argument("--sampler-base", type=int, default=7500000)
    p.add_argument("--worlds-per-batch", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-commands", type=int, required=True,
                   help="explicit slots including forced final STOP; matched saved profile uses 64")
    p.add_argument("--gpu-fraction", type=float, default=0.25)
    p.add_argument("--label", default="")
    p.add_argument("--fixed-program", default=None,
                   help="control: force this program at step 0 (STOP later); the network is still "
                        "loaded and sampled so the loop is identical")
    a = p.parse_args(argv)
    if a.games < 1 or a.worlds_per_batch < 1:
        p.error('games and worlds-per-batch must be positive')
    if not math.isfinite(a.temperature) or a.temperature <= 0:
        p.error('temperature must be finite and positive')
    if not math.isfinite(a.gpu_fraction) or not 0 < a.gpu_fraction <= 1:
        p.error('gpu-fraction must be in (0,1]')
    if not 2 <= a.max_commands <= MAX_COMMANDS or min(a.seed_base, a.sampler_base) < 0:
        p.error('supported command budget and nonnegative seed bases required')
    if Path(a.out).exists() or Path(a.out+'.batches.jsonl').exists():
        p.error('evaluation output already exists; preserve prior evidence')

    fixed_program = None
    if a.fixed_program:
        with open(a.fixed_program, encoding='utf-8') as stream:
            loaded = json.load(stream)
        if isinstance(loaded, dict) and 'program' in loaded:
            loaded = loaded['program']
        if isinstance(loaded, list) and loaded and isinstance(loaded[0], dict) and 'program' in loaded[0]:
            loaded = loaded[0]['program']
        fixed_program = [c for c in loaded if c.get('op') != 'STOP']
        complete_program = validate_program([*fixed_program, {'op':'STOP'}])
        if len(complete_program) > a.max_commands:
            p.error('fixed control exceeds the explicit budget including STOP')

    import torch
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    checkpoint = Path(a.checkpoint)
    model_path = checkpoint / "model.pt"
    model_file_sha = file_sha(model_path)
    if a.model_sha256 is not None and a.model_sha256 != model_file_sha:
        p.error('selected model hash differs; no gameplay performed')
    graph, manifest = policy.load_model(model_path, device="cuda",
                                        expected_file_sha256=model_file_sha, allow_remaining_window_transition=True)
    graph.eval()
    adapter_identity = plan_features.identity()
    steps = (0,) if a.mode == "opening" else FIVE
    lanes = min(a.worlds_per_batch, a.games)
    encoder = fast.FastEncoder(graph, expected_identity=fast.identity())
    sampler = None
    sampler_lanes = None
    from farm_neat.evaluation import shop_router
    opponent_identity = (shop_router.identity()['content_sha256'] if a.opponent == 'router'
                         else hashlib.sha256(json.dumps(PASS_ACTION, sort_keys=True).encode()).hexdigest())

    metadata = {
        "label": a.label, "fixed_program": a.fixed_program, "checkpoint": str(checkpoint),
        "fixed_program_sha256": file_sha(a.fixed_program) if a.fixed_program else None,
        "model_file_sha256": model_file_sha, "model_manifest_sha256": manifest["content_sha256"],
        "externally_pinned_model": a.model_sha256 is not None,
        "runtime_profile": {"python": sys.version, "torch": str(torch.__version__),
                            "numpy": str(sys.modules['numpy'].__version__),
                            "device": torch.cuda.get_device_name(), "dtype": str(next(graph.parameters()).dtype),
                            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                            "autocast": bool(torch.is_autocast_enabled('cuda')),
                            "threads": torch.get_num_threads()},
        "fast_profile_sha256": fast.identity()['content_sha256'],
        "opponent_identity_sha256": opponent_identity,
        "evaluator_source_sha256": file_sha(__file__),
        "model_optimizer_steps": manifest["optimizer_steps"],
        "mode": a.mode, "decision_steps": list(steps), "opponent": a.opponent,
        "temperature": a.temperature, "max_commands": a.max_commands,
        "forced_stop": "FINAL_SLOT_IS_STOP_ONLY; BUDGET_INCLUDES_STOP",
        "seed_base": a.seed_base, "sampler_base": a.sampler_base,
        "adapter_identity_sha256": adapter_identity["content_sha256"],
        "executor_source_sha256": campaign.file_sha(ROOT / "src/farm_neat/economy/orders.py"),
    }
    def batch_fn(slots):
        nonlocal sampler, sampler_lanes
        if sampler_lanes != len(slots):
            # Capture the actual remainder shape, never create dummy physical worlds.
            sampler = None
            torch.cuda.empty_cache()
            sampler = fast.GraphedSampler(graph, len(slots), a.temperature)
            sampler_lanes = len(slots)
        return run_batch(graph, encoder, sampler, adapter_identity,
                         seeds=[s['seed'] for s in slots], seats=[s['seat'] for s in slots],
                         sampler_seeds=[s['sampler_seed'] for s in slots],
                         opponent_kind=a.opponent, steps=steps, max_commands=a.max_commands,
                         fixed_program=fixed_program)
    result = evaluate_registered(batch_fn, output=a.out, metadata=metadata, games=a.games,
                                 lanes=lanes, seed_base=a.seed_base, sampler_base=a.sampler_base,
                                 opponent=a.opponent)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('per_game', 'seeds', 'scored_seeds', 'exposed_seeds', 'exposed_identities')},
                     indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
