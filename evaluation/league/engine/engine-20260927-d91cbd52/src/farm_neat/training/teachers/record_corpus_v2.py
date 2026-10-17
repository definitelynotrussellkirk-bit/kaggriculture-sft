"""Play evolved programs on many seeds/seats and record decision-step observations for BC.

Per game: at each decision step (0,144,288,432,576) store the agent observation and the
FullSeasonPlan snapshot BEFORE the step's program is applied, the program applied at that
step (full program at 0, STOP-only later), and the terminal cash. Opponent: PASS or ShopRouter.

  record_corpus_v2.py HALL.json OUT.jsonl.gz [programs] [seeds-per-program] [workers]
"""
import copy, gzip, json, os, sys, time
from multiprocessing import Pool
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the src/ directory, so farm_neat imports work when run as a script
sys.path.insert(0, _SRC)
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment, PASS_ACTION
STEPS = (0, 144, 288, 432, 576)

def play(args, *, revisions=None, capture=False, timeout=None):
    """Existing five-publication game with optional explicit revision witnesses.

    Defaults retain the original corpus schema. Detailed captures are a separate
    counterfactual-data interface, not ordinary BC examples or sampled behavior.
    """
    prog, seed, seat, opp, pid = args
    from farm_neat.economy.action_space import validate_program
    if revisions is None: revisions = {}
    if (not isinstance(revisions, dict) or any(type(k) is not int or k not in STEPS[1:] for k in revisions)
            or type(capture) is not bool or (timeout is not None and not 0 < timeout <= 300)):
        raise ValueError('explicit eligible revisions and bounded capture required')
    revisions = {k:validate_program(v) for k,v in revisions.items()}
    from farm_neat.economy.orders import FullSeasonPlan
    from farm_neat.evaluation.shop_router import ShopRouterAdapter
    env = ExactKaggricultureEnvironment(seed=seed)
    ctl = FullSeasonPlan(f"corpus-{pid}-{seed}-{seat}", publication_steps=STEPS)
    other = ShopRouterAdapter() if opp == "router" else None
    records, publications, actions = [], [], []
    started = time.monotonic()
    try:
        while not env.done:
            if timeout is not None and time.monotonic()-started > timeout:
                raise TimeoutError(f'program collection bound at step{env.step_index}')
            obs = env.agent_observation(seat)
            if env.step_index in STEPS:
                ctl.observe(obs)
                program = [*prog, {"op": "STOP"}] if env.step_index == 0 else revisions.get(env.step_index, [{"op": "STOP"}])
                records.append({"step": env.step_index, "observation": copy.deepcopy(obs), "snapshot": ctl.snapshot(), "program": program})
                proposal = ctl.propose(obs, program); ctl = ctl.publish(obs, proposal)
                if capture:
                    publications.append({'step':env.step_index, 'proposal':copy.deepcopy(proposal),
                        'opponent_snapshot':None if other is None else copy.deepcopy(other.snapshot())})
            joint = [copy.deepcopy(PASS_ACTION), copy.deepcopy(PASS_ACTION)]
            joint[seat] = ctl(obs, env.configuration)
            if other is not None: joint[1 - seat] = other.act(env.agent_observation(1 - seat), env.configuration)
            env.step(joint); ctl.observe(env.agent_observation(seat))
            if capture: actions.append(copy.deepcopy(joint))
            if other is not None: other.observe(env.agent_observation(1 - seat))
        f = env.agent_observation(seat)["farms"]
        result = {"program_id": pid, "seed": seed, "seat": seat, "opponent": opp, "cash": [float(f[seat]["money"]), float(f[1 - seat]["money"])],
                "configuration": env.configuration, "records": records}
        if capture:
            result['capture']={'schema':'farm-explicit-program-revision-capture-v1',
                'physical_steps':env.step_index,'joint_actions':actions,'publications':publications,
                'terminal_observation':env.agent_observation(seat),'seconds':time.monotonic()-started,
                'neural_samples':0,'optimizer_updates':0}
        return result
    except Exception as e:
        result = {"program_id": pid, "seed": seed, "seat": seat, "opponent": opp, "error": repr(e)[:300]}
        if capture: result['capture']={'physical_steps':env.step_index,'joint_actions':actions,
            'publications':publications,'records':records,'incomplete':True}
        return result

def main(hall_path, out_path, top=12, seeds=40, workers=28):
    hall = json.load(open(hall_path))
    progs = [h["program"] for h in hall[:top]]
    jobs = [(p, 5100000 + s, s % 2, "router" if s % 4 < 2 else "pass", i) for i, p in enumerate(progs) for s in range(seeds)]
    with Pool(min(workers, len(jobs))) as pool, gzip.open(out_path, "wt") as out:  # never more workers than games
        n = 0; errs = 0; cash = []
        for r in pool.imap_unordered(play, jobs, chunksize=1):
            out.write(json.dumps(r) + "\n"); n += 1
            if "error" in r: errs += 1
            else: cash.append(r["cash"][0])
    cash.sort()
    print(f"games {n} errors {errs} own cash: min {cash[0]:.0f} p25 {cash[len(cash)//4]:.0f} median {cash[len(cash)//2]:.0f} max {cash[-1]:.0f}; >3000: {sum(c>3000 for c in cash)}/{len(cash)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], top=int(sys.argv[3]) if len(sys.argv) > 3 else 12,
         seeds=int(sys.argv[4]) if len(sys.argv) > 4 else 40,
         workers=int(sys.argv[5]) if len(sys.argv) > 5 else 28)
