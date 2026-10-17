#!/usr/bin/env python3
"""N flow harness fallback semantics (review PRO/ASTRA CHECK-NFLOW-DEPLOY #2), on real exact-engine observations:
1. the decoder raises at step K (before act returns): the agent submits PASS, the seat's clock advances to K, the
   recurrent memory is RESET (not left describing K-1), and the committed history keeps step K-1's action plus
   step K's PASS;
2. step K+1 decodes normally from fresh memory with that history; no further fallbacks;
3. a rule failure AFTER act returned at step J (pending state exists) keeps the memory act committed and records PASS.
  python3 test_fallback_memory.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402

import agent as AG  # noqa: E402
import model as MD  # noqa: E402
from farm_neat.game.exact_engine import ExactKaggricultureEnvironment  # noqa: E402


def main():
    torch.manual_seed(0)
    net = MD.PolicyNet().eval()
    ag = AG.Agent("preloaded", policy=net, log=False)
    env = ExactKaggricultureEnvironment(seed=20260930)
    K, J, seat = 10, 16, 0
    real_decide, real_rule = net.decide, AG.sell_rule
    fails = []
    for t in range(20):
        obs = env.agent_observation(seat)
        if t == K:
            net.decide = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected decoder failure"))
        if t == J:
            AG.sell_rule = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected rule failure"))
        act = ag(obs, env.configuration)
        net.decide, AG.sell_rule = real_decide, real_rule
        st = ag._states[seat]["state"]
        steps = [int(h[0]) for h in st.hist]
        if t == K:
            if not ag.last_fallback or act != AG.pass_action():
                fails.append("decoder failure did not fall back to PASS")
            if ag._states[seat]["last_step"] != K or st.mem is not None:
                fails.append(f"after decoder failure: last_step={ag._states[seat]['last_step']} mem={'set' if st.mem is not None else None}")
            if steps != [K - 1, K]:
                fails.append(f"history after decoder failure {steps} != [{K - 1}, {K}]")
        elif t == K + 1:
            if ag.last_fallback or st.mem is None or steps != [K, K + 1]:
                fails.append(f"step after failure: fallback={ag.last_fallback} mem={'set' if st.mem is not None else None} hist={steps}")
        elif t == J:
            if not ag.last_fallback or st.mem is None or ag._states[seat]["last_step"] != J:
                fails.append("rule failure after act: memory should be kept and the clock advanced")
            pass_lab = AG.pass_action()
            if steps[-1] != J:
                fails.append("rule failure: PASS not recorded at J")
        env.step([act if s == seat else AG.pass_action() for s in range(2)])
    print({"fallbacks": ag.fallbacks, "fails": fails})
    print("PASS" if not fails and ag.fallbacks == 2 else "FAIL")
    return 0 if not fails and ag.fallbacks == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
