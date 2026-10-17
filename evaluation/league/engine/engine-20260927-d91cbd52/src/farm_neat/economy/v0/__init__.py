"""V0 plan action space: the network edits a canonical plan; the compiler plays the game.

Frozen vocabulary (vocabulary.py), the plan and its canonical edit set (plan.py), the
public-state projections and masks (projection.py), the compiler from a plan to executor
stages (compiler.py), the epoch controller that runs a policy over the vocabulary in a
native game (epochs.py) and the round trip of retained portfolios and Router tape 0
(roundtrip.py). docs/guides/action-space-v0.md is the contract; every projection here
is deterministic given the observation and a declared flow belief.
"""
