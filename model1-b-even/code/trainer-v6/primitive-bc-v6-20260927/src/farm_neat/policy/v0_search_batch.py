"""Whole-wave V0 scorer adapter for batched stochastic search.

``make_tensor_decoder`` binds an already constructed :class:`V0Scorer` to the
``BatchClassPaths`` callback seam.  It does not load, move, clone, compile, or
otherwise rebind the model.  Every current live-prefix wave is obtained only
through ``PrefixTensorFrame.score_inputs()``, scored in one ``score_batch``
call, and advanced with the scorer's padded logits.  Sampling, alias-class mass,
STOP/cap handling, censoring, and RNG ownership remain in ``BatchClassPaths``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from farm_neat.policy.v0_scorer import V0Scorer

if TYPE_CHECKING:
    from farm_neat.training.teachers.v0_regret_proposals import BatchClassPaths

__all__ = ["make_tensor_decoder"]


def make_tensor_decoder(scorer: V0Scorer) -> Callable[["BatchClassPaths"], None]:
    """Return the production whole-session tensor decoder for ``scorer``.

    The supplied scorer remains on its caller-selected device.  The adapter
    keeps it in evaluation mode and uses inference mode for the complete
    callback.  Exceptions are intentionally not swallowed: the owning proposal
    boundary charges and censors a failed batch without rerolling it.
    """
    scorer.eval()

    def decode(session: "BatchClassPaths") -> None:
        scorer.eval()
        with torch.inference_mode():
            while (frame := session.frame()) is not None:
                scored = scorer.score_batch(**frame.score_inputs())
                session.advance(scored.logits)
                del scored, frame

    return decode
