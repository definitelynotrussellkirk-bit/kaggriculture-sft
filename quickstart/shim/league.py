"""PASS-only stand-in for the opponent league that play.py imports (the full league is not part of this repo)."""

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class League:
    def new_actor(self, name, seat, sampler_seed):
        if name != "PASS":
            raise ValueError(f"quickstart league only has PASS, not {name!r}")
        return lambda observation, configuration: {"farmer": ["PASS"], "hands": [], "market": []}
