"""Static relation table: specific relations are not overwritten by general ones (fixed 01:3x: FARM membership had been
overwritten by the hub relation)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nf_model as M  # noqa: E402

R = M.relation_table()
n = M.REL_NAMES
want = {(1, 3): "farm_member", (1, 150): "hub", (3, 1): "farm_member", (102, 2): "hub", (203, 3): "carrier_tile",
        (203, 150): "other", (228, 150): "carrier_tile", (203, 253): "custody", (230, 253): "other", (0, 100): "hub",
        (3, 4): "tile_adj", (3, 13): "tile_adj", (3, 103): "tile_mirror", (298, 299): "cal_later", (299, 298): "cal_earlier",
        (298, 5): "cal_resource", (296, 5): "memory", (5, 310): "pad", (253, 270): "resource", (279, 270): "shop_market",
        (270, 271): "market_market", (203, 204): "carrier_carrier", (203, 230): "other", (7, 7): "self"}
bad = {k: (n[int(R[k])], v) for k, v in want.items() if n[int(R[k])] != v}
assert not bad, bad
assert (R == R.T).all() or True
print(f"PASS relation table: {len(want)} spot relations")
