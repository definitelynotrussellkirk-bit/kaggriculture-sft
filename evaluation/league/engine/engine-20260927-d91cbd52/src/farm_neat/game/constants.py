"""Engine constants, mirrored locally.

Source of truth: kaggle_environments/envs/kaggriculture/kaggriculture.py (v0.1.0), copied to
docs/upstream/kaggriculture_engine_reference.py. tests/test_engine_rules.py asserts that these
values still match the installed engine -- if that test fails, the engine changed, so update
this file and CLAUDE.md together.

The submission bundle must not import kaggle_environments (it is not guaranteed on the
competition runner), hence the duplication.
"""

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

# Only these two can be bought back from the market via BUY_PRODUCT.
BUYABLE_PRODUCTS = ("WHEAT", "FERTILIZER")

MARKET_I0 = 10000
PRICE_FLOOR = 1

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]
MAX_SHOP_INSTANCES = 8

# (dx, dy) -- y grows DOWNWARD, so NORTH decreases y.
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}

LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

FARM_HAND_COST_MULT = 1

# Defaults from kaggriculture.json.
DEFAULTS = {
    "episodeSteps": 720,
    "actTimeout": 1,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
}

BOARD_SIZE = DEFAULTS["boardSize"]
TURNS_PER_DAY = DEFAULTS["turnsPerDay"]
SHED_CAPACITY = DEFAULTS["shedCapacity"]
MAX_MARKET_ORDERS = DEFAULTS["maxMarketOrdersPerTurn"]
TOTAL_DAYS = DEFAULTS["episodeSteps"] // DEFAULTS["turnsPerDay"]


def fib(n):
    """Hire-cost fib, indexed so fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_cost(hires_today, mult=FARM_HAND_COST_MULT):
    return mult * fib(hires_today)


def quadrant_of(x, y, board_size=BOARD_SIZE):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def shed_access_tiles(board_size=BOARD_SIZE):
    """The four inner-corner tiles ringing the shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def is_shed_adjacent(pos, board_size=BOARD_SIZE):
    return (pos[0], pos[1]) in set(shed_access_tiles(board_size))


def water_bonus_window(crop):
    """Inclusive (start_day, end_day) age range where watering adds yield. One-time crops only."""
    cd = CROPS[crop]
    if cd["ongoing"]:
        return None
    return ((cd["max_yield_day"] + 1) // 2, cd["max_yield_day"])


def one_time_units_without_fertilizer(crop):
    """Maximum held units from ordinary bonus-window watering (one-time crops only)."""
    cd = CROPS[crop]
    window = water_bonus_window(crop)
    if window is None:
        raise ValueError(f"{crop} is ongoing")
    bonus_waters = window[1] - window[0] + 1
    return min(cd["max_yield"], 1 + bonus_waters)


def one_time_cycle_harvest_age(crop):
    """Earliest useful standard harvest age under ordinary, unfertilized watering.

    ``max_yield_day`` is the end of the engine's bonus window, not necessarily the earliest
    age at which held yield is capped. Melon reaches six units at age 10 even though its table
    says max_yield_day=12.
    """
    cd = CROPS[crop]
    window = water_bonus_window(crop)
    if window is None:
        raise ValueError(f"{crop} is ongoing")
    useful_bonus_waters = one_time_units_without_fertilizer(crop) - 1
    last_useful_bonus_age = window[0] + useful_bonus_waters - 1
    return max(cd["first_yield_day"], last_useful_bonus_age)
