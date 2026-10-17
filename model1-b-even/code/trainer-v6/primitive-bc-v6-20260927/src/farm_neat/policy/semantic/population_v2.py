"""Small NW population curriculum; reuses semantic V1 mechanics and neural code.

ENSURE_COWS is a monotone minimum managed-population requirement, never an
implicit disposal/replacement command. No profitability or economic-plan search.
This vocabulary is NEW despite retaining29 candidate slots and64 features.
"""
from __future__ import annotations

import copy
from pathlib import Path
import types

import numpy as np

from farm_neat.training.schedule_model import bound, file_sha
from farm_neat.policy.semantic import policy_v1 as original
from farm_neat.economy import orders_v1 as mechanics

SCHEMA = "farm-semantic-population-actor-v2"
ORDER_SCHEMA = "farm-semantic-population-orders-v2"
ROOT = Path(__file__).resolve().parents[4]
NW_SITES = tuple(sorted(((x,y) for y in range(5) for x in range(5)),
                       key=lambda p:(8-p[0]-p[1],p[1],p[0])))
ACTIONS = copy.deepcopy(original.ACTIONS)
for index,target in zip((3,6,9,12),(1,4,8,12),strict=True):
    ACTIONS[index] = {"kind":"ENSURE_COWS","target":target,"until":"GAME_END"}
ACTION_IDS = tuple(range(29))
WIDTH,SHAPES,TRAINABLE = original.WIDTH,original.SHAPES,original.TRAINABLE
ACTION_SCHEMA = bound({"schema":"farm-semantic-population-vocabulary-v2","actions":ACTIONS,
    "scope":"NW_MONOTONE_COW_POPULATION; FOUR_WHEAT_SITES; ORIGINAL_SALE_LABOR_CONTROLS",
    "not_v1_action_meanings_or_checkpoints":True,
    "population_columns_55_to62":["requested/12","committed/12","placed/12","allocatable/25",
        "unallocated_shortfall/12","awaiting_placement/12","candidate_increment/12","candidate_total/12"]})


class PopulationOrders(mechanics.SemanticOrders):
    def __init__(self,*,hands=2):
        super().__init__(hands=hands)
        self.requested_cows = 0
        self.population_report = {}

    def ensure_cows(self,total):
        mechanics._integer(total,0,12,"minimum managed cow population")
        self.requested_cows=max(self.requested_cows,total)

    def keep_animal(self,tile,animal="COW",*,until="GAME_END"):
        site=mechanics._site(tile)
        if site not in NW_SITES:
            raise ValueError("population curriculum only allocates NW sites")
        if f"{site[0]},{site[1]}" not in self.orders and self.committed_cows()>=12:
            raise ValueError("population curriculum supports at most12 committed cows")
        super().keep_animal(site,animal,until=until)
        self.requested_cows=max(self.requested_cows,self.committed_cows())

    def committed_cows(self):
        return sum(o["kind"]=="ANIMAL" and o["item"]=="COW" for o in self.orders.values())

    def capacity(self,observation):
        tiles=observation["farms"][observation["player"]]["tiles"]
        eligible=[]
        for x,y in NW_SITES:
            if f"{x},{y}" in self.orders:
                continue
            tile=tiles[y][x]
            if tile is None or isinstance(tile,dict) and (tile.get("kind")=="WEED" or
                    tile.get("kind")=="PASTURE" and tile.get("animal") is None):
                eligible.append((x,y))
        placed=sum(isinstance(tiles[o["tile"][1]][o["tile"][0]],dict) and
            tiles[o["tile"][1]][o["tile"][0]].get("animal")=="COW"
            for o in self.orders.values() if o["kind"]=="ANIMAL" and o["item"]=="COW")
        committed=self.committed_cows()
        return {"requested":self.requested_cows,"committed":committed,"placed":placed,
            "allocatable":len(eligible),"eligible_sites":eligible,
            "unallocated_shortfall":max(0,self.requested_cows-committed),
            "awaiting_placement":committed-placed}

    def _allocate(self,observation):
        status=self.capacity(observation)
        for tile in status["eligible_sites"][:status["unallocated_shortfall"]]:
            self.keep_animal(tile)
        status=self.capacity(observation)
        self.population_report={k:v for k,v in status.items() if k!="eligible_sites"}
        self.population_report["allocation_shortfall_cause"]=(
            "NO_UNCOMMITTED_COMPATIBLE_UNLOCKED_NW_SITE" if status["unallocated_shortfall"] else None)

    def act(self,observation,configuration=None):
        self._allocate(observation)
        result=super().act(observation,configuration)
        self.last_report["population"]=copy.deepcopy(self.population_report)
        return result

    __call__=act

    def snapshot(self):
        state=super().snapshot()
        state.update(schema=ORDER_SCHEMA,requested_cows=self.requested_cows,
                     population_report=copy.deepcopy(self.population_report))
        return state

    @classmethod
    def restore(cls,state):
        if state.get("schema")!=ORDER_SCHEMA:
            raise ValueError("versioned population snapshot required")
        old=copy.deepcopy(state); old["schema"]=mechanics.SCHEMA
        requested=old.pop("requested_cows"); report=old.pop("population_report")
        result=mechanics.SemanticOrders.restore.__func__(cls,old)
        mechanics._integer(requested,0,12,"saved requested cow population")
        if requested<result.committed_cows():
            raise ValueError("saved target below irreversible commitments")
        result.requested_cows=requested; result.population_report=report
        return result


def sources():
    return {**original.sources(),"src/farm_neat/policy/semantic/population_v2.py":file_sha(__file__)}


def initial_orders():
    orders=PopulationOrders(hands=2)
    for item in ("WHEAT","MILK","FERTILIZER"):
        orders.sell_when(item,price_above=0)
    return orders


_pack_actions=copy.deepcopy(ACTIONS)
for command in _pack_actions:
    if command["kind"]=="ENSURE_COWS":
        command["kind"]="COW"  # existing family one-hot; new target/state fields below
_pack=types.FunctionType(original.pack.__code__,{**original.__dict__,"ACTIONS":_pack_actions})


def pack(observation,orders):
    if not isinstance(orders,PopulationOrders):
        raise ValueError("population-v2 order state required")
    rows,mask=_pack(observation,orders)
    status=orders.capacity(observation)
    rows[:,6] /= 3  # own observable cows now scaled by12, not the old four-site cap
    rows[:,55:61]=np.asarray([status["requested"]/12,status["committed"]/12,
        status["placed"]/12,status["allocatable"]/25,status["unallocated_shortfall"]/12,
        status["awaiting_placement"]/12],np.float32)
    for i,command in enumerate(ACTIONS):
        target=command["target"] if command["kind"]=="ENSURE_COWS" else status["requested"]
        rows[i,61]=max(0,target-status["committed"])/12
        rows[i,62]=target/12
        if command["kind"]=="ENSURE_COWS":
            rows[i,41]=target/12
            mask[i]=target>status["requested"]
    return rows,mask


def apply_command(orders,index):
    command=ACTIONS[index]
    if command["kind"]=="ENSURE_COWS":
        orders.ensure_cows(command["target"])
    else:
        original.apply_command(orders,index)


# Rebind existing implementations in one private namespace. Original source,
# action schema, checkpoint parser and current campaign remain untouched.
Policy,torch_graph,validate_inputs=original.Policy,original.torch_graph,original.validate_inputs
_scope={**original.__dict__,"SCHEMA":SCHEMA,"ACTION_SCHEMA":ACTION_SCHEMA,
    "ACTIONS":ACTIONS,"ACTION_IDS":ACTION_IDS,"sources":sources,
    "SemanticOrders":PopulationOrders,"initial_orders":initial_orders,"pack":pack,
    "apply_command":apply_command,"Policy":Policy}
for _name in ("save_model","load_model","initialize"):
    _function=getattr(original,_name)
    globals()[_name]=types.FunctionType(_function.__code__,_scope,argdefs=_function.__defaults__)
    _scope[_name]=globals()[_name]


def fit(*args,**kwargs):
    """Future authorized numerical fitting seam; no fitting is run by this module."""
    function=types.FunctionType(original.fit.__code__,_scope,argdefs=original.fit.__defaults__)
    function.__kwdefaults__=original.fit.__kwdefaults__
    return bound({**function(*args,**kwargs),"schema":"farm-semantic-population-ppo-fit-v2"})
