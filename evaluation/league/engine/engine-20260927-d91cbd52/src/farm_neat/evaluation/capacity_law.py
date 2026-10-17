"""Conditional quantity cuts; not a scheduler, economic policy, or value model.

Extend K-UNIT-ECONOMICS-ONE-TILE, Q2-ROUTE-SLACK-PREFIX-689e1fe6 and
K-CASH-FLOW-PREFIX-834E2F40 without changing their bound source bytes.

For q identical complete batches with a declared biological upper output u, each necessary
resource/prefix inequality fixed + q*per_batch <= capacity gives an integer cut.
Intersect those cuts and the exact affine cash-prefix interval. Infeasible is
proved only for those stated conditions. A surviving bound remains UNKNOWN as a
native schedule: pooled inventories, temporal aliasing, interacting products,
shared storage and market fill still need their own certificates.

Partial service can still yield base output: complete fertilized batches do NOT
bound total crop output. Split base output from optional bonus output using
refresh_output_bound below. Every capacity is for the SAME stated day/phase window. A shed cut is stock at a
specific cut, not units/day. A market cut counts orders, not product units. Only
supply quantities may be forecast here; no future shop or rival private input.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from farm_neat.economy.heuristic.state import manhattan
from farm_neat.evaluation.service_timeline import cash_flow_segment
from farm_neat.evaluation.unit_economics import formula, engine
from farm_neat.game.constants import PRODUCTS


def _integer(value: int, name: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f'{name}: integer' + (f' >= {minimum}' if minimum is not None else '') + ' required')
    return value


@dataclass(frozen=True)
class Cut:
    """Necessary resource inequality; coefficients are counted physical units.

    per_batch=0 describes a fixed queue order, setup cost, or fixed other work.
    For inputs, capacity must upper-bound the amount that can reach the selected
    service workers by the cutoff (pooled carried stock is only a loose bound).
    """
    name: str
    capacity: int
    per_batch: int
    fixed: int = 0
    unit: str = 'count'

    def __post_init__(self):
        if not self.name or not self.unit:
            raise ValueError('named cut with units required')
        for name in ('capacity','per_batch','fixed'):
            _integer(getattr(self,name),name,0)


@dataclass(frozen=True)
class CashEvent:
    """One ordered bank delta in cents: fixed_cents + q*per_batch_cents.

    Amounts must be exact fixed-commit effects or caller-certified optimistic
    amounts for an upper bound. Prices/fills are NOT inferred. Every shared-bank
    withdrawal belongs here; omitting them weakens, never strengthens, a cut.
    """
    fixed_cents: int
    per_batch_cents: int

    def __post_init__(self):
        _integer(self.fixed_cents,'fixed_cents')
        _integer(self.per_batch_cents,'per_batch_cents')


def cash_quantity_interval(start_cents: int, events: Sequence[CashEvent],
                           requested: int, protected_cents: int = 0) -> dict:
    """Exact integer feasible interval for the supplied affine cash event stream.

    Interleaved groups must already be merged in native commit order. Unlike a
    scalar total-cost cap, this retains every prefix, receipts and a possible
    lower quantity bound. It is an algebra identity, NOT a financing forecast.
    """
    for n,v in [('start_cents',start_cents),('requested',requested),('protected_cents',protected_cents)]:
        _integer(v,n,0)
    lo, hi, f, d = 0, requested, start_cents-protected_cents, 0
    lower_witness, upper_witness = None, None
    if f < 0:
        return {'status':'INFEASIBLE','lower_batches':0,'upper_batches':-1,'binding_prefix':-1}
    for i,e in enumerate(events):
        if not isinstance(e,CashEvent):raise ValueError('CashEvent sequence required')
        f += e.fixed_cents; d += e.per_batch_cents
        if d < 0:
            cut = f // -d
            if cut < hi:hi,upper_witness=cut,i
        elif d > 0:
            cut = -(f // d)  # ceil(-f/d), exact for negative and positive f
            if cut > lo:lo,lower_witness=cut,i
        elif f < 0:
            return {'status':'INFEASIBLE','lower_batches':lo,'upper_batches':-1,'binding_prefix':i}
    status='INFEASIBLE' if lo>hi else 'CONDITIONAL_INTERVAL'
    out={'status':status,'lower_batches':lo,'upper_batches':hi,
         'lower_binding_prefix':lower_witness,'upper_binding_prefix':upper_witness}
    if status!='INFEASIBLE':
        # Reuse the admitted cash summary as an independent boundary assertion.
        for q in {lo,hi}:
            _,required=cash_flow_segment([e.fixed_cents+q*e.per_batch_cents for e in events])
            assert start_cents >= protected_cents+required
    return out


def quantity_bound(product: str, requested_batches: int, units_per_batch: int,
                   cuts: Sequence[Cut], *, start_cents: int | None = None,
                   cash_events: Sequence[CashEvent] = (), protected_cents: int = 0) -> dict:
    """Upper output units FROM COMPLETE identical batches in a declared window.

    The units_per_batch is a biological/held-stock UPPER bound. Resource
    per_batch coefficients must be necessary LOWER costs for earning that batch.
    Partial/unmodeled batches are excluded, not assumed worthless.
    Different products share the SAME capacities: never add independent maxima.
    fixed travel may be used only when it is required by the chosen itinerary.
    """
    if product not in PRODUCTS:raise ValueError('known engine product required')
    _integer(requested_batches,'requested_batches',0);_integer(units_per_batch,'units_per_batch',0)
    if cash_events and start_cents is None:raise ValueError('cash stream requires starting cash')
    if any(not isinstance(c,Cut) for c in cuts):raise ValueError('Cut sequence required')
    if any(c.name in {'requested_batches','cash_prefix'} for c in cuts):
        raise ValueError('reserved resource cut name')
    if len({c.name for c in cuts})!=len(cuts):raise ValueError('distinct resource cut names required')
    caps={'requested_batches':requested_batches}; lo=0
    for c in cuts:
        remaining=c.capacity-c.fixed
        if remaining<0:
            caps[c.name]=-1
        elif c.per_batch:
            caps[c.name]=remaining//c.per_batch
    cash=None
    if start_cents is not None:
        cash=cash_quantity_interval(start_cents,cash_events,requested_batches,protected_cents)
        caps['cash_prefix']=cash['upper_batches'];lo=cash['lower_batches']
    hi=min(caps.values())
    impossible=hi<lo or (cash is not None and cash['status']=='INFEASIBLE')
    return {'product':product,'status':'INFEASIBLE_UNDER_CONDITIONS' if impossible else 'UPPER_BOUND_ONLY',
            'lower_batches':lo,'upper_batches':max(0,hi) if not impossible else 0,
            'upper_units':0 if impossible else units_per_batch*hi,
            'binding':[k for k,v in caps.items() if v==hi], 'cuts':caps,'cash':cash,
            'native_feasibility':'UNKNOWN',
            'boundary':'Same window and batch definition; biological upper, necessary lower costs, no value or WDL claim.'}


def day_unit_turns(day: int, hire_hours: Sequence[int] = ()) -> int:
    """Exact available unit-turns when each hand is bought at its listed hour.

    Eight hands plus farmer: nominal 216, at most 208 on a full day (hands cannot
    act on their hiring turn). Native final day 29 has only hours 0..22: 199.
    Does not assert those hires are funded or fit the ten-order market queue.
    """
    _integer(day,'day',0)
    if day>29:raise ValueError('native day 0..29 required')
    hours=23 if day==29 else 24
    for h in hire_hours:
        _integer(h,'hire hour',0)
        if h>=hours:raise ValueError('hire outside actionable day')
    return hours+sum(hours-h-1 for h in hire_hours)


def fixed_route_moves(routes: Sequence[Sequence[tuple[int,int]]]) -> int:
    """Required walking for FIXED worker/stop order, ignoring obstacles (none).

    Each route starts at that worker's actual origin. Do not sum shed-to-site
    distances as independent round trips; shared paths are not separate trips.
    This omits actions, waits, pickups and suffix deadlines from the parent.
    Free assignment/unknown itineraries must use a safe weaker bound, e.g. zero.
    """
    for r in routes:
        for p in r:
            if len(p)!=2 or any(type(v) is not int or not 0<=v<10 for v in p):
                raise ValueError('native 10x10 integer coordinate required')
    return sum(manhattan(a,b) for r in routes for a,b in zip(r,r[1:]))


def ongoing_cohort_upper(crop: str, planted_day: int, *, last_day: int = 29,
                        fertilizer_bonus: bool = False) -> dict:
    """Per-day fresh biological output upper, conditional on the given birth.

    Ignore neglect, early removal, held-cap blocking and harvest/delivery; retain
    finite generation and phase. No final refresh into day 30 is available.
    Fertilizer bonus=True is an optimistic input/service assumption, not a fact.
    """
    E=engine()
    if crop not in E.CROPS or not E.CROPS[crop]['ongoing']:raise ValueError('ongoing crop required')
    _integer(planted_day,'planted_day',0);_integer(last_day,'last_day',0)
    if last_day>29 or type(fertilizer_bonus) is not bool:raise ValueError('native horizon/bool policy required')
    c=E.CROPS[crop]
    events={d:1+int(fertilizer_bonus) for d in
            (planted_day+c['first_yield_day']+i*c['interval'] for i in range(c['max_yield'])) if d<=last_day}
    return {'crop':crop,'by_day':events,'units':sum(events.values()),'stage':'fresh biological units, not sales'}


def declared_site_ceiling(producer: str, first_day: int, *, fertilizer: bool = False) -> int:
    """Shift the parent's DECLARED one-tile policy to its earliest access day.

    Not a global optimum over service policies. Land/authorization start can only
    delay planting. An inaccessible site can pass first_day=30 for zero output.
    """
    _integer(first_day,'first_day',0)
    if first_day>=30:return 0
    if producer not in engine().CROPS:raise ValueError('crop policy ceiling only')
    if type(fertilizer) is not bool:raise ValueError('bool fertilizer policy required')
    policy='watered_fertilized' if fertilizer else 'watered'
    return int(formula(producer,policy,30-first_day)['units'])


def refresh_output_bound(crop: str, event_count: int, already_bonus_eligible: int,
                         additional_bonus_candidates: int, bonus_cuts: Sequence[Cut]) -> dict:
    """Fresh ongoing-crop units = base events + at most one bonus per event.

    All event counts are upper bounds at ONE named next-day refresh. Already
    eligible means both watered and fertilized at that refresh's preceding day.
    Additional candidates need specified remaining resources. Partition mixed
    prerequisite classes separately or use safe weak lower costs; do not charge
    already-fertilized plants a new input. Held-cap clipping and neglect can only
    lower this output. Does not imply a HARVEST, deposit, fill, or terminal cash.
    """
    if crop not in engine().CROPS or not engine().CROPS[crop]['ongoing']:
        raise ValueError('ongoing crop required')
    for n,v in [('event_count',event_count),('already_bonus_eligible',already_bonus_eligible),
                ('additional_bonus_candidates',additional_bonus_candidates)]:_integer(v,n,0)
    if already_bonus_eligible+additional_bonus_candidates>event_count:
        raise ValueError('bonus classes must be disjoint subsets of events')
    new=quantity_bound(crop,additional_bonus_candidates,1,bonus_cuts)
    bonus=already_bonus_eligible+new['upper_units']
    return {'product':crop,'base_units_upper':event_count,'bonus_units_upper':bonus,
            'fresh_units_upper':event_count+bonus,'additional_bonus_bound':new,
            'stage':'fresh biological output at one declared refresh, not harvested or sold units'}


def reduce_native_receipt(path, expected_sha256: str) -> dict:
    """Offline reduction of the diagnostic probe's hash-pinned receipt.

    Input counts are evidence supplied by the native probe, NOT actor features.
    This command never initializes an environment or runs a policy. Its hash
    binds the exact input; the probe's separate PLAN/check binds source/control.
    """
    import hashlib
    import json
    from pathlib import Path
    raw=Path(path).read_bytes()
    actual=hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError('native receipt SHA-256 mismatch')
    data=json.loads(raw)
    if len(data['cases']) != data['counts']['games']:
        raise ValueError('case count mismatch')
    out=[]
    for case in data['cases']:
        if len(case['rows'])!=2 or len(case['cash'])!=2:
            raise ValueError('two-seat diagnostic required')
        seats=[]
        for i,row in enumerate(case['rows']):
            generated=row['strawberry_refresh_units']
            raw_events=row['strawberry_raw_opportunities']
            bonus=row['strawberry_bonus_opportunities']
            if generated != raw_events+bonus-row['strawberry_cap_loss']:
                raise ValueError('fresh output accounting mismatch')
            if case['cash'][i] != 3000+row['receipts_total']-row['spend_total']:
                raise ValueError('cash accounting mismatch')
            harvested=row['harvested'].get('STRAWBERRY',0)
            if harvested>generated:
                raise ValueError('harvest exceeds generated stock in empty-stock diagnostic')
            sold=row['sold'].get('STRAWBERRY',0)
            if sold>harvested:
                raise ValueError('sales exceed harvested stock in no-strawberry-buy diagnostic')
            one={'plant_successes':row['success'].get('PLANT:STRAWBERRY',0),
                 'strawberry_water_successes':row['success'].get('WATER:STRAWBERRY',0),
                 'all_crop_water_attempts':row['actions'].get('WATER',0),
                 'base_events':raw_events,'bonus_units':bonus,
                 'held_cap_loss':row['strawberry_cap_loss'],
                 'fresh_units':generated,'harvested_units':harvested,'sold_units':sold,
                 'generation_to_harvest_gap':generated-harvested,
                 'hire_spend':row['spend'].get('HIRE',0),
                 'available_unit_turns':row['available_unit_turns'],
                 'peak_hands':row['max_hands'],'terminal_cash':case['cash'][i]}
            if row['cohorts']:
                one['given_births_unfertilized_biological_upper']=sum(
                    ongoing_cohort_upper('STRAWBERRY',c['planted_day'])['units'] for c in row['cohorts'])
                one['birth_days']=[c['planted_day'] for c in row['cohorts']]
                one['birth_bound_condition']='One unchanged unfertilized cohort per listed birth; excludes replanning; not a total-output bound when fertilized.'
            seats.append(one)
        out.append({'name':case['name'],'scenario':case['scenario']['id'],
                    'opponent':case['scenario']['opponent'],'wdl':case['wdl'],
                    'cash_margin':case['margin'],'seats':seats,
                    'day0':case['day0']})
    return {'schema':'farm-capacity-reduction-v1','input_sha256':actual,
            'plan_sha256':data['plan_sha256'],'counts':data['counts'],'cases':out,
            'boundary':'Two exposed diagnostics, not policy-strength comparison or held-out validation. Full PLAN/source verification is separate.'}


def main(argv=None) -> int:
    """Explicit offline entry point; never launch a game from tooling/imports."""
    import argparse
    import json
    import sys
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    reduce=commands.add_parser('reduce',help='Hash-check and compress an existing native probe receipt; zero games')
    reduce.add_argument('--input',required=True,type=Path)
    reduce.add_argument('--sha256',required=True)
    reduce.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    try:
        result=reduce_native_receipt(args.input,args.sha256)
        text=json.dumps(result,indent=2,sort_keys=True)+'\n'
        if args.out:
            # Reused paths are evidence, not scratch. Caller selects a NEW path.
            with args.out.open('x') as f:f.write(text)
        else:sys.stdout.write(text)
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,f'capacity-law: {error}\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
