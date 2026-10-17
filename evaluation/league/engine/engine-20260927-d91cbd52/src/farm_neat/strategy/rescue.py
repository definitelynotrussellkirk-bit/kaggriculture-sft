"""Narrow intraday production FEED proposals; never a notebook-policy transplant.

LateFeedRescue implements the existing StrategyModule protocol with name
``production``. The production pipeline may append these proposals to its dawn
planner's output; this is not a second owner, executor, or selection policy.

Inputs: the actor's current observation, memory['baseline_action'] containing
ALL final parent worker commands/market orders, and an explicit same-step
``derived['rescue_reservations'] = {'step': step, 'touches': [...]}``. The latter
is the coordinator/capacity producer's complete set of already claimed resources;
an explicit empty list means no other claims, missing means UNKNOWN. Recompute
it after selection. Broad labor/land/inventory claims conflict conservatively.

Scope: default 10x10 / 24-hour / 30-day game, hours 22 or 23 before day 29,
<=25 observed workers. Only an explicitly PASSing worker already ON an animal,
carrying one wheat, is eligible. The animal must have exactly one unfed dawn,
not be fed today, and have a known earlier placed_day. Any co-located committed
parent worker blocks that target, including a protected FEED. No travel, pickup,
purchase, cash buffer, cohort cancellation, or imagined idle hand is supplied.
The bounded scan is O(100 * workers); at most one alternative per animal.
Proposals compete through the existing touches contract, not as an executable
batch. Requirements are marginal under these guards, not portfolio obligations.

Outputs: ordinary production Proposals; explain() also returns refusal reasons.
Requirements are exact LOCAL consumption; terminal own/rival/margin deltas and
feasibility remain UNKNOWN. Another module owns future funding and value.
Adoption must revalidate ALL guards and perform adoption.Session.act's complete
joint-endpoint preflight, including preserved base workers and dawn processing.
The existing ProductionAdapter supports dawn ADD/RETIRE, not this new intraday
schema; this module does not route around it or grant an execution certificate.

confirm_rescue checks a bounded, contiguous OWN-observation trace and the recorded
final emitted action through the next dawn. It can report observed survival,
never causal cash gain. Identity is (seat, coordinate, kind, animal, placed_day),
NOT a universal animal UUID; requiring an earlier placed_day and observing every
intervening step excludes same-day native replacements in this narrow window.
It neither stores pending state nor increments a counter on an absent tile.
Excluded inputs: evaluator seed, rival private state, future shops/flow, teacher
labels. Excluded outputs: route, native-adoption readiness, feasibility proof,
paired payoff, and policy-strength claims.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from farm_neat.game.constants import ANIMALS
from farm_neat.strategy.coordinator import resource_conflicts
from farm_neat.strategy.types import Estimate, Proposal, Requirements, StrategyContext

SCHEMA = 'farm-production-feed-rescue-v1'
PARENTS = (
    '6a083b9ab892e34fda27ec1e9a87861de856d7fd0d58dd47c150780ff7c01987',
    '396c84c7c1267ab475e43718a75f45894e58ada255893a927d477f046dd5f6d7',
)
GUARDS = (
    'source own observation, final baseline action and reservations unchanged',
    'same observed animal identity, worker position and carried wheat',
    'worker explicitly PASS; no co-located committed parent worker',
    'resource claims disjoint and revalidated at execution',
    'joint-endpoint preflight over ALL emitted workers and dawn refresh passes',
    'pinned default engine configuration',
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _own(observation: Mapping) -> dict:
    """Only own state participates in this local preimage, never a future world."""
    seat = observation['player']
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('invalid player')
    return {'step': observation['step'], 'player': seat,
            'farm': observation['farms'][seat], 'private': observation['private']}


def _position(value: Any) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(n) is int and 0 <= n < 10 for n in value))


def _identity(tile: Any) -> dict | None:
    if not isinstance(tile, Mapping):
        return None
    animal = tile.get('animal')
    if not isinstance(animal, str) or animal not in ANIMALS:
        return None
    if tile.get('kind') != ANIMALS[animal]['structure']:
        return None
    born = tile.get('placed_day')
    if type(born) is not int or born < 0:
        return None
    return {'kind': tile['kind'], 'animal': animal, 'placed_day': born}


class LateFeedRescue:
    """A stateless production producer. The caller owns reservation completeness."""
    name = 'production'

    def propose(self, context: StrategyContext) -> tuple[Proposal, ...]:
        return self._scan(context)[0]

    def explain(self, context: StrategyContext) -> dict:
        proposals, refusals = self._scan(context)
        return {'schema': SCHEMA, 'status': 'PROPOSED' if proposals else 'REFUSED',
                'proposals': [p.to_dict() for p in proposals], 'refusals': refusals,
                'feasibility': 'UNKNOWN', 'adoption_executed': False,
                'paired_effect': 'UNKNOWN'}

    def _scan(self, context: StrategyContext) -> tuple[tuple[Proposal, ...], list[dict]]:
        if not isinstance(context, StrategyContext):
            raise TypeError('one StrategyContext required')
        refusals: list[dict] = []

        def refuse(reason: str, **details: Any):
            refusals.append({'reason': reason, **details})
            return (), refusals

        obs, now = context.observation, context.step
        if (obs.get('step') != now or type(obs.get('step')) is not int
                or obs.get('player') != context.player or type(obs.get('player')) is not int):
            return refuse('OBSERVATION_CONTEXT_MISMATCH')
        day, hour = divmod(now, 24)
        if day >= 29:
            return refuse('LAST_DAY_NO_SURVIVAL_HORIZON')
        if hour not in (22, 23):
            return refuse('OUTSIDE_LATE_WINDOW')
        if (('day' in obs and obs['day'] != day) or ('hour' in obs and obs['hour'] != hour)):
            return refuse('OBSERVATION_CLOCK_MISMATCH')
        reservations = context.derived.get('rescue_reservations')
        if not isinstance(reservations, Mapping) or set(reservations) != {'step', 'touches'}:
            return refuse('RESERVATIONS_UNKNOWN')
        if type(reservations['step']) is not int or reservations['step'] != now:
            return refuse('STALE_RESERVATIONS')
        claimed = reservations['touches']
        if (not isinstance(claimed, (list, tuple))
                or any(not isinstance(t, str) or not t.strip() for t in claimed)):
            return refuse('INVALID_RESERVATIONS')
        claimed = sorted(set(claimed))
        parent = context.memory.get('baseline_action')
        try:
            own = _own(obs)
            farm, private = own['farm'], own['private']
            positions = [farm['farmer'], *farm['hands']]
            board = farm['tiles']
            if (not 1 <= len(positions) <= 25 or not all(_position(p) for p in positions)
                    or len(board) != 10 or any(len(row) != 10 for row in board)):
                return refuse('OUTSIDE_DEFAULT_GEOMETRY_OR_WORKER_BOUND')
            inventories = private['inventories']
            if (not isinstance(inventories, list) or len(inventories) != len(positions)
                    or any(not isinstance(inv, Mapping) for inv in inventories)):
                return refuse('MISSING_WORKER_CUSTODY')
            if (not isinstance(parent, Mapping) or set(parent) != {'farmer', 'hands', 'market'}
                    or not isinstance(parent['hands'], list)
                    or len(parent['hands']) != len(positions) - 1
                    or not isinstance(parent['market'], list)):
                return refuse('FINAL_PARENT_ACTION_REQUIRED')
            commands = [parent['farmer'], *parent['hands']]
            if any(not isinstance(c, list) or not c or not isinstance(c[0], str) for c in commands):
                return refuse('MALFORMED_PARENT_COMMAND')
            own_hash = _digest(own)
            # Bind the complete final parent word, including untouched actions.
            binding = _digest({'own_sha256': own_hash, 'parent': dict(parent), 'claims': claimed})
        except (KeyError, TypeError, IndexError, ValueError):
            return refuse('MALFORMED_OWN_INPUT')

        proposals = []
        for y, row in enumerate(board):
            for x, tile in enumerate(row):
                identity = _identity(tile)
                if identity is None:
                    continue
                if (type(tile.get('consecutive_unfed')) is not int
                        or tile['consecutive_unfed'] != 1 or tile.get('fed_today') is not False
                        or identity['placed_day'] >= day):
                    continue
                target = [x, y]
                workers = [u for u, pos in enumerate(positions) if list(pos) == target]
                if not workers:
                    refuse('NO_WORKER_ON_TARGET', target=target)
                    continue
                committed = [u for u in workers if commands[u] != ['PASS']]
                if committed:
                    for unit in committed:
                        refuse('PARENT_FEED_PROTECTED' if commands[unit] == ['FEED']
                               else 'PARENT_WORKER_COMMITTED', actor=unit, target=target,
                               command=commands[unit])
                    continue
                for unit in workers:
                    wheat = inventories[unit].get('WHEAT', 0)
                    if type(wheat) is not int or wheat < 1:
                        refuse('NO_CARRIED_WHEAT', actor=unit, target=target)
                        continue
                    touches = ('labor', 'land', f'unit:{unit}', 'inventory:WHEAT',
                               f'animal:{x}:{y}', f'tile:{x}:{y}')
                    conflicts = resource_conflicts(touches, claimed)
                    if conflicts:
                        refuse('RESOURCE_RESERVED', actor=unit, target=target,
                               conflicts=list(conflicts))
                        continue
                    candidate = {'schema': SCHEMA, 'kind': 'FEED_RESCUE', 'step': now,
                        'player': context.player, 'actor': unit, 'target': target,
                        'identity': identity, 'unit_command': ['FEED'],
                        'baseline_command': ['PASS'], 'control_action': dict(parent),
                        'own_observation_sha256': own_hash, 'binding_sha256': binding,
                        'reserved_touches': claimed}
                    requirements = Requirements(
                        cash=Estimate.exact(0), reserve=Estimate.exact(0),
                        inventory={'WHEAT': Estimate.exact(1)}, market_slots=Estimate.exact(0),
                        worker_hours=Estimate.exact(1), land_quadrants=Estimate.exact(0),
                        window=(now, now))
                    proposals.append(Proposal(self.name, f'feed-rescue:{now}:{unit}:{x}:{y}',
                        candidate, touches, requirements, guards=GUARDS,
                        invalidated_by=('step or day advanced', 'worker moved or custody changed',
                            'animal replaced or fed', 'final parent action changed',
                            'module reservations changed'), parents=PARENTS))
                    break  # alternative selection, not a multi-worker feed batch
        if not proposals and not refusals:
            refuse('NO_OBSERVED_ENDANGERED_ANIMAL')
        return tuple(proposals), refusals


def confirm_rescue(proposal: Proposal, observations: Sequence[Mapping], *,
                   emitted_action: Mapping | None = None) -> dict:
    """Check observation continuity through dawn, not a counterfactual payoff.

    ``observations`` starts at the proposal's exact preimage and contains every
    own observation up to next dawn (at most three). ``emitted_action`` is the
    recorded FINAL joint own action at the first step, not an inner planned word.
    A caller must supply authentic observations/action logs; this function does
    not authenticate their transport or call the native interpreter. `confirmed`
    means SURVIVAL_OBSERVED in that bounded trace, never beneficial intervention.
    """
    def result(status: str, reason: str, *, confirmed=False):
        return {'status': status, 'reason': reason, 'confirmed': confirmed,
                'paired_benefit': 'UNKNOWN', 'scope': 'contiguous own-observation survival only'}

    try:
        c = proposal.candidate
        if proposal.module != 'production' or c['schema'] != SCHEMA or c['kind'] != 'FEED_RESCUE':
            return result('UNCONFIRMED', 'NOT_A_RESCUE_PROPOSAL')
        now, seat, unit = c['step'], c['player'], c['actor']
        if (type(now) is not int or not 0 <= now < 696 or now % 24 not in (22, 23)
                or type(unit) is not int or unit < 0 or not _position(c['target'])
                or c['unit_command'] != ['FEED'] or c['baseline_command'] != ['PASS']):
            return result('UNCONFIRMED', 'INVALID_RESCUE_PROPOSAL')
        expected = copy.deepcopy(c['control_action'])
        if unit == 0:
            expected['farmer'] = ['FEED']
        else:
            expected['hands'][unit - 1] = ['FEED']
        if emitted_action != expected:
            return result('UNCONFIRMED', 'EMITTED_ACTION_MISMATCH')
        if not 1 <= len(observations) <= 25 - now % 24:
            return result('UNCONFIRMED', 'INVALID_TRACE_LENGTH')
        if _digest(_own(observations[0])) != c['own_observation_sha256']:
            return result('UNCONFIRMED', 'SOURCE_PREIMAGE_MISMATCH')
        x, y = c['target']
        for index, obs in enumerate(observations):
            if (type(obs.get('step')) is not int or obs['step'] != now + index
                    or type(obs.get('player')) is not int or obs['player'] != seat):
                return result('UNCONFIRMED', 'NONCONTIGUOUS_OR_WRONG_SEAT')
            tile = obs['farms'][seat]['tiles'][y][x]
            if _identity(tile) != c['identity']:
                return result('UNCONFIRMED', 'MISSING_OR_REPLACED_ANIMAL')
            if not index:
                continue
            dawn = obs['step'] % 24 == 0
            if (tile.get('fed_today') is not (False if dawn else True)
                    or type(tile.get('consecutive_unfed')) is not int
                    or tile['consecutive_unfed'] != (0 if dawn else 1)):
                return result('UNCONFIRMED', 'FEED_OR_DAWN_EFFECT_NOT_OBSERVED')
            if index == 1 and not dawn:
                before = observations[0]['private']['inventories'][unit].get('WHEAT', 0)
                after = obs['private']['inventories'][unit].get('WHEAT', 0)
                if type(after) is not int or after != before - 1:
                    return result('UNCONFIRMED', 'FEED_CUSTODY_EFFECT_NOT_OBSERVED')
        if len(observations) < 25 - now % 24:
            return result('PENDING_DAWN', 'CONTIGUOUS_NEXT_DAWN_OBSERVATION_REQUIRED')
        return result('SURVIVAL_OBSERVED', 'SAME_ANIMAL_ALIVE_AFTER_FED_DAWN', confirmed=True)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return result('UNCONFIRMED', 'MALFORMED_TRACE_OR_PROPOSAL')



def _rescue_guard_evidence(context: StrategyContext, proposal: Proposal, phase: str) -> dict:
    """Re-derive permission and full-word feasibility independently for each phase.

    No owner is replaced: commitments validates the published plan/window and its
    actual receipts; the unchanged adopter lowers and checks the joint endpoint.
    Neither the native transport nor a previous True answer is an authority.
    """
    from farm_neat.strategy import adoption, commitments
    denied = {g: False for g in proposal.guards}
    denied[adoption.RESCUE_AUTHORIZATION] = False
    result = dict(phase=phase, answers=denied, status='NO_AUTHORIZATION')
    try:
        if phase not in ('ADMIT', 'EXECUTE'):
            raise ValueError('ADMIT or EXECUTE required')
        # Also validates the complete recursive actor-only serialization boundary.
        result.update(context_sha256=adoption.context_digest(context),
                      proposal_sha256=adoption.digest(proposal.to_dict()))
        window = context.memory.get('declared_service_window')
        check = commitments.validate_service_window(context, window)
        if check['status'] != 'READY':
            raise ValueError(check.get('reason', 'window is no longer ready'))
        answers = commitments.service_window_guard(context, proposal, phase)
        if (answers.get(adoption.RESCUE_AUTHORIZATION) is not True
                or any(answers.get(g) is not True for g in proposal.guards)):
            raise ValueError('owner does not authorize this exact rescue at this prefix')
        # Do not attest the proposal's joint-preflight guard from feasibility alone.
        # Compile/merge/check through the accepted implementation on this actual word.
        patches = adoption.ProductionAdapter().compile(context, proposal, adoption.NativeLowerer())
        if len(patches) != 1 or patches[0].step != context.step:
            raise ValueError('exactly one current-step rescue patch required')
        action = adoption._merge_patch(context.memory['baseline_action'], patches[0], context.observation)
        proof = adoption.joint_preflight(context.observation, action, patches[0])
        if action != check['turn']['action']:
            raise ValueError('lowered final word differs from the owner-authorized word')
        result.update(status='READY', answers=dict(answers), window_id=window['window_id'],
                      revision=window['authority']['revision'], joint_preflight=proof,
                      final_action_sha256=adoption.digest(action))
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        result['reason'] = str(error)
    except RuntimeError as error:
        result.update(status='SOURCE_OR_RUNTIME_DRIFT', reason=str(error))
    return result


def rescue_input_guard(context: StrategyContext, proposal: Proposal, phase: str) -> dict[str, bool]:
    """Drop-in Session guard. Revalidates owner, proposal and joint word on EVERY call."""
    return _rescue_guard_evidence(context, proposal, phase)['answers']


def produce_rescue_inputs(context: StrategyContext, *,
                          selected_others: Sequence[Proposal] | None = None) -> dict:
    """Transport ONE existing owner-authorized current rescue, never select economics.

    Inputs are the genuine actor context containing active_plan,
    declared_service_window, exact final baseline_action, and the owner's complete
    same-step rescue_reservations. ``selected_others`` MUST explicitly enumerate
    ALL other selected proposals after selection (empty tuple is an assertion by
    that caller, not a default). No missing claims become empty. We union their
    touches into the current reservations and revalidate the immutable owner
    declaration; a changed set requires the OWNER to reissue, even if disjoint.
    Claims are not inferred from idle workers, survival, or K0006 route intentions.

    READY returns the existing native-gate context/proposal/actor_permission wire
    formats plus independently rederived guard evidence. The action/permission is
    bound to this step, full observation, final word, window, and active revision.
    Adoption must call rescue_input_guard again at EXECUTE, not cache these answers.
    No snapshot, evaluator metadata, private rival state, policy call or native
    transition is consumed or produced. Ordinary contexts fail closed.

    Scope is the accepted owner's <=2-turn hour-22 window; not the low-level
    LateFeedRescue producer's broader hour-23 eligibility. Bounded by 128 selected
    proposals, <=25 workers, and the accepted owner/lowerer component projections.
    Completeness/authenticity is the same actor-owned trust root as commitments;
    this function cannot authenticate fabricated FullSeasonPlan memory.
    """
    from dataclasses import replace
    from farm_neat.strategy import adoption, commitments

    def refuse(status, reason, **details):
        return dict(schema='solver-rescue-inputs-v1', status=status, reason=reason,
                    inputs=None, native_refusal=False, opportunity_rate='UNKNOWN', **details)

    if not isinstance(context, StrategyContext):
        return refuse('NO_AUTHORIZATION', 'StrategyContext required')
    if not context.memory.get('active_plan') or not context.memory.get('declared_service_window'):
        return refuse('NO_AUTHORIZATION', 'earlier published active plan AND owner-issued window required')
    reservations = context.derived.get('rescue_reservations')
    if (not isinstance(reservations, Mapping) or set(reservations) != {'step', 'touches'}
            or type(reservations['step']) is not int or reservations['step'] != context.step
            or not isinstance(reservations['touches'], (list, tuple))
            or any(not isinstance(t, str) or not t.strip() for t in reservations['touches'])):
        return refuse('INCOMPLETE_RESERVATIONS', 'complete explicit same-step owner claims required')
    if (not isinstance(selected_others, (tuple, list)) or len(selected_others) > 128
            or any(not isinstance(p, Proposal) for p in selected_others)):
        return refuse('INCOMPLETE_RESERVATIONS', 'explicit bounded post-selection set of other Proposals required')
    if len({p.key for p in selected_others}) != len(selected_others):
        return refuse('INCOMPLETE_RESERVATIONS', 'duplicate selected proposal')
    if any(p.requirements.window is None or not p.requirements.window[0] <= context.step <= p.requirements.window[1]
           for p in selected_others):
        return refuse('INCOMPLETE_RESERVATIONS', 'selected proposal has no current-step reservation window')
    try:
        # Use the existing actor serialization, not a second context codec.
        adoption._wire([p.to_dict() for p in selected_others])
        claims = sorted(set(reservations['touches']).union(*(p.touches for p in selected_others)))
        current = replace(context, derived={**context.derived,
                          'rescue_reservations': {'step': context.step, 'touches': claims}})
        window = current.memory['declared_service_window']
        if claims != window.get('origin', {}).get('claims'):
            return refuse('INCOMPLETE_RESERVATIONS',
                          'post-selection claims differ from complete owner declaration; owner must reissue',
                          rescue_reservations={'step': context.step, 'touches': claims})
        check = commitments.validate_service_window(current, window)
        if check['status'] != 'READY':
            return refuse('NO_AUTHORIZATION', check.get('reason', 'owner window is no longer ready'))
        choices = [p for p in LateFeedRescue().propose(current)
                   if p.candidate['actor'] == window['worker'] and p.candidate['target'] == window['target']]
        if current.step != window['due_step'] or len(choices) != 1:
            return refuse('NO_ELIGIBLE_RESCUE', 'no unique eligible rescue at the actual owner due prefix')
        proposal, = choices
        guards = {phase: _rescue_guard_evidence(current, proposal, phase) for phase in ('ADMIT', 'EXECUTE')}
        for proof in guards.values():
            if proof['status'] != 'READY':
                return refuse(proof['status'], proof.get('reason', 'guard refused'), guards=guards)
        context_wire = adoption._wire(dict(step=current.step, player=current.player,
            observation=current.observation, memory=current.memory, shops=current.shops, derived=current.derived))
        permission = dict(schema='solver-rescue-actor-permission-v1',
            context_sha256=adoption.context_digest(current), proposal_sha256=adoption.digest(proposal.to_dict()),
            answers={phase: proof['answers'] for phase, proof in guards.items()})
        return dict(schema='solver-rescue-inputs-v1', status='READY', native_refusal=False,
            opportunity_rate='UNKNOWN', inputs=dict(context=context_wire, proposal=proposal.to_dict(),
                                                    actor_permission=permission),
            guards=guards, rescue_reservations=context_wire['derived']['rescue_reservations'],
            selection_scope='ONE_PREAUTHORIZED_SERVICE_NOT_VALUE_BASED_POLICY_SELECTION')
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        return refuse('NO_AUTHORIZATION', str(error))
    except RuntimeError as error:
        return refuse('SOURCE_OR_RUNTIME_DRIFT', str(error))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='JSON containing observation, baseline_action, reservations')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.input.read_text())
    obs = data['observation']
    ctx = StrategyContext(obs['step'], obs['player'], obs,
        memory={'baseline_action': data['baseline_action']},
        shops=tuple(obs.get('town', {}).get('unlocked_shops', ())),
        derived={'rescue_reservations': data.get('reservations')})
    text = json.dumps(LateFeedRescue().explain(ctx), sort_keys=True, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
