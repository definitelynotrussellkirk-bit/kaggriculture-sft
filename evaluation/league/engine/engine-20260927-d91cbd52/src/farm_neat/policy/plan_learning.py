"""Group-relative policy losses, learned plan evaluation and future-label sampling.

Owner: learning maintainer. Scope: math/representation used by the maintained
fast-runner's explicit grouped mode, not a second collector/trainer. Executed
receipts establish training, not this module. Tests: test_full_economic_plan_learning.py.

One group must contain complete games with identical world/seat/frozen-opponent
conditions and independent learner plan sampling. The caller's fresh witness
establishes those identities; tensors alone cannot authenticate a group.
Future labels never enter the actor. They describe factual outcomes under the
recorded policies, NOT counterfactual profitability or globally optimal labor.
The optional plan critic learns terminal outcomes as an auxiliary objective;
its predictions never replace simulator labels or enter group advantages.
"""
from __future__ import annotations

import copy
import math
import random

import torch
from torch import nn

SCHEMA = 'farm-full-economic-plan-grpo-aux-v1'
MARGIN_REWARD_SCHEMA = 'farm-terminal-asymmetric-wdl-margin-v1'
DRAW_PENALTY_REWARD_SCHEMA = 'farm-terminal-asymmetric-wdl-margin-draw-loss-v2'
# Physical snapshot values, except the explicitly named preceding dispatch
# diagnostics. None means unavailable, never an invented zero. Dollar values are
# bank balances, not hypothetical asset liquidation or a shared-wealth reward.
TARGETS = ('self_cash', 'opponent_cash', 'shed_units', 'carried_units', 'seed_units',
    'live_crop_tiles', 'live_animal_tiles', 'weed_tiles', 'hired_hands',
    'harvestable_units', 'planned_due_water_feed', 'planned_unassigned_jobs',
    'planned_market_slots', 'shed_free_units')
SCALES = (1000., 1000., 100., 100., 100., 25., 25., 25., 10., 100., 25., 25., 10., 100.)


def snapshot_targets(observation, previous_dispatch, configuration):
    """Extract post-turn auxiliary labels, never policy inputs or an admission.

    Uses only this seat's ordinary observation and the preceding turn's dispatcher
    report. Harvestable units means mature tile inventory, NOT guaranteed reachable
    harvest/deposit/sale. Job counts are preceding planned demand, NOT measured
    fulfillment or proof that hiring would help. Unknown own stock stays unknown.
    The independent engine witness must reproduce these same snapshots on intake.
    """
    from farm_neat.game.constants import CROPS, ANIMALS

    def number(value):
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative observed quantity required')
        return value

    def stock(value):
        if value is None: return None
        if not isinstance(value,dict): raise ValueError('observed inventory mapping or unknown required')
        return sum(number(n) for n in value.values())

    step,seat = observation['step'],observation['player']
    if (type(step) is not int or not 1 <= step <= 719 or type(seat) is not int or seat not in (0,1)
            or observation['day'] != step//24 or observation['hour'] != step%24):
        raise ValueError('post-turn physical clock and own seat required')
    farms,private = observation['farms'],observation['private']
    if not isinstance(farms,list) or len(farms)!=2 or not isinstance(private,dict):
        raise ValueError('both public farms and own private observation required')
    farm=farms[seat]; board=farm['tiles']; hands=farm['hands']
    if (not isinstance(board,list) or len(board)!=10 or any(not isinstance(row,list) or len(row)!=10 for row in board)
            or not isinstance(hands,list)):
        raise ValueError('complete public farm grid and hand positions required')
    plants=animals=weeds=harvestable=0
    for row in board:
        for tile in row:
            if tile is None or tile=='LOCKED': continue
            if not isinstance(tile,dict): raise ValueError('public tile required')
            if tile['kind']=='WEED': weeds+=1
            elif tile['kind']=='PLANT':
                plants+=1; spec=CROPS[tile['crop']]
                if step//24-number(tile['planted_day'])>=spec['first_yield_day']:
                    harvestable+=number(tile['yield_units'])
            elif tile.get('animal') is not None:
                if tile['animal'] not in ANIMALS: raise ValueError('known animal required')
                animals+=1;harvestable+=number(tile['yield_units'])
    shed=stock(private.get('shed')); seed=stock(private.get('seeds'))
    inventories=private.get('inventories'); carried=None
    if inventories is not None:
        if not isinstance(inventories,list) or len(inventories)!=len(hands)+1:
            raise ValueError('own farmer and hand inventories must align')
        totals=[stock(inv) for inv in inventories]
        carried=None if None in totals else sum(totals)
    report=previous_dispatch or {}
    if report and report.get('step') != step-1:
        raise ValueError('dispatch labels must describe the immediately preceding turn')
    def demand(name):
        return number(report[name]) if report.get(name) is not None else None
    return dict(zip(TARGETS,(number(farm['money']),number(farms[1-seat]['money']),shed,carried,seed,
        plants,animals,weeds,len(hands),harvestable,demand('due_water_feed'),demand('unassigned_jobs'),
        demand('market_slots_used'),None if shed is None else number(configuration.get('shedCapacity',100))-shed)))


def _finite_tensor(value, *, ndim):
    if (not isinstance(value, torch.Tensor) or value.ndim != ndim
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError('finite floating tensor of declared rank required')


def group_advantages(rewards):
    """[groups, trials] -> detached within-group population-standardized rewards.

    Fixed complete groups, terminal LOSS/DRAW/WIN only. All-equal groups have
    exact zero actor signal; do not omit them or count them as informative.
    """
    _finite_tensor(rewards, ndim=2)
    if rewards.shape[0] < 1 or rewards.shape[1] < 2 or not bool(((rewards == -1) | (rewards == 0) | (rewards == 1)).all()):
        raise ValueError('at least two complete terminal W/D/L trials per group')
    r = rewards.detach().double()
    centered = r-r.mean(1, keepdim=True)
    std = centered.square().mean(1, keepdim=True).sqrt()
    informative = std[:, 0] > 0
    advantages = torch.where(std > 0, centered/std.clamp_min(1e-12), torch.zeros_like(r))
    return advantages, {'groups':len(r), 'trials_per_group':r.shape[1],
        'informative_groups':int(informative.sum()), 'tied_groups':int((~informative).sum()),
        'critic_used':False, 'independent_world_count':'CALLER_WITNESS_REQUIRED'}


def validate_margin_reward(config):
    """Explicit objective change; never the default or a ladder score definition."""
    required={'schema','weight','cash_scale','deadband','loss_weight','loss_cash_scale','normalization'}
    draw_penalty=isinstance(config,dict) and config.get('schema')==DRAW_PENALTY_REWARD_SCHEMA
    if draw_penalty: required.add('draw_reward')
    if (not isinstance(config, dict) or set(config) != required
            or config['schema'] not in (MARGIN_REWARD_SCHEMA,DRAW_PENALTY_REWARD_SCHEMA)
            or (draw_penalty and (type(config['draw_reward']) not in (int,float) or config['draw_reward'] != -2))
            or config['normalization'] != 'GROUP_MEAN_ONLY'
            or type(config['weight']) not in (int,float) or not math.isfinite(config['weight'])
            or not 0 < config['weight'] <= .25
            or type(config['cash_scale']) not in (int,float) or not math.isfinite(config['cash_scale'])
            or config['cash_scale'] <= 0
            or type(config['deadband']) not in (int,float) or not math.isfinite(config['deadband'])
            or config['deadband'] < 0
            or type(config['loss_weight']) not in (int,float) or not math.isfinite(config['loss_weight'])
            or not 0 < config['loss_weight'] <= 4
            or type(config['loss_cash_scale']) not in (int,float) or not math.isfinite(config['loss_cash_scale'])
            or config['loss_cash_scale'] <= 0):
        raise ValueError('versioned asymmetric terminal margin reward and group-mean-only normalization required')
    return copy.deepcopy(config)


def margin_group_advantages(wdl, terminal_cash, config):
    """[G,K] WDL + [G,K,2] own/opponent witnessed bank cash -> detached advantages.

    Win: 1 + weight * excess/(cash_scale + excess), excess=max(delta-deadband,0).
    Draw: 0 in frozen v1, -2 in explicit owner draw-loss v2.
    Loss: -1-loss_weight*log1p(deficit/loss_cash_scale).
    Subtract the group mean;
    do NOT divide by group std (that would erase the bonus weight in all-loss
    groups). Losses have no deadband; log1p distinguishes even large losses
    without their linear magnitude dominating. The WIN bonus alone is bounded.
    V1 retains win/draw/loss ordering; v2 intentionally prefers a small loss
    to a draw. Neither rewards mere action. Expected shaped return
    is NOT identical to maximizing win probability, and this asymmetric reward
    is not zero-sum. Evaluate on unshaped WDL.
    This is terminal margin feedback, not a dense capital-use reward.
    """
    config = validate_margin_reward(config)
    _, original = group_advantages(wdl)  # strict terminal W/D/L validation
    _finite_tensor(terminal_cash, ndim=3)
    if (terminal_cash.shape != (*wdl.shape,2) or terminal_cash.device != wdl.device
            or bool((terminal_cash < 0).any())):
        raise ValueError('aligned nonnegative witnessed own/opponent terminal bank cash required')
    cash = terminal_cash.detach().double()
    delta = cash[:,:,0]-cash[:,:,1]
    if not bool((delta.sign() == wdl.detach()).all()):
        raise ValueError('terminal cash and W/D/L disagree')
    excess = (delta-config['deadband']).clamp_min(0)
    win_bonus = config['weight'] * (excess/(config['cash_scale']+excess))
    loss_penalty = config['loss_weight'] * torch.log1p((-delta).clamp_min(0)/config['loss_cash_scale'])
    bonus = torch.where(delta < 0, -loss_penalty, win_bonus)
    # Center separately to avoid subtracting near-equal -1 values for small bonuses.
    r = wdl.detach().double()
    if config['schema']==DRAW_PENALTY_REWARD_SCHEMA:
        r=torch.where(r==0,r.new_tensor(config['draw_reward']),r)
    advantages = r-r.mean(1,keepdim=True) + bonus-bonus.mean(1,keepdim=True)
    tied = ((r == r[:,:1]) & (bonus == bonus[:,:1])).all(1)
    advantages[tied] = 0  # exact identical terminal results never rank
    rewards = r+bonus
    return advantages, {**original, 'reward':config,
        'wdl_informative_groups':original['informative_groups'],
        'informative_groups':int((~tied).sum()), 'tied_groups':int(tied.sum()),
        'margin_only_groups':int(((r == r[:,:1]).all(1) & ~tied).sum()),
        'reward_min':float(rewards.min()), 'reward_max':float(rewards.max()),
        'margin_advantage_abs_max':float((bonus-bonus.mean(1,keepdim=True)).abs().max()),
        'normalization':'GROUP_MEAN_ONLY', 'strength_metric':'UNSHAPED_WDL',
        'loss_penalty_is_bounded':False, 'reward_is_zero_sum':False}


def grpo_terms(logp, old_logp, advantages, entropy, *, clip_ratio=.2, entropy_coefficient=.01):
    """Equal-program clipped group-relative actor loss; no V, TD or value loss.

    Program probability is the product of its conditional factors (sum logp),
    including sampled STOP. No per-factor time discount or length normalization.
    Reference-policy KL is not enabled here; returned behavior KL is a diagnostic.
    Old probabilities and advantages are labels and are always detached.
    """
    for t in (logp, old_logp, advantages, entropy): _finite_tensor(t, ndim=1)
    if (not len(logp) or any(t.shape != logp.shape or t.device != logp.device for t in (old_logp, advantages, entropy))
            or type(clip_ratio) not in (int,float) or not 0 < clip_ratio < 1
            or type(entropy_coefficient) not in (int,float) or not math.isfinite(entropy_coefficient)
            or entropy_coefficient < 0 or bool((entropy < 0).any())
            or bool((logp > 1e-6).any()) or bool((old_logp > 1e-6).any())):
        raise ValueError('aligned finite program scores and bounded coefficients required')
    ratio = (logp-old_logp.detach()).exp()
    if not bool(torch.isfinite(ratio).all()): raise ValueError('nonfinite importance ratio')
    a = advantages.detach()
    actor = -torch.minimum(ratio*a, ratio.clamp(1-clip_ratio, 1+clip_ratio)*a).mean()
    ent = entropy.mean()
    return {'loss':actor-entropy_coefficient*ent, 'actor_loss':actor,
        'entropy':ent, 'clip_fraction':((ratio-1).abs()>clip_ratio).double().mean(),
        'sampled_behavior_kl':(ratio-1-(logp-old_logp.detach())).mean()}


def plan_context(graph, context, records):
    """Replay known choices through the EXISTING policy prefix, without logits.

    [B,D] state context -> [B,D] completed-plan context, with gradients to the
    shared encoder, option/path embeddings and prefix GRU. This does not sample,
    advance a game or differentiate through discrete choices/the engine.
    """
    from farm_neat.policy import network as policy
    from farm_neat.rollout.batched_replay import _paths
    _finite_tensor(context, ndim=2)
    if len(records) != len(context) or not len(context) or context.shape[1] != graph.config.width:
        raise ValueError('one complete factor record per state context required')
    sequences = _paths(policy, graph, records)
    hidden = context
    for pos in range(max(map(len, sequences))):
        active = [i for i, seq in enumerate(sequences) if pos < len(seq)]
        paths, depths, options = [], [], []
        for i in active:
            (path, _, _, _, _), choices, selected = sequences[i][pos]
            base, depth = policy._path(path)
            paths.append(policy.PATH_IDS[base]); depths.append(depth)
            options.append(policy.OPTION_IDS[policy._key(choices[selected])])
        indices = torch.tensor(active, device=context.device)
        p = graph.path_embedding(torch.tensor(paths, device=context.device))
        p = p+graph.replacement_embedding(torch.tensor(depths, device=context.device))
        token = graph.option_embedding(torch.tensor(options, device=context.device))+p
        advanced = graph.prefix(token, hidden.index_select(0, indices))
        hidden = hidden.index_copy(0, indices, advanced)
    return hidden


class PlanCritic(nn.Module):
    """State + completed-plan contexts [B,D] -> terminal LOSS/DRAW/WIN logits.

    The shared encoder and prefix receive gradients. Unlike the legacy pre-action
    value head, this head can distinguish plans in the identical opening state.
    Targets refer to the recorded executor/continuation/opponent distribution,
    not the optimal value or knowledge of the realized hidden world. Retain this
    head at inference only if plan-selection evaluation supports its use.
    """
    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError('positive hidden width required')
        self.width = width
        self.net = nn.Sequential(nn.Linear(2*width,width), nn.SiLU(), nn.Linear(width,3))

    def forward(self, state, plan):
        _finite_tensor(state, ndim=2); _finite_tensor(plan, ndim=2)
        if (state.shape != plan.shape or not len(state) or state.shape[1] != self.width
                or state.device != plan.device or state.dtype != plan.dtype):
            raise ValueError('aligned nonempty state and sampled-plan contexts required')
        return self.net(torch.cat((state,plan),dim=-1))

    @staticmethod
    def score(logits):
        """Expected competition score P(win)+0.5P(draw), not cash or a label."""
        _finite_tensor(logits, ndim=2)
        if logits.shape[1] != 3 or not len(logits):
            raise ValueError('nonempty LOSS/DRAW/WIN logits required')
        p = logits.softmax(-1)
        return p[:,2]+.5*p[:,1]


def plan_critic_terms(logits, rewards):
    """Equal-plan proper log loss on externally witnessed terminal -1/0/+1.

    Caller authenticates complete game, observation/plan and source identities.
    Repeated-world estimates are not independent-world strength measurements.
    Do not train on this critic's own predictions or an ex-post best-plan label.
    Metrics are detached; this loss does not change GRPO advantages/rewards.
    """
    _finite_tensor(logits, ndim=2); _finite_tensor(rewards, ndim=1)
    if (logits.shape != (len(rewards),3) or not len(rewards) or logits.device != rewards.device
            or not bool(((rewards == -1) | (rewards == 0) | (rewards == 1)).all())):
        raise ValueError('one actual terminal LOSS/DRAW/WIN outcome per plan required')
    classes = rewards.detach().long()+1
    loss = nn.functional.cross_entropy(logits,classes)
    with torch.no_grad():
        probabilities = logits.softmax(-1)
        labels = nn.functional.one_hot(classes,3).to(probabilities.dtype)
        brier = (probabilities-labels).square().sum(-1).mean()
        score = PlanCritic.score(logits)
    return {'loss':loss, 'brier':brier, 'expected_score':score,
        'observed_score':(rewards.detach()+1)/2, 'used_for_grpo_advantage':False}


class FutureHeads(nn.Module):
    """Training-only [B,D] state + plan + [B,Q] physical horizon -> [B,Q,14].

    Predictions use signed log1p(value/scale). Query time is a horizon, not future
    state. Keeping these parameters separate lets deployment discard the heads.
    They are auxiliary forecasts, never a baseline used in GRPO advantages.
    """
    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width < 1: raise ValueError('positive hidden width required')
        self.width = width
        self.net = nn.Sequential(nn.Linear(2*width+2,width), nn.SiLU(), nn.Linear(width,len(TARGETS)))

    def forward(self, state, plan, horizon):
        _finite_tensor(state, ndim=2); _finite_tensor(plan, ndim=2); _finite_tensor(horizon, ndim=2)
        if (state.shape != plan.shape or state.shape[1] != self.width or horizon.shape[0] != len(state)
                or not horizon.shape[1] or bool(((horizon < 1) | (horizon > 719)).any())
                or bool((horizon != horizon.floor()).any())):
            raise ValueError('aligned state/plan and1..719 whole physical-turn horizons required')
        q = torch.stack((horizon/719., horizon.log1p()/math.log(720)),dim=-1).to(state.dtype)
        c = torch.cat((state,plan),dim=-1)[:,None,:].expand(-1,horizon.shape[1],-1)
        return self.net(torch.cat((c,q),dim=-1))


def future_loss(prediction, targets, known, query_weights):
    """Per-query known-field mean, then explicit normalized query mixture/plan.

    Raw numeric labels are transformed here; unknown entries cannot contribute
    to loss or gradient. Samples selected around future boundaries are a targeted
    supervised distribution, not an unbiased all-turn performance estimate.
    """
    _finite_tensor(prediction, ndim=3)
    if (targets.shape != prediction.shape or known.shape != prediction.shape or known.dtype != torch.bool
            or prediction.shape[-1] != len(TARGETS) or query_weights.shape != prediction.shape[:2]
            or any(t.device != prediction.device for t in (targets,known,query_weights))
            or not targets.is_floating_point() or not bool(torch.isfinite(targets[known]).all())):
        raise ValueError('aligned numeric targets, knownness and per-query weights required')
    _finite_tensor(query_weights, ndim=2)
    if bool((query_weights < 0).any()): raise ValueError('nonnegative query weights required')
    safe = targets.detach().masked_fill(~known,0)
    scale = prediction.new_tensor(SCALES)
    encoded = safe.sign()*torch.log1p(safe.abs()/scale)
    error = torch.nn.functional.smooth_l1_loss(prediction,encoded,reduction='none').masked_fill(~known,0)
    count = known.sum(-1)
    weights = query_weights.detach()*(count > 0)
    mass = weights.sum(-1)
    if bool((mass <= 0).any()): raise ValueError('at least one positive-weight known future query per plan')
    return ((error.sum(-1)/count.clamp_min(1)*weights).sum(-1)/mass).mean()


class BoundarySampler:
    """Two bounded reservoirs: uniform future steps and near/crossed boundaries.

    Caller supplies exact observed metrics and named signed slack in declared
    units (e.g. cash-cost dollars or capacity-used units), not inferred outcomes.
    Label extraction/witness remains caller-owned. Selection uses outcome history
    ONLY for auxiliary sampling; results must never be fed to the policy.
    """
    def __init__(self, *, seed, per_stratum=4, tolerances):
        if type(seed) is not int or seed < 0 or type(per_stratum) is not int or not 1 <= per_stratum <= 32:
            raise ValueError('explicit sampling seed and bounded reservoir required')
        if not tolerances or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in tolerances.values()):
            raise ValueError('named boundary tolerances in physical units required')
        self.rng = random.Random(seed); self.seed = seed; self.cap = per_stratum
        self.tolerances = dict(tolerances); self.previous = None
        self.pools = {'uniform':[], 'boundary':[]}; self.seen = {'uniform':0, 'boundary':0}
        self.boundary_steps = set()

    def _add(self, kind, row):
        if kind == 'boundary':
            if row['step'] in self.boundary_steps: return
            self.boundary_steps.add(row['step'])
        self.seen[kind] += 1; pool = self.pools[kind]
        if len(pool) < self.cap: pool.append(copy.deepcopy(row)); return
        index = self.rng.randrange(self.seen[kind])
        if index < self.cap: pool[index] = copy.deepcopy(row)

    def observe(self, *, step, values, slack):
        if (type(step) is not int or not 1 <= step <= 719 or (self.previous and step != self.previous['step']+1)
                or (self.previous is None and step != 1) or set(values) != set(TARGETS)
                or set(slack) != set(self.tolerances)):
            raise ValueError('consecutive future steps, exact target and slack schema required')
        for v in (*values.values(),*slack.values()):
            if v is not None and (type(v) not in (int,float) or not math.isfinite(v)):
                raise ValueError('finite observed values or explicit unknown required')
        row = {'step':step,'values':dict(values),'slack':dict(slack)}
        self._add('uniform',row)
        near = any(v is not None and abs(v) <= self.tolerances[k] for k,v in slack.items())
        crossed = self.previous and any(v is not None and self.previous['slack'][k] is not None
            and (v >= 0) != (self.previous['slack'][k] >= 0) for k,v in slack.items())
        if crossed: self._add('boundary',self.previous)
        if near or crossed: self._add('boundary',row)
        self.previous = row

    def finish(self, *, horizon=719):
        # A late plan's queries are relative to its decision. The caller still
        # authenticates a complete game ending at719; this never labels a cutoff.
        if type(horizon) is not int or not 1 <= horizon <= 719:
            raise ValueError('explicit remaining physical horizon required')
        if not self.previous or self.previous['step'] != horizon:
            raise ValueError('complete719-turn outcome required; no censored terminal credit')
        present = [k for k,p in self.pools.items() if p]
        selected = {}
        for kind in present:
            weight = 1/len(present)/len(self.pools[kind])
            for row in self.pools[kind]:
                item = selected.setdefault(row['step'],{**copy.deepcopy(row),'weight':0.,'strata':[]})
                item['weight'] += weight; item['strata'].append(kind)
        return {'schema':SCHEMA+'-future-sample','seed':self.seed,'tolerances':dict(self.tolerances),
            **({'physical_horizon':horizon} if horizon != 719 else {}),
            'candidate_counts':dict(self.seen),'queries':[selected[k] for k in sorted(selected)],
            'future_selected_targets_not_actor_inputs':True,'training_admission':False}
