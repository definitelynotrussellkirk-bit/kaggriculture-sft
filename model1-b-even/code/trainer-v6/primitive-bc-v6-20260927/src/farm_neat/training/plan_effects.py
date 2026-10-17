"""Supervise existing plan forecasts on witnessed KEEP/edit terminal outcomes.

This stage reuses campaign.Learner, its two optimizers, schedule and checkpoint
transaction. It does not invent policy likelihoods for explicitly authored plans.
"""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from farm_neat.evaluation.common import canonical,digest,equal,file_sha,read,write_new
from farm_neat.training import campaign,program_bc,moe_lab,sustain
from farm_neat.training.schedule import learning_rate_at
from farm_neat.policy import network,plan_observation,plan_learning
from farm_neat.economy.orders import FullSeasonPlan


def project_plans(given, programs):
    """Publish explicit requests without stepping; merge only identical tensors.

    Metadata, outcome labels and hidden seeds never determine an alias. Preserve
    each alternative's loss weight even when two requests encode identically.
    """
    encoder_identity=plan_observation.identity();unique=[];indices=[];seen={}
    for program in programs:
        ctl=FullSeasonPlan.restore(given['snapshot'])
        proposal=ctl.propose(given['observation'],program)
        if proposal['rejected']:raise ValueError('refused candidate cannot enter outcome fitting')
        ctl=ctl.publish(given['observation'],proposal)
        pack=plan_observation.pack(given['observation'],ctl,given['configuration'],expected_identity=encoder_identity)
        h=hashlib.sha256()
        for name in ('rows','knownness','token_mask'):
            value=pack[name];h.update(str((name,value.dtype.str,value.shape)).encode());h.update(value.tobytes())
        key=h.hexdigest()
        if key not in seen:
            seen[key]=len(unique);unique.append({'pack':pack,'tensor_sha256':key})
        indices.append(seen[key])
    return unique,indices


def pack_group(identity,given,menu,graph,plan_context='raw_factor_prefix'):
    """Actor-visible state and explicit candidates only; no terminal label input."""
    ctl=FullSeasonPlan.restore(given['snapshot'])
    packed=plan_observation.pack(given['observation'],ctl,given['configuration'],expected_identity=plan_observation.identity())
    architecture=network.architecture(graph.config)['content_sha256']
    if plan_context not in ('raw_factor_prefix','published_plan_encoder'):raise ValueError('known plan context required')
    result={'identity':identity,'pack':packed,'traces':[program_bc.target_trace(program,
        step=identity['step'],max_commands=64,architecture_sha256=architecture) for program in menu.values()],
        'edits':list(menu),'horizon':719-identity['step'],'input_sha256':packed['content_sha256']}
    if plan_context=='published_plan_encoder':
        projected,indices=project_plans(given,menu.values())
        result.update(plan_context=plan_context,projected=projected,projection_indices=indices)
    return result


def load_groups(directory, expected_sha, graph,plan_context='raw_factor_prefix'):
    """Decode/validate/pack once. Repeated optimizer presentations reuse these packs."""
    directory=Path(directory);equal(file_sha(directory/'RESULT.json'),expected_sha,'exact completed data')
    result=read(directory/'RESULT.json');plan=read(directory/'PLAN.json')
    equal(result['plan_sha256'],file_sha(directory/'PLAN.json'),'data registration')
    if not result['complete'] or len(result['groups'])!=len(plan['groups']):
        raise ValueError('complete registered paired data required')
    registration=ROOT/result['resume_path'] if result.get('resume_path') else directory/'RESUME.json'
    operative=read(registration) if result.get('resume_sha256') else plan
    if result.get('resume_sha256'):equal(file_sha(registration),result['resume_sha256'],'data repair')
    for name,sha in operative['source_pins'].items():equal(file_sha(ROOT/name),sha,'data source '+name)
    groups=[];seen=set()
    for entry in result['groups']:
        identity=entry['group'];key=digest(identity)
        if key in seen or identity not in plan['groups'] or not entry['complete'] or not entry['no_op_alias_verified']:
            raise ValueError('unique complete registered groups with verified KEEP alias required')
        seen.add(key);path=ROOT/entry['file'];equal(file_sha(path),entry['sha256'],'actual game group')
        with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
        menu=plan['edits'][identity['base']]
        if len(rows)!=len(menu) or rows[0]['edit']!='keep' or {r['edit'] for r in rows}!=set(menu):
            raise ValueError('complete alternative set with KEEP at position0 required')
        given=rows[0]['input'];equal(digest(given),rows[0]['input_sha256'],'actor input')
        baseline=rows[0];cash=[]
        for row in rows:
            equal(row['group'],identity,'world/seat/phase/opponent identity')
            equal(row['input_sha256'],baseline['input_sha256'],'same state')
            equal(row['program'],menu[row['edit']],'actual economic request')
            equal(len(row['actions']),719,'complete physical game')
            equal(digest(row['actions']),row['actions_sha256'],'retained actions')
            equal(digest(row['actions'][:identity['step']]),baseline['prefix_actions_sha256'],'matched executed prefix')
            if row['physical_steps']!=719 or any(p['rejected'] for p in row['publications']):
                raise ValueError('complete unrefused authored alternatives required')
            farms=row['terminal_observation']['farms'];seat=identity['seat']
            terminal=[float(farms[seat]['money']),float(farms[1-seat]['money'])]
            equal(terminal,[row['own_cash'],row['opponent_cash']],'actual terminal labels')
            if any(x<0 for x in terminal):raise ValueError('negative terminal cash sentinel')
            if row['edit']=='hands_same':equal(row['actions_sha256'],baseline['actions_sha256'],'no-op alias realization')
            cash.append(terminal)
        ordered={r['edit']:r['program'] for r in rows}
        groups.append({**pack_group(identity,given,ordered,graph,plan_context),'cash':cash})
    return groups


def forward(learner,groups):
    """Encode each state once; score its alternatives through the existing plan prefix."""
    import torch
    mode=getattr(learner,'execution',{}).get('plan_context','raw_factor_prefix')
    gain=campaign.validate_plan_context_gain(getattr(learner,'execution',{}).get('plan_context_gain',1.))
    if not groups or any(g.get('plan_context','raw_factor_prefix')!=mode for g in groups):
        raise ValueError('group representation must match the declared learner execution')
    if gain!=1. and (mode!='published_plan_encoder' or any(not g.get('edits') or g['edits'][0]!='keep' for g in groups)):
        raise ValueError('KEEP at position zero in each published-plan group required for context gain')
    context=encode_context(learner,groups)
    sizes=[len(g['traces']) for g in groups]
    counts=torch.tensor(sizes,device=context.device)
    state=torch.repeat_interleave(context,counts,dim=0)
    if mode=='published_plan_encoder':
        packs=[];indices=[]
        for group in groups:
            selected=group['projection_indices']
            if len(selected)!=len(group['traces']) or any(type(i) is not int or not 0<=i<len(group['projected']) for i in selected):
                raise ValueError('complete within-state projected indices required')
            indices.extend(len(packs)+i for i in selected);packs.extend(group['projected'])
        plans=encode_context(learner,packs)
        plans=plans.index_select(0,torch.tensor(indices,device=state.device))
        if gain!=1.:
            starts=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
            keep=plans.index_select(0,starts).repeat_interleave(counts,dim=0)
            plans=keep+gain*(plans-keep)
    else:
        records=[r for g in groups for r in g['traces']]
        plans=plan_learning.plan_context(learner.graph,state,records)
    horizon=torch.tensor([[g['horizon']] for g in groups for _ in g['traces']],device=state.device,dtype=state.dtype)
    logits=learner.auxiliary['critic'](state,plans)
    encoded=learner.auxiliary['future'](state,plans,horizon)[:,0,:2]
    return logits,encoded,sizes


def kind_mean(tokens,packs):
    """Equal weight per present entity kind; padding and absent kinds get none."""
    import numpy as np
    import torch
    kinds=len(plan_observation.codec.KINDS)
    if tokens.ndim!=3 or len(tokens)!=len(packs):raise ValueError('aligned encoded tokens and packs required')
    weights=np.zeros(tuple(tokens.shape[:2]),dtype=np.float32)
    for index,pack in enumerate(packs):
        rows=pack['rows'];mask=pack['token_mask'];known=pack['knownness'][:,:kinds]
        if len(rows)>tokens.shape[1] or not mask.any():raise ValueError('nonempty valid token set required')
        typed=rows[:,:kinds]
        if (not known[mask].all() or not np.isin(typed[mask],[0.,1.]).all()
                or not (typed[mask].sum(1)==1).all()):raise ValueError('one known entity kind per valid token required')
        typed=typed*mask[:,None];counts=typed.sum(0);present=counts>0
        weights[index,:len(rows)]=(typed/np.maximum(counts,1)).sum(1)/present.sum()
    return (tokens*torch.as_tensor(weights,device=tokens.device,dtype=tokens.dtype)[...,None]).sum(1)


def encode_context(learner,batch):
    """The original full encoder; an explicit optional readout of final tokens."""
    readout=getattr(learner,'execution',{}).get('encoder_readout','mean')
    if readout=='mean':return program_bc.encode_batch(learner,batch)[0]
    if readout!='kind_mean':raise ValueError('known entity encoder readout required')
    captured=[]
    hook=learner.graph.final_norm.register_forward_hook(lambda module,args,result:captured.append(result))
    try:program_bc.encode_batch(learner,batch)
    finally:hook.remove()
    if len(captured)!=1:raise ValueError('one full encoder final-token tensor required')
    return kind_mean(captured[0],[g['pack'] for g in batch])


def decode_cash(encoded,precision='float32'):
    """Versioned forecast arithmetic; the historical model-dtype path is exact."""
    import torch
    if precision not in ('float32','float64'):raise ValueError('known cash diagnostic precision required')
    if precision=='float64':encoded=encoded.to(torch.float64)
    raw=encoded.sign()*encoded.abs().expm1()*1000.
    if not bool(torch.isfinite(raw).all()):raise ValueError('nonfinite decoded forecast')
    return raw


def terms(logits,encoded,cash,sizes,effect_coefficient=1.,effect_objective='joint',effect_diagnostic_precision='float32'):
    """Equal state groups; no-op aliases retain a real zero-delta target."""
    import torch
    effect_coefficient=campaign.validate_effect_coefficient(effect_coefficient,effect_objective)
    if effect_diagnostic_precision=='float64' and effect_objective=='joint':raise ValueError('float64 diagnostics require inactive cash losses')
    if logits.shape!=(len(cash),3) or encoded.shape!=cash.shape or cash.shape[1]!=2 or sum(sizes)!=len(cash) or any(n<2 for n in sizes):
        raise ValueError('aligned complete KEEP/edit groups and two cash labels required')
    if not bool(torch.isfinite(cash).all()) or bool((cash<0).any()):raise ValueError('finite actual cash labels')
    raw=decode_cash(encoded,effect_diagnostic_precision)
    classes=(cash[:,0]>cash[:,1]).long()*2+(cash[:,0]==cash[:,1]).long()
    ce=torch.nn.functional.cross_entropy(logits,classes,reduction='none')
    absolute=torch.nn.functional.smooth_l1_loss(encoded,torch.log1p(cash/1000.),reduction='none').mean(-1)
    pieces={'wdl':[],'cash':[],'effect':[]};start=0
    for size in sizes:
        stop=start+size;p=raw[start:stop,0]-raw[start:stop,1];y=cash[start:stop,0]-cash[start:stop,1]
        pieces['wdl'].append(ce[start:stop].mean());pieces['cash'].append(absolute[start:stop].mean())
        pieces['effect'].append(torch.nn.functional.smooth_l1_loss((p[1:]-p[0])/10000.,(y[1:]-y[0])/10000.))
        start=stop
    losses={k:torch.stack(v).mean() for k,v in pieces.items()}
    losses['total']=(.1*losses['wdl'] if effect_objective in ('wdl_only','wdl_contrast','wdl_contrast_rank')
        else .1*losses['wdl']+losses['cash']+effect_coefficient*losses['effect'])
    if effect_objective in ('wdl_contrast','wdl_contrast_rank'):
        score=logits.softmax(-1)@logits.new_tensor([0.,.5,1.]);target=classes.to(score.dtype)*.5
        contrast=[];start=0
        for size in sizes:
            stop=start+size
            error=(score[start+1:stop]-score[start])-(target[start+1:stop]-target[start])
            contrast.append(error.square().mean());start=stop
        losses['wdl_contrast']=torch.stack(contrast).mean()
        losses['total']=losses['total']+losses['wdl_contrast']
        if effect_objective=='wdl_contrast_rank':
            ranks=[];start=0
            for size in sizes:
                stop=start+size;p=score[start:stop];y=target[start:stop]
                target_gap=y[:,None]-y[None,:]
                ordered=target_gap>0
                shortfall=(target_gap-(p[:,None]-p[None,:])).relu().square()
                # All tied groups contribute zero without a host/GPU synchronization.
                ranks.append((shortfall*ordered).sum()/ordered.sum().clamp_min(1))
                start=stop
            losses['wdl_rank']=torch.stack(ranks).mean()
            losses['total']=losses['total']+losses['wdl_rank']
    return losses,raw


def forecasts(learner,groups):
    """Frozen predictions and deterministic choices without reading outcomes."""
    import torch
    with torch.inference_mode():
        logits,encoded,sizes=forward(learner,groups)
        raw=decode_cash(encoded,getattr(learner,'execution',{}).get('effect_diagnostic_precision','float32'))
        probabilities=logits.softmax(-1).cpu().tolist();predicted=raw.cpu().tolist()
    rows=[];offset=0
    for g,size in zip(groups,sizes,strict=True):
        cash=predicted[offset:offset+size];p=probabilities[offset:offset+size]
        rows.append({'group':g['identity'],'input_sha256':g['input_sha256'],'edits':g['edits'],
            'predicted_cash':cash,'wdl_probabilities':p,
            'margin_choice':max(range(size),key=lambda i:(cash[i][0]-cash[i][1],-i)),
            'wdl_choice':max(range(size),key=lambda i:(p[i][2]+.5*p[i][1],-i))})
        offset+=size
    return rows


def restore_state(learner):
    import torch
    return {'actor':moe_lab._state_digest(learner.graph.state_dict()),'adam':moe_lab._state_digest(learner.optimizer.state_dict()),
        'heads':moe_lab._state_digest(learner.auxiliary.state_dict()),'head_adam':moe_lab._state_digest(learner.aux_optimizer.state_dict()),
        'cpu_rng':moe_lab._state_digest(torch.get_rng_state()),'cuda_rng':moe_lab._state_digest(torch.cuda.get_rng_state_all())}


def future_state(learner):
    """Exact unused-head and inherited optimizer state for the WDL-only control."""
    head=learner.auxiliary['future']
    return {'weights':moe_lab._state_digest(head.state_dict()),
        'adam':moe_lab._state_digest({name:learner.aux_optimizer.state.get(parameter,{})
            for name,parameter in head.named_parameters()})}


def zero_first_moments(optimizer, parameters):
    """Explicit experiment: preserve parameters, second moments, ages and maps."""
    import torch
    targets=[]
    for parameter in parameters:
        state=optimizer.state.get(parameter,{})
        if not state:continue
        if 'exp_avg' in state:
            targets.append((state['exp_avg'],))
        elif 'state1' in state:
            value=state['state1']
            if value.dtype==torch.uint8:
                if 'absmax1' not in state:raise ValueError('blockwise first-moment scale required')
                targets.append((value,state['absmax1']))
            elif value.is_floating_point():targets.append((value,))
            else:raise ValueError('unsupported first-moment storage')
        else:raise ValueError('unrecognized optimizer first moment')
    with torch.no_grad():
        for tensors in targets:
            for tensor in tensors:tensor.zero_()
    return len(targets)


def reset_adam_state(optimizer, parameters):
    """Discard selected Adam moments/ages; retain weights, groups and other state.

    The optimizer initializes fresh state on the next actual gradient. This is
    a distinct intervention from zeroing momentum at an inherited tensor age.
    """
    parameters=list(parameters)
    owned={id(p) for group in optimizer.param_groups for p in group['params']}
    if len({id(p) for p in parameters})!=len(parameters) or any(id(p) not in owned for p in parameters):
        raise ValueError('unique optimizer-owned reset parameters required')
    targets=[]
    for parameter in parameters:
        state=optimizer.state.get(parameter,{})
        if not state:continue
        if not ({'step','exp_avg','exp_avg_sq'}<=state.keys() or {'step','state1','state2'}<=state.keys()):
            raise ValueError('recognized Adam state required before reset')
        targets.append(parameter)
    for parameter in targets:del optimizer.state[parameter]
    return len(targets)


def restore_verification(args):
    """Fresh-process restore of actor, both Adams, heads, RNG, age and predictions."""
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('SLURM GPU allocation required')
    parent=sustain.checkpoint(ROOT,args.parent);directory=ROOT/parent['directory']
    equal(parent['model_sha256'],args.model_sha256,'child actor');equal(parent['optimizer_sha256'],args.optimizer_sha256,'child optimizer')
    plan=sustain.read(directory/'PLAN.json');receipt=read(directory/'RECEIPT.json')
    learner=campaign.Learner(directory,parent,checkpoint_blocks=True,plan_learning=plan['execution']['plan_learning'],shop_revision_every=2,full_plan=True)
    learner.execution=campaign.effect_execution(plan['execution'])
    if plan.get('deterministic_algorithms',False):
        import torch
        torch.use_deterministic_algorithms(True)
    equal(restore_state(learner),receipt['ending_state'],'fresh-process exact actor, optimizers, heads and RNG')
    equal(learner.age,receipt['global_optimizer_steps_after'],'restored age')
    groups=load_groups(args.data,args.data_sha256,learner.graph,learner.execution.get('plan_context','raw_factor_prefix'))
    actual=measure(learner,groups,plan['microbatch_groups']);expected=read(directory/'AFTER.json')
    expected.pop('content_sha256',None)
    equal(actual,expected,'identical before-restart and fresh-process predictions')
    write_new(directory/'RESTORE.json',{'status':'EXACT_RESTORE_AND_TRAINING_FORECASTS_VERIFIED',
        'model_sha256':args.model_sha256,'optimizer_sha256':args.optimizer_sha256,'new_games':0,'updates':0,
        'groups':len(groups),'ending_state':restore_state(learner)})


def measure(learner,groups,microbatch=2):
    import torch
    rows=[];totals={k:0. for k in ('wdl','cash','effect','total')}
    if learner.execution.get('effect_objective') in ('wdl_contrast','wdl_contrast_rank'):totals['wdl_contrast']=0.
    if learner.execution.get('effect_objective')=='wdl_contrast_rank':totals['wdl_rank']=0.
    with torch.inference_mode():
        for start in range(0,len(groups),microbatch):
            part=groups[start:start+microbatch];logits,encoded,sizes=forward(learner,part)
            labels=torch.tensor([row for g in part for row in g['cash']],device=encoded.device)
            loss,raw=terms(logits,encoded,labels,sizes,learner.execution.get('effect_coefficient',1.),learner.execution.get('effect_objective','joint'),learner.execution.get('effect_diagnostic_precision','float32'))
            for key in totals:totals[key]+=float(loss[key])*len(part)/len(groups)
            probabilities=logits.softmax(-1).cpu().tolist();predicted=raw.cpu().tolist();offset=0
            for g,size in zip(part,sizes,strict=True):
                forecasts=predicted[offset:offset+size];p=probabilities[offset:offset+size]
                rows.append({'group':g['identity'],'input_sha256':g['input_sha256'],'edits':g['edits'],
                    'actual_cash':g['cash'],'predicted_cash':forecasts,'wdl_probabilities':p,
                    'margin_choice':max(range(size),key=lambda i:(forecasts[i][0]-forecasts[i][1],-i)),
                    'wdl_choice':max(range(size),key=lambda i:(p[i][2]+.5*p[i][1],-i))})
                offset+=size
    return {'losses':totals,'groups':rows,'scope':'TRAINING_DATA_DESCRIPTIVE; NOT_PROSPECTIVE_PLAYING_EVIDENCE'}


def sampling_pool(groups,path=None,expected_sha=None):
    """Explicit integer exposure weights; retain every complete witnessed group."""
    if path is None:
        if expected_sha is not None:raise ValueError('sampling digest requires a file')
        return list(range(len(groups))),None
    if expected_sha is None:raise ValueError('sampling file requires its exact digest')
    equal(file_sha(path),expected_sha,'registered group multiplicities')
    spec=read(path)
    equal(spec['schema'],'farm-complete-group-multiplicities-v1','sampling schema')
    records=spec['groups'];keys=[digest(g['identity']) for g in groups]
    if len(records)!=len(keys) or len({digest(r['group']) for r in records})!=len(keys):
        raise ValueError('every group exactly once in multiplicity declaration')
    counts={digest(r['group']):r['count'] for r in records}
    if set(counts)!=set(keys) or any(type(v) is not int or not 1<=v<=16 for v in counts.values()):
        raise ValueError('matching groups with positive bounded integer multiplicities required')
    ordered=[counts[k] for k in keys]
    if sum(ordered)>512:raise ValueError('bounded sampling pool of at most512 entries required')
    pool=[i for i,count in enumerate(ordered) for _ in range(count)]
    return pool,{'schema':spec['schema'],'file_sha256':expected_sha,'multiplicities':ordered,
        'semantics':'SHUFFLE_REPEATED_COMPLETE_GROUP_INDICES_EACH_CYCLE; ALL_ALTERNATIVES_RETAINED; NO_NEW_GAME_CREDIT'}


def fit(args):
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('SLURM GPU allocation required')
    if not 1<=args.updates<=1024 or not 1<=args.microbatch_groups<=args.batch_groups<=32:raise ValueError('bounded update and group batch sizes required')
    if getattr(args,'reset_actor_critic_adam',False) and (args.zero_first_moments or args.effect_objective not in ('wdl_only','wdl_contrast','wdl_contrast_rank')):
        raise ValueError('full Adam reset requires inactive future loss and excludes first-moment-only reset')
    parent=sustain.checkpoint(ROOT,args.parent)
    equal(parent['model_sha256'],args.model_sha256,'parent actor');equal(parent['optimizer_sha256'],args.optimizer_sha256,'parent optimizer')
    output=moe_lab.output_directory(args.output);torch.cuda.set_per_process_memory_fraction(.8)
    parent_dir=ROOT/parent['directory'];parent_plan=sustain.read(parent_dir/'PLAN.json');parent_result=sustain.read(parent_dir/'RESULT.json')
    game=copy.deepcopy(parent_plan['execution'].get('resume_game_execution',parent_plan['execution']))
    transition=program_bc.rebind_source_pins(parent_plan,parent_result,game,allowlist=args.rebind_source_pins,
        reason='Explicit supervised outcome stage; inherited actor/auxiliary tensors, optimizers, RNG, clock and architecture retained.' if args.rebind_source_pins else '')
    learner=campaign.Learner(parent_dir,parent,checkpoint_blocks=True,plan_learning=game['plan_learning'],shop_revision_every=2,full_plan=True)
    equal(learner.execution,game,'unchanged game resume contract')
    learner.execution=campaign.effect_execution(game,args.plan_context,args.effect_coefficient,args.plan_context_gain,args.encoder_readout,args.effect_objective,args.effect_diagnostic_precision)
    original=torch.load(parent_dir/'optimizer.pt',map_location='cpu',weights_only=True)
    entry=restore_state(learner)
    for key,value in (('adam',original['optimizer']),('heads',original['plan_learning_state']['heads']),
        ('head_adam',original['plan_learning_state']['optimizer']),('cpu_rng',original['torch_rng']),('cuda_rng',original['cuda_rng'])):
        equal(entry[key],moe_lab._state_digest(value),'exact entry '+key)
    del original
    if args.deterministic_algorithms:torch.use_deterministic_algorithms(True)
    future_before=future_state(learner) if args.effect_objective in ('wdl_only','wdl_contrast','wdl_contrast_rank') else None
    started=time.monotonic();groups=load_groups(args.data,args.data_sha256,learner.graph,args.plan_context)
    pool,sampling=sampling_pool(groups,args.group_multiplicities,args.group_multiplicities_sha256)
    if learner.age+args.updates>learner.settings['planned_optimizer_steps']:raise ValueError('inherited schedule exhausted')
    plan={'schema':campaign.EFFECT_STAGE,'execution':learner.execution,'source_pins':campaign.sources(),
        'environment':{**learner.profile,'bitsandbytes':str(learner.bnb.__version__)},'parent':parent,
        'start_update':learner.age,'updates':args.updates,'schedule':learner.settings,
        'data_result_sha256':args.data_sha256,'data_directory':str(Path(args.data).relative_to(ROOT)),
        'groups':len(groups),'alternatives':sum(len(g['traces']) for g in groups),'batch_groups':args.batch_groups,
        'microbatch_groups':args.microbatch_groups,'shuffle_seed':args.shuffle_seed,'source_transition':transition,
        'evaluation':'Reserved worlds are not loaded. Both WDL and margin choices must be reported on the prospective panel.',
        'retention':{'purpose':'Learn explicit edit effects in the existing full model','dependencies':[parent['directory'],str(Path(args.data).relative_to(ROOT))],
            'remove_when':'Verified successor/evaluation permits retirement; protected parent remains available.'}}
    if args.deterministic_algorithms:plan['deterministic_algorithms']=True
    if args.zero_first_moments:plan['optimizer_intervention']='ZERO_ACTOR_AND_CRITIC_FIRST_MOMENTS_ONCE; PRESERVE_SECOND_MOMENTS_AGES_AND_FUTURE'
    if getattr(args,'reset_actor_critic_adam',False):plan['optimizer_intervention']='FRESH_ACTOR_AND_CRITIC_ADAM_STATE; PRESERVE_WEIGHTS_GLOBAL_SCHEDULE_AND_FUTURE'
    if sampling is not None:plan['group_sampling']=sampling
    write_new(output/'PLAN.json',plan);sha=file_sha(output/'PLAN.json')
    admission=write_new(output/'ADMISSION.json',{'plan_sha256':sha,'complete_groups':len(groups),'entry_state':entry,
        'data_result_sha256':args.data_sha256,'authored_targets_not_policy_samples':True})
    write_new(output/'BEFORE.json',measure(learner,groups,args.microbatch_groups))
    if args.zero_first_moments:
        counts=dict(actor=zero_first_moments(learner.optimizer,list(learner.graph.parameters())),
                    critic=zero_first_moments(learner.aux_optimizer,list(learner.auxiliary['critic'].parameters())))
        write_new(output/'FIRST-MOMENT-RESET.json',dict(plan_sha256=sha,entry_state=entry,
            reset_state=restore_state(learner),tensors=counts,optimizer_updates=0))
    if getattr(args,'reset_actor_critic_adam',False):
        counts=dict(actor=reset_adam_state(learner.optimizer,list(learner.graph.parameters())),
                    critic=reset_adam_state(learner.aux_optimizer,list(learner.auxiliary['critic'].parameters())))
        write_new(output/'ADAM-RESET.json',dict(plan_sha256=sha,entry_state=entry,
            reset_state=restore_state(learner),tensors=counts,optimizer_updates=0,
            global_age=learner.age,per_tensor_age='Fresh on next actual gradient; unused tensors remain absent.'))
    generator=torch.Generator(device='cpu').manual_seed(args.shuffle_seed);order=[];history=[];presentations=0
    torch.cuda.reset_peak_memory_stats()
    for update in range(args.updates):
        if len(order)<args.batch_groups:order+=[pool[i] for i in torch.randperm(len(pool),generator=generator).tolist()]
        chosen=order[:args.batch_groups];order=order[args.batch_groups:]
        learner.optimizer.zero_grad(set_to_none=True);learner.aux_optimizer.zero_grad(set_to_none=True)
        totals={k:0. for k in ('wdl','cash','effect','total')};tick=time.monotonic()
        if args.effect_objective in ('wdl_contrast','wdl_contrast_rank'):totals['wdl_contrast']=0.
        if args.effect_objective=='wdl_contrast_rank':totals['wdl_rank']=0.
        for start in range(0,len(chosen),args.microbatch_groups):
            part=[groups[i] for i in chosen[start:start+args.microbatch_groups]]
            logits,encoded,sizes=forward(learner,part)
            cash=torch.tensor([row for g in part for row in g['cash']],device=encoded.device)
            losses,_=terms(logits,encoded,cash,sizes,learner.execution['effect_coefficient'],learner.execution.get('effect_objective','joint'),learner.execution.get('effect_diagnostic_precision','float32'));weight=len(part)/len(chosen)
            (losses['total']*weight).backward()
            for key in totals:totals[key]+=float(losses[key].detach())*weight
        norm=torch.nn.utils.clip_grad_norm_(learner.parameters,learner.settings['gradient_clip'],error_if_nonfinite=True)
        lr=learning_rate_at(learner.age+1,learner.settings)
        for optimizer in (learner.optimizer,learner.aux_optimizer):
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step()
        learner.age+=1;presentations+=len(chosen)
        row={'age':learner.age,'groups':chosen,'losses':totals,'pre_clip_gradient_norm':float(norm),'lr':lr,'seconds':time.monotonic()-tick}
        with (output/'history.jsonl').open('a') as stream:stream.write(canonical(row)+'\n');stream.flush();os.fsync(stream.fileno())
        history.append(row)
        if (update+1)%16==0:print(canonical({'updates':update+1,'age':learner.age,'losses':totals,'seconds':time.monotonic()-started}),flush=True)
    after=restore_state(learner)
    if after['actor']==entry['actor'] or after['heads']==entry['heads']:raise ValueError('supervised stage must change actor and heads')
    if future_before is not None:equal(future_state(learner),future_before,'unused future head and inherited Adam unchanged')
    write_new(output/'AFTER.json',measure(learner,groups,args.microbatch_groups))
    receipt=write_new(output/'RECEIPT.json',{'plan_sha256':sha,'optimizer_steps':args.updates,'global_optimizer_steps_after':learner.age,
        'entry_state':entry,'ending_state':after,'state_group_presentations':presentations,
        'authored_alternative_presentations':sum(len(groups[i]['traces']) for row in history for i in row['groups']),
        'auxiliary_updates':args.updates,'fresh_physical_games':0,'policy_samples':0,'seconds':time.monotonic()-started,
        'maximum_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'shuffle_state':generator.get_state().tolist(),'pending_order':order,
        **({'unused_future_state':future_before,'unused_future_state_unchanged':True} if future_before is not None else {})})
    manifest=learner.publish(output,receipt,sha,admission,[])
    result={'schema':campaign.EFFECT_STAGE,'status':sustain.EFFECT_STATUS,'plan_sha256':sha,
        'source_pins':campaign.sources(),'global_optimizer_updates':learner.age,'optimizer_updates':args.updates,
        'before_model_sha256':parent['model_sha256'],'after_model_sha256':file_sha(output/'model.pt'),
        'optimizer_sha256':file_sha(output/'optimizer.pt'),'manifest':manifest['content_sha256'],
        'receipt_sha256':file_sha(output/'RECEIPT.json'),'new_physical_games':0,'new_policy_samples':0}
    write_new(output/'RESULT.json',result)
    equal(sustain.checkpoint(ROOT,output)['model_sha256'],result['after_model_sha256'],'published checkpoint')
    print(canonical(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True);parser.add_argument('--model-sha256',required=True)
    parser.add_argument('--optimizer-sha256',required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--data-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=128);parser.add_argument('--batch-groups',type=int,default=8)
    parser.add_argument('--microbatch-groups',type=int,default=2);parser.add_argument('--shuffle-seed',type=int,required=True)
    parser.add_argument('--group-multiplicities',type=Path)
    parser.add_argument('--group-multiplicities-sha256')
    parser.add_argument('--rebind-source-pins',nargs='*',default=[])
    parser.add_argument('--plan-context',choices=('raw_factor_prefix','published_plan_encoder'),default='raw_factor_prefix')
    parser.add_argument('--effect-coefficient',type=float)
    parser.add_argument('--effect-objective',choices=('joint','wdl_only','wdl_contrast','wdl_contrast_rank'),default='joint')
    parser.add_argument('--effect-diagnostic-precision',choices=('float32','float64'),default='float32')
    parser.add_argument('--plan-context-gain',type=float,default=1.)
    parser.add_argument('--encoder-readout',choices=('mean','kind_mean'),default='mean')
    parser.add_argument('--deterministic-algorithms',action='store_true')
    reset=parser.add_mutually_exclusive_group()
    reset.add_argument('--zero-first-moments',action='store_true')
    reset.add_argument('--reset-actor-critic-adam',action='store_true')
    parser.add_argument('--verify-restore',action='store_true')
    args=parser.parse_args();args.data=args.data.absolute();args.output=args.output.absolute()
    if args.verify_restore:return restore_verification(args)
    try:fit(args)
    except Exception as error:
        if args.output.is_dir() and not (args.output/'RUN-FAILURE.json').exists():
            write_new(args.output/'RUN-FAILURE.json',{'status':'FAILED_NO_PROMOTION','error':repr(error),
                'parent':args.parent,'parent_model_sha256':args.model_sha256,'preserve_partial_outputs':True})
        raise


if __name__=='__main__':main()
