"""Source-versioned many-game index over existing full-economic B32 cohorts.

Owner: learning maintainer. Data plumbing, not a collector, fitter or admission.
Production defaults to2048 fresh games;16 worlds/32 seats is only a microbatch.
Each shard is produced by the maintained CPU witness, retains its exact episode
objects and is validated by the unchanged cohort/trace/feature contracts. No
neural verification, optimizer, model loading, automatic retry or deletion here.
Completed chunks can be reused; an interrupted chunk remains explicitly blocked.
The reader holds at most one16-world shard internally. Callers must not list() it.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import copy
from pathlib import Path
from types import FunctionType

ROOT=Path(__file__).resolve().parents[3]
SCHEMA='farm-full-economic-many-game-buffer-v1'
DEFAULT_GAMES=2048
# Changing this pin list changes the source identity of every future checkpoint; intended.
OWN=('src/farm_neat/rollout/buffer.py','tests/contracts/test_buffer.py')
_IMPORTED_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BEHAVIOR_KEYS=('model_sha256','model_manifest_sha256','model_optimizer_age','rollout_iteration',
    'architecture_sha256','numerics','temperature','command_budget','source_pins','source_sha256',
    'environment','neural_batch','cublas_workspace_config','decision_steps','physical_steps_per_world')
MEMBERSHIP_KEYS=('cohort_id','seeds','actor_rng_seeds','worlds','lanes','lane_order','content_sha256')
LIMITS={'compressed_shard_bytes':128_000_000,'uncompressed_shard_bytes':768_000_000,
        'json_line_bytes':64_000_000}


def _json(raw):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value: raise ValueError('duplicate JSON key')
            value[key]=item
        return value
    def constant(value): raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def _sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def _path(root,name):
    root=Path(root).absolute()
    if (type(name) is not str or not name or Path(name).is_absolute()
            or any(part in ('','.','..') for part in name.split('/'))):
        raise ValueError('canonical relative path required')
    path=root/name
    if root!=root.resolve() or path!=path.resolve() or not path.is_relative_to(root/'work/runs'):
        raise ValueError('nonsymlink data paths below workspace work/runs required')
    return path


def _folder(root,folder):
    path=Path(folder).absolute()
    try: name=path.relative_to(Path(root).absolute()).as_posix()
    except ValueError: raise ValueError('buffer must remain in its workspace') from None
    return _path(root,name)


def _runtime(rollout_module=None):
    if rollout_module is not None: return rollout_module
    from farm_neat.rollout import batched_rollout as full_economic_batched_rollout
    return full_economic_batched_rollout


def _identity(r):
    if _sha(ROOT/OWN[0])!=_IMPORTED_SHA: raise ValueError('loaded buffer source changed')
    pins={**r.sources(),**{name:_sha(ROOT/name) for name in OWN}}
    for name,value in pins.items():
        p=ROOT/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or p!=p.resolve()
                or _sha(p)!=value): raise ValueError('buffer/producer source changed')
    hooks={}
    for name in ('validate_plan','witness','cohort_groups','sources'):
        fn=getattr(r,name)
        if not isinstance(fn,FunctionType): raise ValueError('source-defined rollout hook required')
        relative=Path(fn.__code__.co_filename).resolve().relative_to(ROOT).as_posix()
        if relative not in pins: raise ValueError('rollout hook missing from declared source closure')
        hooks[name]={'file':relative,'sha256':pins[relative],'qualname':fn.__qualname__}
    r.attest(r.sources())
    return r.bound({'schema':SCHEMA+'-execution','rollout_schema':r.SCHEMA,
        'source_pins':pins,'hooks':hooks,'neural_verification':False,'training_admission':False})


def _read(path,expected=None,limit=8_000_000):
    if not path.is_file() or path!=path.resolve(): raise ValueError('regular nonsymlink file required')
    if path.stat().st_size>limit: raise ValueError('bounded metadata size exceeded')
    data=path.read_bytes(); actual=hashlib.sha256(data).hexdigest()
    if expected is not None and actual!=expected: raise ValueError('externally pinned file changed')
    return _json(data),actual


def _write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'),allow_nan=False);stream.write('\n')


def make_plan(members,*,root=ROOT,target_games=DEFAULT_GAMES,mode='PRODUCTION',rollout_module=None):
    """Register immutable existing microplans and as-yet unexposed producer paths.

    Each member is {plan_path, plan_file_sha256, producer_path}, workspace-relative.
    TEST_FIXTURE permits smaller buffers but can never claim production readiness.
    """
    r=_runtime(rollout_module)
    if (mode not in ('PRODUCTION','TEST_FIXTURE') or type(target_games) is not int
            or target_games%16 or not 16<=target_games<=1_000_000
            or (mode=='PRODUCTION' and target_games<DEFAULT_GAMES)):
        raise ValueError('production >=2048 games in complete16-world microcohorts required')
    if not isinstance(members,list) or len(members)!=target_games//16:
        raise ValueError('all immutable microcohort memberships required')
    frozen=[]; seeds=set(); samplers=set(); ids=set(); paths=set(); behavior=None
    for ordinal,member in enumerate(members):
        if not isinstance(member,dict) or set(member)!={'plan_path','plan_file_sha256','producer_path'}:
            raise ValueError('exact microcohort member fields required')
        path=_path(root,member['plan_path']); producer=_path(root,member['producer_path'])
        r._sha(member['plan_file_sha256'])
        plan,_=_read(path,member['plan_file_sha256']);r.validate_plan(plan)
        if plan['worlds']!=16 or plan['lanes']!=32: raise ValueError('B32 microbatch is16 complete worlds')
        if any(key not in plan for key in BEHAVIOR_KEYS): raise ValueError('complete behavior identity required')
        # Include variant-specific identities too (e.g. monetary prior and actual
        # old/new checkpoint schema), not merely the common original fields.
        current={key:value for key,value in plan.items() if key not in MEMBERSHIP_KEYS}
        if behavior is None: behavior=current
        r._equal(current,behavior,'single frozen rollout behavior')
        if seeds.intersection(plan['seeds']) or samplers.intersection(plan['actor_rng_seeds']):
            raise ValueError('physical seeds and sampler lanes must be globally distinct')
        if plan['cohort_id'] in ids or str(path) in paths or str(producer) in paths:
            raise ValueError('distinct immutable cohort and path membership required')
        for previous in frozen:
            prior=_path(root,previous['producer_path'])
            if producer.is_relative_to(prior) or prior.is_relative_to(producer):
                raise ValueError('producer namespaces cannot overlap')
        seeds.update(plan['seeds']);samplers.update(plan['actor_rng_seeds']);ids.add(plan['cohort_id'])
        paths.update((str(path),str(producer)))
        frozen.append({**member,'ordinal':ordinal,'plan_content_sha256':plan['content_sha256'],
            'cohort_id':plan['cohort_id'],'seeds':plan['seeds'],'actor_rng_seeds':plan['actor_rng_seeds']})
    return r.bound({'schema':SCHEMA+'-index','mode':mode,'target_games':target_games,
        'microcohort_games':16,'inference_microbatch_seats':32,'microcohorts':len(frozen),
        'seat_episodes':target_games*2,'model_decisions':target_games*60,
        'physical_joint_turns':target_games*719,'behavior':behavior,'members':frozen,
        'execution':_identity(r),'limits':dict(LIMITS),'training_admission':False})


def _validate(plan,root,r):
    r._sealed(plan)
    members=[{k:m[k] for k in ('plan_path','plan_file_sha256','producer_path')} for m in plan['members']]
    expected=make_plan(members,root=root,target_games=plan['target_games'],mode=plan['mode'],rollout_module=r)
    r._equal(plan,expected,'immutable many-game index/source/membership')


def create_index(folder,plan,*,root=ROOT,rollout_module=None):
    r=_runtime(rollout_module);folder=_folder(root,folder);_validate(plan,root,r)
    target=folder/'ROLLOUT-INDEX.json'
    if folder.exists():
        existing,sha=_read(target);r._equal(existing,plan,'idempotent index creation')
        return {'index_file_sha256':sha,'already_existed':True}
    for member in plan['members']:
        producer=_path(root,member['producer_path'])
        if producer.exists(): raise ValueError('index must precede every producer exposure')
        if folder.is_relative_to(producer) or producer.is_relative_to(folder):
            raise ValueError('buffer metadata and producer namespaces must be separate')
    folder.mkdir(parents=True);_write(target,plan)
    return {'index_file_sha256':_sha(target),'already_existed':False}


def _open(folder,expected_index_sha256,root,r):
    folder=_folder(root,folder);r._sha(expected_index_sha256)
    plan,_=_read(folder/'ROLLOUT-INDEX.json',expected_index_sha256);_validate(plan,root,r)
    chunks=folder/'chunks'
    if chunks.exists():
        expected={f'{i:06d}' for i in range(plan['microcohorts'])}
        if chunks!=chunks.resolve() or any(p.name not in expected or p!=p.resolve() for p in chunks.iterdir()):
            raise ValueError('unexpected/symlink chunk namespace')
    return folder,plan


def _chunk(folder,ordinal,plan):
    if type(ordinal) is not int or not 0<=ordinal<plan['microcohorts']:
        raise ValueError('registered chunk ordinal required')
    return folder/'chunks'/f'{ordinal:06d}'


def _receipt(folder,plan,index_sha,ordinal,r):
    chunk=_chunk(folder,ordinal,plan)
    receipt,receipt_sha=_read(chunk/'COMPLETE.json');r._sealed(receipt)
    for key,value in (('schema',SCHEMA+'-chunk'),('index_file_sha256',index_sha),('ordinal',ordinal),
        ('member',plan['members'][ordinal]),('execution_sha256',plan['execution']['content_sha256']),
        ('physical_games',16),('seat_episodes',32),('model_decisions',960),('training_admission',False)):
        r._equal(receipt[key],value,'completed shard '+key)
    raw=chunk/'SHARD.jsonl.gz'
    if (not raw.is_file() or raw!=raw.resolve() or raw.stat().st_size!=receipt['compressed_bytes']
            or not 0<receipt['compressed_bytes']<=plan['limits']['compressed_shard_bytes']
            or _sha(raw)!=receipt['shard_file_sha256']): raise ValueError('exact bounded shard bytes required')
    return receipt,receipt_sha


def status(folder,expected_index_sha256,*,root=ROOT,rollout_module=None):
    """Read-only status; never launch/retry a producer or witness."""
    r=_runtime(rollout_module);folder,plan=_open(folder,expected_index_sha256,root,r)
    states=[];completed=0
    for ordinal,member in enumerate(plan['members']):
        chunk=_chunk(folder,ordinal,plan);producer=_path(root,member['producer_path'])
        if (chunk/'COMPLETE.json').exists():
            receipt,_=_receipt(folder,plan,expected_index_sha256,ordinal,r)
            state='COMPLETED_REUSABLE';completed+=1
        elif chunk.exists(): state='MATERIALIZATION_INCOMPLETE_NO_AUTORETRY'
        elif (producer/'FAILURE.json').exists(): state='PRODUCER_FAILED_NO_REROLL'
        elif (producer/'RESULT.json').exists(): state='PRODUCER_CLOSED_CPU_WITNESS_PENDING'
        elif producer.exists(): state='PRODUCER_INCOMPLETE_NO_REROLL'
        else: state='UNSTARTED'
        states.append({'ordinal':ordinal,'status':state})
    return r.bound({'schema':SCHEMA+'-status','index_file_sha256':expected_index_sha256,
        'status':'ALL_SHARDS_PRESENT_NOT_TRAINING_ADMITTED' if completed==len(states) else 'INCOMPLETE',
        'mode':plan['mode'],'target_games':plan['target_games'],'completed_chunks':completed,
        'completed_physical_games':completed*16,'completed_seat_episodes':completed*32,
        'completed_model_decisions':completed*960,'optimizer_updates':0,'chunks':states,
        'training_admission':False,'liveness_claim':False})


def _checked(checked,expected,r):
    if not isinstance(checked,dict) or set(checked)!={'plan','episodes','engine_witness'}:
        raise ValueError('actual original witness result required')
    r._equal(checked['plan'],expected,'witness original microplan');r.validate_plan(expected)
    episodes,witness=checked['episodes'],checked['engine_witness'];r._sealed(witness)
    groups=r.cohort_groups(expected,episodes)
    for key,value in (('schema',r.SCHEMA+'-engine-witness'),('plan_sha256',expected['content_sha256']),
        ('training_batch_sha256',r.digest({'episodes':episodes})),
        ('cohorts_sha256',r.digest([g['batch'] for g in groups])),('reconstructed_physical_games',16),
        ('physical_steps',16*719),('decision_calls',960),('responsive_reference_games',0),
        ('method','FRESH_ENGINE_RECORDED_PROGRAMS_EXACT_EXECUTOR_INPUTS_ACTIONS_SUCCESSORS'),
        ('terminal_training_admission',False)):
        r._equal(witness[key],value,'original witness '+key)
    if len(witness['terminal_cash'])!=16 or len(witness['world_witnesses'])!=16:
        raise ValueError('all16 independent reconstruction witnesses required')
    for world,(cash,proof) in enumerate(zip(witness['terminal_cash'],witness['world_witnesses'],strict=True)):
        if not isinstance(cash,list) or len(cash)!=2 or any(r.base._finite(v)<0 for v in cash):
            raise ValueError('actual finite terminal cash pair required')
        r._sealed(proof)
        for key,value in (('schema',r.SCHEMA+'-engine-witness'),
            ('plan_sha256',r.world_plan(expected,world)['content_sha256']),
            ('training_batch_sha256',r.digest({'episodes':episodes[2*world:2*world+2]})),
            ('reconstructed_physical_games',1),('physical_steps',719),('decision_calls',60),
            ('responsive_reference_games',0),('terminal_training_admission',False),
            ('method','FRESH_ENGINE_RECORDED_PROGRAMS_EXACT_EXECUTOR_INPUTS_ACTIONS_SUCCESSORS'),
            ('terminal_cash',cash)):
            r._equal(proof[key],value,'per-world witness '+key)
        delta=cash[0]-cash[1];reward=int(delta>0)-int(delta<0)
        r._equal([episodes[2*world+s]['terminal_reward'] for s in (0,1)],[reward,-reward],'fresh WDL')
    return groups


def _write_shard(path,checked,limits):
    total=0
    encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False)
    class Sink:
        def __init__(self,stream): self.stream=stream;self.bytes=0
        def write(self,value):
            if self.bytes+len(value)>limits['compressed_shard_bytes']: raise ValueError('compressed shard cap')
            self.bytes+=len(value);return self.stream.write(value)
        def flush(self): self.stream.flush()
    with path.open('xb') as raw, gzip.GzipFile(filename='',fileobj=Sink(raw),mode='wb',mtime=0) as stream:
        header={'schema':SCHEMA+'-shard','plan':checked['plan'],'engine_witness':checked['engine_witness']}
        for record in (header,*checked['episodes']):
            line_bytes=0
            for piece in encoder.iterencode(record):
                data=piece.encode();line_bytes+=len(data);total+=len(data)
                if line_bytes>limits['json_line_bytes'] or total>limits['uncompressed_shard_bytes']:
                    raise ValueError('decoded shard/record cap')
                stream.write(data)
            if line_bytes+1>limits['json_line_bytes'] or total+1>limits['uncompressed_shard_bytes']:
                raise ValueError('decoded shard/record cap')
            stream.write(b'\n');total+=1
    return total


def _decode(path,limits):
    total=0
    with gzip.open(path,'rb') as stream:
        def one():
            nonlocal total
            line=stream.readline(limits['json_line_bytes']+1);total+=len(line)
            if (not line.endswith(b'\n') or len(line)>limits['json_line_bytes']
                    or total>limits['uncompressed_shard_bytes']): raise ValueError('truncated/oversized shard record')
            return _json(line)
        header=one()
        if set(header)!={'schema','plan','engine_witness'} or header['schema']!=SCHEMA+'-shard':
            raise ValueError('exact shard header required')
        episodes=[one() for _ in range(32)]
        if stream.read(1): raise ValueError('extra shard episode or suffix')
    return {'plan':header['plan'],'engine_witness':header['engine_witness'],'episodes':episodes},total


def _stored(checked,size,receipt,r):
    r._equal(size,receipt['uncompressed_bytes'],'decoded length')
    witness=checked['engine_witness']
    for field,target in (('content_sha256','engine_witness_sha256'),
            ('training_batch_sha256','training_batch_sha256'),
            ('producer_plan_file_sha256','producer_plan_file_sha256'),
            ('producer_result_file_sha256','producer_result_file_sha256')):
        r._equal(witness[field],receipt[target],'stored witness '+field)


def materialize_chunk(folder,ordinal,expected_index_sha256,*,root=ROOT,rollout_module=None):
    """Run the existing CPU witness once, writing one bounded lossless shard.

    It does NOT collect a game or verify a neural graph. A partial directory is
    preserved/refused; owner reconciliation, not an automatic reroll, is needed.
    """
    r=_runtime(rollout_module);folder,plan=_open(folder,expected_index_sha256,root,r)
    chunk=_chunk(folder,ordinal,plan);member=plan['members'][ordinal]
    if (chunk/'COMPLETE.json').exists():
        receipt,_=_receipt(folder,plan,expected_index_sha256,ordinal,r)
        checked,size=_decode(chunk/'SHARD.jsonl.gz',plan['limits'])
        expected,_=_read(_path(root,member['plan_path']),member['plan_file_sha256'])
        _checked(checked,expected,r)
        _stored(checked,size,receipt,r)
        _receipt(folder,plan,expected_index_sha256,ordinal,r)
        return receipt
    if chunk.exists(): raise ValueError('interrupted materialization preserved; no automatic retry')
    producer=_path(root,member['producer_path'])
    if (producer/'FAILURE.json').exists() or not (producer/'RESULT.json').is_file():
        raise ValueError('complete nonfailed producer required; no reroll performed')
    expected,_=_read(_path(root,member['plan_path']),member['plan_file_sha256'])
    producer_plan,producer_sha=_read(producer/'PLAN.json');r._equal(producer_plan,expected,'producer plan')
    producer_result,result_sha=_read(producer/'RESULT.json');r._sealed(producer_result)
    chunk.parent.mkdir(exist_ok=True);chunk.mkdir()
    _write(chunk/'STARTED.json',{'index_file_sha256':expected_index_sha256,'ordinal':ordinal})
    checked=r.witness(producer,expected_plan_file_sha256=producer_sha)
    _checked(checked,expected,r)
    for key,value in (('producer_plan_file_sha256',producer_sha),
            ('producer_result_file_sha256',result_sha),('producer_result_sha256',producer_result['content_sha256'])):
        r._equal(checked['engine_witness'][key],value,'unchanged producer '+key)
    size=_write_shard(chunk/'SHARD.jsonl.gz',checked,plan['limits'])
    witness=checked['engine_witness']
    receipt=r.bound({'schema':SCHEMA+'-chunk','index_file_sha256':expected_index_sha256,
        'ordinal':ordinal,'member':member,'execution_sha256':plan['execution']['content_sha256'],
        'shard_file_sha256':_sha(chunk/'SHARD.jsonl.gz'),'compressed_bytes':(chunk/'SHARD.jsonl.gz').stat().st_size,
        'uncompressed_bytes':size,'training_batch_sha256':witness['training_batch_sha256'],
        'engine_witness_sha256':witness['content_sha256'],'producer_plan_file_sha256':producer_sha,
        'producer_result_file_sha256':result_sha,
        'physical_games':16,'seat_episodes':32,'model_decisions':960,
        'neural_verification':False,'training_admission':False})
    del checked
    reopened,decoded=_decode(chunk/'SHARD.jsonl.gz',plan['limits']);_checked(reopened,expected,r)
    _stored(reopened,decoded,receipt,r)
    if _sha(chunk/'SHARD.jsonl.gz')!=receipt['shard_file_sha256']:
        raise ValueError('shard changed during readback')
    _read(producer/'PLAN.json',producer_sha);_read(producer/'RESULT.json',result_sha)
    if (producer/'FAILURE.json').exists(): raise ValueError('producer failure appeared during materialization')
    _validate(plan,root,r);_write(chunk/'COMPLETE.json',receipt)
    return receipt


def finalize(folder,expected_index_sha256,*,root=ROOT,rollout_module=None):
    """Seal hash membership; does not turn CPU witnesses into PPO admission."""
    r=_runtime(rollout_module);folder,plan=_open(folder,expected_index_sha256,root,r)
    members=[]
    for ordinal in range(plan['microcohorts']):
        receipt,sha=_receipt(folder,plan,expected_index_sha256,ordinal,r)
        members.append({'ordinal':ordinal,'receipt_file_sha256':sha,'shard_file_sha256':receipt['shard_file_sha256']})
    result=r.bound({'schema':SCHEMA+'-complete','index_file_sha256':expected_index_sha256,
        'mode':plan['mode'],'physical_games':plan['target_games'],'seat_episodes':plan['seat_episodes'],
        'model_decisions':plan['model_decisions'],'microcohorts':plan['microcohorts'],'members':members,
        'production_size_ready':plan['mode']=='PRODUCTION','neural_verification':False,'training_admission':False})
    target=folder/'ROLLOUT-COMPLETE.json'
    if target.exists(): existing,_=_read(target);r._equal(existing,result,'immutable completion')
    else: _write(target,result)
    return {'completion_file_sha256':_sha(target),'receipt':result}


def _completion(folder,plan,expected_index_sha256,expected_completion_sha256,r):
    r._sha(expected_completion_sha256)
    complete,_=_read(folder/'ROLLOUT-COMPLETE.json',expected_completion_sha256);r._sealed(complete)
    for key,value in (('schema',SCHEMA+'-complete'),('index_file_sha256',expected_index_sha256),
            ('mode',plan['mode']),('physical_games',plan['target_games']),('seat_episodes',plan['seat_episodes']),
            ('model_decisions',plan['model_decisions']),('microcohorts',plan['microcohorts']),
            ('production_size_ready',plan['mode']=='PRODUCTION'),
            ('neural_verification',False),('training_admission',False)):
        r._equal(complete[key],value,'closed buffer '+key)
    if len(complete['members'])!=plan['microcohorts']: raise ValueError('all closed shards required')
    for ordinal,member in enumerate(complete['members']):
        if set(member)!={'ordinal','receipt_file_sha256','shard_file_sha256'}:
            raise ValueError('exact closed shard membership required')
        r._equal(member['ordinal'],ordinal,'closed ordinal')
        r._sha(member['receipt_file_sha256']);r._sha(member['shard_file_sha256'])
    return complete


def _load_shard(folder,plan,index_sha,member,root,r):
    ordinal=member['ordinal']
    receipt,sha=_receipt(folder,plan,index_sha,ordinal,r)
    r._equal(sha,member['receipt_file_sha256'],'externally closed shard receipt')
    r._equal(receipt['shard_file_sha256'],member['shard_file_sha256'],'closed shard hash')
    chunk=_chunk(folder,ordinal,plan);data,decoded=_decode(chunk/'SHARD.jsonl.gz',plan['limits'])
    registered=plan['members'][ordinal]
    expected,_=_read(_path(root,registered['plan_path']),registered['plan_file_sha256'])
    groups=_checked(data,expected,r)
    _stored(data,decoded,receipt,r)
    if _sha(chunk/'SHARD.jsonl.gz')!=member['shard_file_sha256']:
        raise ValueError('shard changed during decoding')
    return receipt,data,groups


def iter_shards(folder,expected_index_sha256,expected_completion_sha256,*,root=ROOT,rollout_module=None):
    """Yield one checked16-world shard; never concatenate a whole rollout.

    Exhaustion rechecks all visited bytes. Abandoning iteration cannot establish
    complete consumption. The caller must keep memory bounded and perform actual
    parent neural verification/global normalization before any future PPO update.
    """
    r=_runtime(rollout_module);folder,plan=_open(folder,expected_index_sha256,root,r)
    complete=_completion(folder,plan,expected_index_sha256,expected_completion_sha256,r)
    watched=[]
    for ordinal,member in enumerate(complete['members']):
        receipt,data,groups=_load_shard(folder,plan,expected_index_sha256,member,root,r)
        # Groups reference the same episode objects; never retain a prior shard.
        del groups
        chunk=_chunk(folder,ordinal,plan)
        watched.extend(((chunk/'COMPLETE.json',member['receipt_file_sha256']),
                        (chunk/'SHARD.jsonl.gz',member['shard_file_sha256'])))
        yield {'ordinal':ordinal,'receipt':receipt,'checked':data,'training_admission':False}
        del data
    for path,sha in watched:
        if _sha(path)!=sha: raise ValueError('earlier shard changed during iteration')
    _open(folder,expected_index_sha256,root,r)
    if _sha(folder/'ROLLOUT-COMPLETE.json')!=expected_completion_sha256:
        raise ValueError('completion changed during iteration')


def _reader_identity(plan,index_sha,completion_sha,r):
    return r.bound({'schema':SCHEMA+'-reader','index_file_sha256':index_sha,
        'completion_file_sha256':completion_sha,'execution_sha256':plan['execution']['content_sha256'],
        'behavior':copy.deepcopy(plan['behavior']),'physical_games':plan['target_games'],
        'groups':plan['microcohorts']*30,'samples':plan['microcohorts']*30*32,
        'group_order':'CHUNK_DAY_LANE','neural_verification':False,'training_admission':False})


class IndexedReader:
    """One-shard cached random day-group access for a future streaming fitter.

    ``group(i)`` returns a private copy of the unchanged ``cohort_groups`` row
    (calls, returns, batch). Global order is chunk/day/lane. ``vectors()`` makes
    three FP64 arrays and stable (episode_id,step) IDs, NOT a full rollout of
    actor inputs. ``verify()`` must be called before committing a consumer result.
    Neither construction nor successful reads constitute neural/PPO admission.
    """
    def __init__(self,folder,expected_index_sha256,expected_completion_sha256,*,root=ROOT,rollout_module=None):
        self.root,self.r=root,_runtime(rollout_module)
        self.folder,self.plan=_open(folder,expected_index_sha256,root,self.r)
        self.index_sha,self.completion_sha=expected_index_sha256,expected_completion_sha256
        self.complete=_completion(self.folder,self.plan,self.index_sha,self.completion_sha,self.r)
        self.group_count=self.plan['microcohorts']*30
        self.sample_count=self.group_count*32
        self.identity=_reader_identity(self.plan,self.index_sha,self.completion_sha,self.r)
        self._ordinal=None;self._groups=None

    def clear_cache(self):
        self._ordinal=None;self._groups=None

    def group(self,index):
        if type(index) is not int or not 0<=index<self.group_count:
            raise ValueError('registered global day-group index required')
        ordinal,day=divmod(index,30)
        if self._ordinal!=ordinal:
            self.clear_cache()  # Drop the previous shard before decoding its successor.
            _,_,groups=_load_shard(self.folder,self.plan,self.index_sha,
                self.complete['members'][ordinal],self.root,self.r)
            self._groups=groups;self._ordinal=ordinal
        return copy.deepcopy(self._groups[day])

    def vectors(self):
        import numpy as np
        self.clear_cache()
        logp=np.empty(self.sample_count,dtype=np.float64)
        values=np.empty_like(logp);returns=np.empty_like(logp);ids=[];offset=0
        for shard in iter_shards(self.folder,self.index_sha,self.completion_sha,
                root=self.root,rollout_module=self.r):
            episodes=shard['checked']['episodes']
            for day,step in enumerate(self.r.STEPS):
                for episode in episodes:
                    call=episode['calls'][day]
                    logp[offset]=call['old_log_probability'];values[offset]=call['old_value']
                    returns[offset]=episode['terminal_reward'];ids.append((episode['episode_id'],step))
                    offset+=1
            del episodes,shard
        if offset!=self.sample_count or len(set(ids))!=offset:
            raise ValueError('complete unique global sample identities required')
        for vector in (logp,values,returns): vector.flags.writeable=False
        return {'old_logp':logp,'old_value':values,'returns':returns,'sample_ids':tuple(ids),
            'groups':self.group_count,'lanes':32,'identity':copy.deepcopy(self.identity),
            'training_admission':False}

    def verify(self):
        """Recheck immutable source/index/completion and every referenced shard."""
        self.clear_cache()
        _,plan=_open(self.folder,self.index_sha,self.root,self.r)
        complete=_completion(self.folder,plan,self.index_sha,self.completion_sha,self.r)
        expected=_reader_identity(plan,self.index_sha,self.completion_sha,self.r)
        for actual,value,label in ((self.plan,plan,'plan'),(self.complete,complete,'completion'),
                (self.identity,expected,'identity'),(self.group_count,expected['groups'],'group count'),
                (self.sample_count,expected['samples'],'sample count')):
            self.r._equal(actual,value,'in-memory reader '+label)
        for member in complete['members']:
            receipt,sha=_receipt(self.folder,plan,self.index_sha,member['ordinal'],self.r)
            self.r._equal(sha,member['receipt_file_sha256'],'final shard receipt')
            self.r._equal(receipt['shard_file_sha256'],member['shard_file_sha256'],'final shard bytes')
        return expected
