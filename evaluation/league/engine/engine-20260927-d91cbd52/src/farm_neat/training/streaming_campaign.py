"""Owner-only many-game B32 -> indexed witness -> streaming PPO connection.

Owner: full-economic maintainer. Original numerical profile only. This adapter
reuses the existing collector, fresh CPU reconstruction, neural resampling,
optimizer core and model serializer. No new game loop, trainer objective, backend,
automatic cleanup, schedule extension, promotion or upload. Use the storage guard.

prepare freezes ONE fresh rollout allocation (default2048 worlds); SMOKE is
explicitly smaller and not production size. run can reuse completed producer/
witness shards, never reroll an interrupted one. Fit publishes only a closed
allocation; next prepare must use the changed child and fresh worlds. Same-buffer
partial optimizer cursor continuation is deliberately not exposed here. An
interrupted fit remains failed/uncommitted, not automatically restarted.

Seed non-exposure is an explicit trusted owner's responsibility, not something
hashes or a local boolean authenticate. Admission is executed from owner-collected
data; imported/rehashed witness objects are not accepted as a replacement.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""): sys.path.insert(0, str(ROOT / "src"))

from farm_neat.training import sustain as io
from farm_neat.training import batched_training as base
from farm_neat.rollout import buffer as buffer_api
from farm_neat.training import streaming_ppo as core

SCHEMA = "farm-full-economic-streaming-campaign-v1"
OPTIMIZER_SCHEMA = SCHEMA+"-optimizer"
EXECUTION = {"schema": SCHEMA+"-execution", "numerical_backend": "UNCHANGED_ORIGINAL_B32",
    "streaming": core.EXECUTION, "gradient_accumulation": "REGISTERED_PER_ROLLOUT_PROTOCOL",
    "published_buffer": "CLOSED_NEXT_CHILD_REQUIRES_FRESH_WORLDS", "automatic_cleanup": False}
# Changing this pin list changes the source identity of every future checkpoint; intended.
OWN = ("src/farm_neat/training/streaming_campaign.py", "tests/contracts/test_streaming_campaign.py")
IMPORTED_SHA = io.sha(__file__)


def sources():
    if io.sha(__file__) != IMPORTED_SHA: raise ValueError("loaded campaign source changed")
    names = (*OWN, "src/farm_neat/training/sustain.py", "tests/contracts/test_sustain.py")
    return {**core.identity()["source_pins"], **{n: io.sha(ROOT/n) for n in names}}


def attest(pins):
    if pins != sources(): raise ValueError("explicit streaming campaign source closure changed")
    io.attest(ROOT, pins); base.attest(base.sources())


def _path(value):
    return io.path(ROOT, value, output=True)


def _parent(folder, *, tensors=True):
    folder = _path(folder)
    item = io.checkpoint(ROOT, folder, tensors=tensors)
    item["manifest_file_sha256"] = io.sha(folder/"model.pt.json")
    return item


def _protocol(value):
    core._sealed(value)
    expected=core.make_protocol(**{k:value[k] for k in
        ("accumulation_cohorts","optimizer_updates","metric_every_updates","metric_cohorts")})
    if value != expected: raise ValueError("exact registered streaming protocol required before worlds")


def prepare(output, *, parent, model_sha256, optimizer_sha256, owner, seeds,
            actor_rng_seeds, exclusive_seeds=False, mode="PRODUCTION", protocol=None,
            command_budget=8, stage_seconds=3600):
    """Metadata only; explicit owner allocation, never world execution."""
    output, parent = _path(output), _path(parent)
    if output.exists(): raise ValueError("new run directory required; never overwrite a plan")
    if (exclusive_seeds is not True or not isinstance(owner,str) or not owner.strip()
            or mode not in ("PRODUCTION","SMOKE") or type(stage_seconds) is not int or stage_seconds < 1):
        raise ValueError("explicit exclusive owner/mode/stage budget required")
    games = len(seeds)
    if games % 16 or not 16 <= games <= 1_000_000 or (mode == "PRODUCTION" and games < 2048):
        raise ValueError("2048+ production worlds or explicit complete16-world SMOKE required")
    if len(actor_rng_seeds) != games*2: raise ValueError("two unique private samplers per world required")
    current = _parent(parent)
    if (current["model_sha256"] != model_sha256 or current["optimizer_sha256"] != optimizer_sha256):
        raise ValueError("external exact parent model/Adam hashes required")
    remaining = current["schedule"]["planned_optimizer_steps"]-current["optimizer_age"]
    group_count = games//16*30
    if protocol is None:
        protocol = core.make_protocol(accumulation_cohorts=8,
            optimizer_updates=(group_count+7)//8, metric_every_updates=8,
            metric_cohorts=sorted(set((0,group_count//2,group_count-1))))
    _protocol(protocol)
    if not 0 < protocol["optimizer_updates"] <= remaining:
        raise ValueError("declared allocation must fit inherited schedule; no extension")
    if max(protocol["metric_cohorts"]) >= group_count: raise ValueError("metric cohort outside allocation")
    manifest = base.policy.read_manifest(parent/"model.pt")
    environment = dict(io.read(parent/"PLAN.json")["environment"])
    environment.pop("bitsandbytes")
    pins = sources(); attest(pins)
    # Build all microplans before writing anything; bad seeds/source fail early.
    plans=[]
    for ordinal in range(games//16):
        plans.append(base.rollout.make_plan(manifest,model_sha256,
            seeds=list(seeds[ordinal*16:(ordinal+1)*16]),
            actor_rng_seeds=list(actor_rng_seeds[ordinal*32:(ordinal+1)*32]),
            command_budget=command_budget,cohort_id=f"{output.name}-{ordinal:06d}",environment=environment))
    if len(set(seeds)) != games or len(set(actor_rng_seeds)) != games*2:
        raise ValueError("globally distinct worlds and sampler streams required")
    output.mkdir(parents=True)
    members=[]
    for ordinal,plan in enumerate(plans):
        file=output/f"plans/{ordinal:06d}.json"; io.write(file,plan)
        members.append({"plan_path":io.relative(ROOT,file),"plan_file_sha256":io.sha(file),
                        "producer_path":io.relative(ROOT,output/f"producers/{ordinal:06d}")})
    index=buffer_api.make_plan(members,root=ROOT,target_games=games,
        mode="PRODUCTION" if mode=="PRODUCTION" else "TEST_FIXTURE")
    created=buffer_api.create_index(output/"buffer",index,root=ROOT)
    plan=base.bound({"schema":SCHEMA+"-plan","owner":owner,"parent":current,
        "mode":mode,"physical_games":games,"seat_episodes":2*games,"model_decisions":60*games,
        "inference_microbatch_seats":32,"day_cohorts":group_count,"protocol":protocol,
        "execution":EXECUTION,"source_pins":pins,"core_identity":core.identity(),
        "stage_seconds":stage_seconds,"index_file_sha256":created["index_file_sha256"],
        "seed_ownership":"EXPLICIT_OWNER_ASSERTION_NOT_GLOBAL_EXPOSURE_AUTHENTICATION",
        "world_seeds":list(seeds),"actor_rng_seeds":list(actor_rng_seeds),
        "schedule":current["schedule"],"automatic_retry":False,"automatic_cleanup":False})
    io.write(output/"PLAN.json",plan)
    return {"plan_file_sha256":io.sha(output/"PLAN.json"),"plan":plan,"executed_games":0}


def _open(output, expected):
    output=_path(output)
    if io.sha(output/"PLAN.json") != expected: raise ValueError("externally pinned owner PLAN required")
    plan=io.read(output/"PLAN.json"); base.rollout._sealed(plan)
    if plan["schema"] != SCHEMA+"-plan" or plan["execution"] != EXECUTION:
        raise ValueError("typed streaming owner plan required")
    attest(plan["source_pins"])
    if plan["core_identity"] != core.identity(): raise ValueError("core source/execution changed")
    _protocol(plan["protocol"])
    state=buffer_api.status(output/"buffer",plan["index_file_sha256"],root=ROOT)
    index=io.read(output/"buffer/ROLLOUT-INDEX.json")
    if (state["target_games"] != plan["physical_games"]
            or index["behavior"]["model_sha256"] != plan["parent"]["model_sha256"]
            or index["behavior"]["model_manifest_sha256"] != plan["parent"]["manifest_sha256"]
            or index["behavior"]["model_optimizer_age"] != plan["parent"]["optimizer_age"]
            or [s for m in index["members"] for s in m["seeds"]] != plan["world_seeds"]
            or [s for m in index["members"] for s in m["actor_rng_seeds"]] != plan["actor_rng_seeds"]):
        raise ValueError("owner/index world count mismatch")
    return output,plan,state


def status(output, expected_plan_sha256):
    output,plan,state=_open(output,expected_plan_sha256)
    if (output/"RESULT.json").exists():
        result=io.read(output/"RESULT.json")
        if result["owner_plan_file_sha256"] != expected_plan_sha256:
            raise ValueError("closed owner/result join mismatch")
        child=_parent(output/"fit",tensors=False)
        if result["child"] != child: raise ValueError("closed child metadata changed")
        return {"status":"CLOSED","child":child,"result":result,"tensor_restore_verified":False}
    if (output/"fit/RESULT.json").exists(): label="FIT_COMMITTED_OWNER_PUBLICATION_INCOMPLETE"
    elif (output/"fit").exists(): label="FIT_INCOMPLETE_REFUSED_NO_AUTORETRY"
    elif (output/"RUN-FAILURE.json").exists(): label="FAILED_OWNER_RECONCILIATION_REQUIRED"
    else: label="COLLECTION_OR_WITNESS_PENDING"
    return {"status":label,"buffer":state,"mode":plan["mode"],"tensor_restore_verified":False}


def validate_parent(saved,result,plan,manifest,*,model_sha,source,profile,parameter_names):
    """Old is checked AS OLD; new records use an explicit private validator view."""
    args=dict(model_sha=model_sha,profile=profile,parameter_names=parameter_names)
    if saved.get("schema") == base.OPTIMIZER_SCHEMA:
        origin,ids,batch=base.validate_parent(saved,result,plan,manifest,source=base.sources(),**args)
        return origin,ids,{"kind":"EXPLICIT_ORIGINAL_B32_TO_INDEXED_STREAMING_PPO",
            "parent_model_sha256":model_sha,"parent_global_age":saved["updates"],
            "old_source_pins":saved["source_pins"],"old_execution":base.EXECUTION,
            "new_source_pins":source,"new_execution":EXECUTION,"batch_origin":batch,
            "optimizer_conversion":"NONE_EXACT_ADAM8BIT_RNG_AGE_AND_SCHEDULE"}
    check=FunctionType(base.legacy.validate_parent.__code__,{**base.legacy.validate_parent.__globals__,
        "SCHEMA":SCHEMA,"OPTIMIZER_SCHEMA":OPTIMIZER_SCHEMA,"EXECUTION":EXECUTION})
    origin,ids=check(saved,result,plan,manifest,source=source,**args)
    transition=saved.get("stream_origin")
    if (not isinstance(transition,dict) or transition != plan.get("stream_origin")
            or transition.get("new_source_pins") != source or transition.get("new_execution") != EXECUTION
            or transition.get("kind") != "EXPLICIT_ORIGINAL_B32_TO_INDEXED_STREAMING_PPO"
            or saved.get("stream_protocol") != plan.get("protocol")
            or saved.get("stream_cursor") != result.get("fit",{}).get("cursor")
            or saved["stream_cursor"].get("closed") is not True):
        raise ValueError("exact source-bound CLOSED streaming parent required")
    core._sealed(saved["stream_cursor"])
    return origin,ids,transition


def _preflight():
    if (not sys.dont_write_bytecode or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or any(os.environ.get(k) != "1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"))):
        raise ValueError("cold -B CUBLAS:4096:8 and all three thread counts1 required before worlds")


def fit(output, expected_plan_sha256):
    """Internal guarded stage. Actual fresh-neural verification precedes updates."""
    _preflight(); output,owner,_=_open(output,expected_plan_sha256)
    if (output/"fit").exists(): raise ValueError("partial/completed fit cannot be overwritten or silently rerun")
    completion=io.sha(output/"buffer/ROLLOUT-COMPLETE.json")
    parent=_path(owner["parent"]["directory"])
    if _parent(parent) != owner["parent"]: raise ValueError("actual immutable parent checkpoint changed")
    import bitsandbytes as bnb
    import torch
    profile={**base.lab.device_profile(),"bitsandbytes":str(bnb.__version__)}
    graph,manifest=base.policy.load_model(parent/"model.pt",device="cuda",
        expected_file_sha256=owner["parent"]["model_sha256"])
    if graph.config != base.policy.DEFAULT: raise ValueError("intended full60M16layer graph required")
    saved=torch.load(parent/"optimizer.pt",map_location="cpu",weights_only=True)
    names=[n for n,_ in graph.named_parameters()]
    origin,prior_ids,transition=validate_parent(saved,io.read(parent/"RESULT.json"),
        io.read(parent/"PLAN.json"),manifest,model_sha=owner["parent"]["model_sha256"],
        source=owner["source_pins"],profile=profile,parameter_names=names)
    if saved["plan_sha256"] != owner["parent"]["plan_file_sha256"]:
        raise ValueError("actual parent optimizer/PLAN file join mismatch")
    optimizer=bnb.optim.AdamW8bit(graph.parameters(),lr=base.learning_rate_at(saved["updates"],saved["settings"]),
        betas=(.9,.999),eps=1e-8,weight_decay=0.,min_8bit_size=4096,block_wise=True)
    # A CPU load_state_dict may alias source moment tensors. Keep the loaded
    # parent object pristine even in same-device tests/continuation checks.
    base.lab.restore_bc_optimizer(optimizer,base.legacy._cpu_copy(saved));graph.eval()
    if saved.get("schema")==OPTIMIZER_SCHEMA and saved["stream_cursor"].get("state") != core._state(
            graph,optimizer,next(graph.parameters()).device):
        raise ValueError("actual restored streaming tensors/Adam/RNG differ from closed cursor")
    target=output/"fit";target.mkdir()
    io.write(target/"STARTED.json",{"owner_plan_file_sha256":expected_plan_sha256,
        "parent_model_sha256":owner["parent"]["model_sha256"],"global_age":saved["updates"]})
    rollback=base.legacy._cpu_copy(graph.state_dict()); started=time.monotonic()
    try:
        neural=[];ids=[]
        for shard in buffer_api.iter_shards(output/"buffer",owner["index_file_sha256"],completion,root=ROOT):
            checked=shard["checked"]
            groups,new_ids=base.admitted_groups(checked,manifest,owner["parent"]["model_sha256"],prior_ids+ids)
            del groups
            if checked["plan"]["environment"] != {k:v for k,v in profile.items() if k!="bitsandbytes"}:
                raise ValueError("original producer/fitter numerical environment mismatch")
            proof=base.rollout.verify_graph(graph,checked["plan"],checked["episodes"])
            neural.append({"ordinal":shard["ordinal"],"receipt":shard["receipt"],"numerical_witness":proof})
            ids.extend(new_ids)
            print(json.dumps({"phase":"EXECUTED_NEURAL_ADMISSION","chunk":shard["ordinal"],"calls":proof["calls"]}),flush=True)
            del checked,shard
        if len(ids)!=owner["seat_episodes"]: raise ValueError("ALL registered fresh episodes required before any update")
        reader=buffer_api.IndexedReader(output/"buffer",owner["index_file_sha256"],completion,root=ROOT)
        admission=base.bound({"schema":SCHEMA+"-executed-admission","source_pins":owner["source_pins"],
            "model_sha256":owner["parent"]["model_sha256"],"model_manifest_sha256":manifest["content_sha256"],
            "reader_identity":reader.verify(),"physical_games":owner["physical_games"],
            "engine_reconstruction_games":owner["physical_games"],"responsive_reference_games":0,
            "episode_ids":ids,"numerical_witnesses":neural,"caller_boolean_accepted":False})
        plan={"schema":SCHEMA+"-fit-plan","owner_plan_file_sha256":expected_plan_sha256,
            "source_pins":owner["source_pins"],"environment":profile,"execution":EXECUTION,
            "schedule":saved["settings"],"start_update":saved["updates"],"protocol":owner["protocol"],
            "parent_model_sha256":owner["parent"]["model_sha256"],
            "parent_optimizer_sha256":owner["parent"]["optimizer_sha256"],
            "admission_sha256":admission["content_sha256"],"bc_origin":origin,"stream_origin":transition}
        io.write(target/"PLAN.json",plan);io.write(target/"ADMISSION.json",admission)
        plan_sha=io.sha(target/"PLAN.json");admission_seconds=time.monotonic()-started
        receipt=core.optimize(graph,optimizer,reader,saved["settings"],saved["updates"],
            protocol=owner["protocol"],expected_identity=owner["core_identity"],
            max_updates=owner["protocol"]["optimizer_updates"])
        if receipt["cursor"]["closed"] is not True: raise ValueError("partial cursor publication not supported here")
        attest(owner["source_pins"])
        if _parent(parent)!=owner["parent"]: raise ValueError("immutable parent changed during fit")
        age=receipt["global_optimizer_steps_after"]
        child=base.policy.save_model(target/"STAGED-model.pt",graph,optimizer_steps=age,
            rollout_iteration=manifest["rollout_iteration"]+1,initialization_seed=manifest["initialization_seed"],
            lineage={"kind":"FRESH_INDEXED_ROLLOUT_STREAMING_PPO","ppo_protocol":SCHEMA,
                "parent_file_sha256":owner["parent"]["model_sha256"],"plan_sha256":plan_sha,
                "admission_sha256":admission["content_sha256"],"execution":EXECUTION})
        torch.save({"schema":OPTIMIZER_SCHEMA,"source_pins":owner["source_pins"],"execution":EXECUTION,
            "environment":profile,"updates":age,"settings":saved["settings"],"optimizer":optimizer.state_dict(),
            "optimizer_config":base.lab.OPTIMIZER_CONFIG,"parameter_names":names,
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
            "model_sha256":io.sha(target/"STAGED-model.pt"),"plan_sha256":plan_sha,"bc_origin":origin,
            "stream_origin":transition,"stream_protocol":owner["protocol"],"stream_cursor":receipt["cursor"],
            "last_rollout_iteration":manifest["rollout_iteration"]+1,
            "recent_episode_ids":(prior_ids+ids)[-4096:]},target/"STAGED-optimizer.pt")
        reader.verify();attest(owner["source_pins"])
        for old,new in (("STAGED-model.pt","model.pt"),("STAGED-model.pt.json","model.pt.json"),
                        ("STAGED-optimizer.pt","optimizer.pt")):
            (target/old).rename(target/new)
        result={"schema":SCHEMA+"-fit-result","status":io.PPO_STATUS,"plan_sha256":plan_sha,
            "source_pins":owner["source_pins"],"optimizer_updates":receipt["optimizer_steps"],
            "global_optimizer_updates":age,"before_model_sha256":owner["parent"]["model_sha256"],
            "after_model_sha256":io.sha(target/"model.pt"),"optimizer_sha256":io.sha(target/"optimizer.pt"),
            "manifest":child["content_sha256"],"admission_sha256":admission["content_sha256"],
            "physical_games":owner["physical_games"],"engine_reconstruction_games":owner["physical_games"],
            "responsive_reference_games":0,"natural_calls":owner["model_decisions"],
            "admission_seconds":admission_seconds,"elapsed_fit_seconds":time.monotonic()-started,
            "fit":receipt,"population_strength_established":False}
        io.write(target/"STAGED-RESULT.json",result);(target/"STAGED-RESULT.json").rename(target/"RESULT.json")
        return result
    except BaseException as exc:
        graph.load_state_dict(rollback,strict=True)
        base.lab.restore_bc_optimizer(optimizer,base.legacy._cpu_copy(saved))
        io.write(target/"FAILURE.json",{"error":type(exc).__name__+": "+str(exc),
            "committed_optimizer_updates":0,"restored_parent_age":saved["updates"],
            "parameters_adam_rng_restored":True,"no_result_no_admitted_child":True})
        raise


def materialize(output, expected_plan_sha256, ordinal):
    output,owner,_=_open(output,expected_plan_sha256)
    return buffer_api.materialize_chunk(output/"buffer",ordinal,owner["index_file_sha256"],root=ROOT)


def run(output, expected_plan_sha256, *, owner, execute=False):
    output,plan,state=_open(output,expected_plan_sha256)
    if execute is not True: return status(output,expected_plan_sha256)
    _preflight()
    if owner != plan["owner"]: raise ValueError("exact registered owner required")
    with (output/"OWNER.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/"RESULT.json").exists(): return status(output,expected_plan_sha256)
        if (output/"RUN-FAILURE.json").exists(): raise ValueError("failed run preserved; explicit owner reconciliation required")
        if _parent(plan["parent"]["directory"]) != plan["parent"]: raise ValueError("parent checkpoint changed")
        started=time.monotonic()
        try:
            index=io.read(output/"buffer/ROLLOUT-INDEX.json")
            for ordinal,member in enumerate(index["members"]):
                producer=_path(member["producer_path"])
                if not (producer/"RESULT.json").exists():
                    if producer.exists(): raise ValueError("partial producer retained; no reroll")
                    io.execute([sys.executable,"-B","-m","farm_neat.rollout.batched_rollout","collect",
                        "--execute","--plan",str(_path(member["plan_path"])),"--plan-sha256",member["plan_file_sha256"],
                        "--model",str(_path(plan["parent"]["directory"])/"model.pt"),"--output",str(producer)],
                        output/f"stages/collect-{ordinal:06d}",root=ROOT,timeout=plan["stage_seconds"])
                # Separate supervised interpreter executes the original fresh CPU witness.
                complete=output/f"buffer/chunks/{ordinal:06d}/COMPLETE.json"
                if not complete.exists():
                    io.execute([sys.executable,"-B",str(Path(__file__)),"materialize","--execute", "--output",str(output),
                        "--plan-sha256",expected_plan_sha256,"--ordinal",str(ordinal)],
                        output/f"stages/witness-{ordinal:06d}",root=ROOT,timeout=plan["stage_seconds"])
            buffer_api.finalize(output/"buffer",plan["index_file_sha256"],root=ROOT)
            if (output/"fit").exists(): raise ValueError("fit stage already exists; no automatic repeated update")
            io.execute([sys.executable,"-B",str(Path(__file__)),"fit","--execute","--output",str(output),
                        "--plan-sha256",expected_plan_sha256],output/"stages/fit",root=ROOT,timeout=plan["stage_seconds"])
            fit_result=io.read(output/"fit/RESULT.json");child=_parent(output/"fit")
            if fit_result["physical_games"] != plan["physical_games"] or fit_result["fit"]["cursor"]["closed"] is not True:
                raise ValueError("complete registered rollout and closed optimizer allocation required")
            final={"schema":SCHEMA+"-result","owner_plan_file_sha256":expected_plan_sha256,
                "status":"FRESH_STREAMING_PPO_CLOSED_NOT_STRENGTH","child":child,"mode":plan["mode"],
                "physical_games":plan["physical_games"],"engine_reconstruction_games":plan["physical_games"],
                "seat_episodes":plan["seat_episodes"],"model_decisions":plan["model_decisions"],
                "gradient_coverage":fit_result["fit"]["coverage"],"optimizer_updates":fit_result["optimizer_updates"],
                "fit_result_file_sha256":io.sha(output/"fit/RESULT.json"),"elapsed_run_seconds":time.monotonic()-started,
                "continued_collection_not_replayed_game_credit":True,"promotion_or_upload":False}
            io.write(output/"STAGED-RESULT.json",final);(output/"STAGED-RESULT.json").rename(output/"RESULT.json")
            return final
        except BaseException as exc:
            if not (output/"RUN-FAILURE.json").exists():
                io.write(output/"RUN-FAILURE.json",{"error":type(exc).__name__+": "+str(exc),
                    "owner_plan_file_sha256":expected_plan_sha256,"automatic_retry":False,
                    "partial_artifacts_retained":True,"optimizer_credit_requires_fit_RESULT":True})
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","status","run","materialize","fit"))
    p.add_argument("--output",required=True);p.add_argument("--plan-sha256");p.add_argument("--execute",action="store_true")
    p.add_argument("--owner");p.add_argument("--parent");p.add_argument("--model-sha256");p.add_argument("--optimizer-sha256")
    p.add_argument("--seed-start",type=int);p.add_argument("--sampler-start",type=int)
    p.add_argument("--games",type=int,default=2048);p.add_argument("--exclusive-seeds",action="store_true")
    p.add_argument("--mode",choices=("PRODUCTION","SMOKE"),default="PRODUCTION")
    p.add_argument("--accumulation-cohorts",type=int,default=8);p.add_argument("--updates",type=int)
    p.add_argument("--metric-every",type=int,default=8);p.add_argument("--metric-cohorts",default="0")
    p.add_argument("--stage-seconds",type=int,default=3600);p.add_argument("--ordinal",type=int)
    a=p.parse_args()
    if a.command=="prepare":
        if a.seed_start is None or a.sampler_start is None: p.error("explicit TRAIN and sampler starts required")
        groups=a.games//16*30
        protocol=core.make_protocol(accumulation_cohorts=a.accumulation_cohorts,
            optimizer_updates=a.updates if a.updates is not None else (groups+a.accumulation_cohorts-1)//a.accumulation_cohorts,
            metric_every_updates=a.metric_every,metric_cohorts=[int(x) for x in a.metric_cohorts.split(",")])
        result=prepare(a.output,parent=a.parent,model_sha256=a.model_sha256,optimizer_sha256=a.optimizer_sha256,
            owner=a.owner,seeds=list(range(a.seed_start,a.seed_start+a.games)),
            actor_rng_seeds=list(range(a.sampler_start,a.sampler_start+2*a.games)),exclusive_seeds=a.exclusive_seeds,
            mode=a.mode,protocol=protocol,stage_seconds=a.stage_seconds)
    elif a.command=="status" or not a.execute: result=status(a.output,a.plan_sha256)
    elif a.command=="run": result=run(a.output,a.plan_sha256,owner=a.owner,execute=True)
    elif a.command=="materialize": result=materialize(a.output,a.plan_sha256,a.ordinal)
    else: result=fit(a.output,a.plan_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__": main()
