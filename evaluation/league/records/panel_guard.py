#!/usr/bin/env python3
"""Offline, fail-closed panel scoring and job planning. Python 3 standard library.

This is a SIDE-CAR, not an automatic deployment or a silent change to the owner's
rubric. Its frozen-roster/KL rule needs a recorded operator/owner approval. It
never logs into machines, submits jobs, trains models, or uploads submissions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

ENGINE = 'engine-20260927-d91cbd52'
REFERENCES = [
    'pub-ice-fire-wr1k-lw', 'pub-shepherds-ledger-wr1k-lw', 'pub-godsmode-shops-lw',
    'tape-mmpq-115130538-lw', 'tape-decem-114962244-lw', 'tape-boey-115025308-lw',
    'tape-victor-tufa-115017808-lw', 'tape-marketshock-115158982-lw',
    'tape-marketshock-115161684-lw', 'v4b-g2-48236-land-a1fx-carrydrain',
    'fwd1land-46236-rs0929',
]
SCHEMA = 'panel-guard-frozen-t8-r11-kl-v2'


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def points(margin: float) -> float:
    if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not math.isfinite(margin):
        raise ValueError('margin must be finite numeric')
    return 1.0 if margin > 0 else 0.0 if margin < 0 else .5


def binary_kl(p: float, q: float) -> float:
    if not (0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError('probabilities must be in [0,1]')
    if p == q:
        return 0.0
    if q in (0,1):
        return math.inf
    return (p * math.log(p/q) if p else 0.0) + ((1-p)*math.log((1-p)/(1-q)) if p<1 else 0.0)


def kl_interval(values: Sequence[float], alpha: float=.05, *, anytime: bool=False,
                family: int=1) -> tuple[float,float]:
    """Chernoff inversion for independent [0,1] world outcomes at a fixed look.

    n*kl(sample_mean,mu) <= log(2/alpha). Not a Beta posterior. Ties are
    fractional rewards, not independent Bernoulli trials. Handles n=1 and
    degenerate samples without pretending unseen outcomes are impossible.
    With anytime=True spend alpha/[n(n+1)] over n; family also divides alpha.
    Neither option repairs deliberately selected worlds or censored outcomes.
    """
    if not (0 < alpha < 1) or type(family) is not int or family < 1:
        raise ValueError('0<alpha<1 and positive integer family are required')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in values):
        raise ValueError('outcomes must be finite numbers in [0,1]')
    n=len(values)
    if not n:
        return (0.0,1.0)
    p=math.fsum(values)/n
    a=alpha/family/(n*(n+1) if anytime else 1)
    threshold=math.log(2/a)/n
    lo,hi=0.0,p
    for _ in range(65):
        mid=(lo+hi)/2
        if binary_kl(p,mid)>threshold:
            lo=mid
        else:
            hi=mid
    lower=lo  # outward rounding, not inward
    lo,hi=p,1.0
    for _ in range(65):
        mid=(lo+hi)/2
        if binary_kl(p,mid)>threshold:
            hi=mid
        else:
            lo=mid
    return lower,hi


def sign_tail(positive: int, nonzero: int) -> float | None:
    if not nonzero:
        return None
    if not 0<=positive<=nonzero:
        raise ValueError('invalid sign count')
    return math.fsum(math.comb(nonzero,j) for j in range(positive,nonzero+1))/2**nonzero


def summarize_pair(worlds: Mapping[int, Mapping[bool,float]], seeds: Sequence[int],
                   *, sampling: str='selected_stress', alpha: float=.05) -> dict:
    """Only the manifest's exact seeds and BOTH seats enter an inferential panel.
    Partial outcomes remain visible as deterministic full-bank score bounds.
    No conditional-success population inference on a partially returned panel.
    """
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('nonempty unique declared world seeds required')
    observed=[]; values=[]; margins=[]; seats={True:[],False:[]}; incomplete=[]
    for seed in seeds:
        row=worlds.get(seed,{})
        if set(row)-{True,False}:
            raise ValueError('invalid seat key')
        for seat,m in row.items():
            observed.append(points(m)); seats[seat].append(m)
        if set(row)=={True,False}:
            values.append((points(row[True])+points(row[False]))/2)
            margins.append((row[True]+row[False])/2)
        else:
            incomplete.append(seed)
    complete=not incomplete
    lb,ub=kl_interval(values,alpha) if complete else (0.0,1.0)
    nslots=2*len(seeds); total=math.fsum(observed)
    wins=sum(v==1 for v in observed); ties=sum(v==.5 for v in observed)
    nt=sum(m!=0 for m in margins); pos=sum(m>0 for m in margins)
    representative=sampling=='independent_random'
    return {
        'lb':lb,'ub':ub,'expected_points':math.fsum(values)/len(values) if values else None,
        'strict_win_fraction_observed':wins/len(observed) if observed else None,
        'tie_fraction_observed':ties/len(observed) if observed else None,
        'W':wins,'L':len(observed)-wins-ties,'T':ties,
        'games_observed':len(observed),'worlds_complete':len(values),'worlds_required':len(seeds),
        'complete':complete,'missing_worlds':incomplete,
        'full_bank_points_range':[total/nslots,(total+nslots-len(observed))/nslots],
        'population_ci_claim':bool(representative and complete),
        'interval_label':('fixed-look bounded-mean 95% CI (random-world assumption)' if representative and complete
                          else 'conditional KL score only; selected worlds are not a population sample' if complete
                          else 'incomplete panel: no inference'),
        'mean_world_margin':statistics.fmean(margins) if margins else None,
        'median_world_margin':statistics.median(margins) if margins else None,
        'world_margin_wins':pos,'world_margin_nontied':nt,'one_sided_sign_p':sign_tail(pos,nt),
        'mean_margin_seat0':statistics.fmean(seats[True]) if seats[True] else None,
        'mean_margin_seat1':statistics.fmean(seats[False]) if seats[False] else None,
        'decided':('A>B' if lb>.5 else 'B>A' if ub<.5 else 'undecided') if representative and complete else 'not_certified',
    }


def fixed_point(candidates: Iterable[str], references: Iterable[str], lower: Mapping[tuple[str,str],float],
                *, k: int=8, max_iterations: int=200) -> dict:
    """Diagnostic only: exact cycle detection; deterministic ties; final S is fresh.
    lower[(e,o)] is directional. Missing edges use the logical bound zero, never
    a model prediction. This does not establish that the returned set is globally
    best, or that a fixed point is unique. A frozen epoch avoids this circularity.
    """
    names=sorted(set(candidates)); refs=set(references)
    if not names or k<1 or max_iterations<1 or refs & set(names):
        raise ValueError('invalid candidates/references/k/iterations')
    k=min(k,len(names))
    def score(roster):
        target=set(roster)|refs
        return {e: math.fsum(lower.get((e,o),0.0) for o in sorted(target-{e}))/len(target-{e})
                if target-{e} else 0.0 for e in names}
    scores=score(names)
    top=tuple(sorted(sorted(names,key=lambda n:(-scores[n],n))[:k]))
    seen={}; trace=[]; status='iteration_limit'; period=None
    for _ in range(max_iterations):
        if top in seen:
            status='cycle'; period=len(trace)-seen[top]; break
        seen[top]=len(trace); trace.append(list(top))
        scores=score(top)
        nxt=tuple(sorted(sorted(names,key=lambda n:(-scores[n],n))[:k]))
        if nxt==top:
            status='fixed_point'; break
        top=nxt
    scores=score(top)
    return {'status':status,'cycle_length':period,'T':sorted(top,key=lambda n:(-scores[n],n)),
            'S':scores,'trace':trace,'best_authority':False}


def manifest_issues(manifest: Mapping) -> list[str]:
    issues=[]
    T=manifest.get('T',[]); R=manifest.get('R',[]); seeds=manifest.get('seeds',[])
    if manifest.get('schema')!=SCHEMA:
        issues.append('wrong_manifest_schema')
    if not isinstance(T,list) or len(T)!=8 or len(set(T))!=8 or any(not isinstance(n,str) for n in T):
        issues.append('T_must_have_exactly_eight_distinct_names')
    if R!=REFERENCES:
        issues.append('required_R11_changed_or_missing')
    if set(T)&set(R):
        issues.append('T_and_R_overlap')
    if (not isinstance(seeds,list) or len(seeds)!=16 or len(set(seeds))!=16 or
        any(type(s) is not int for s in seeds)):
        issues.append('exactly_sixteen_distinct_integer_worlds_required')
    if manifest.get('engine')!=ENGINE:
        issues.append('wrong_engine')
    if manifest.get('sampling') not in ('selected_stress','independent_random'):
        issues.append('unknown_sampling_design')
    if not manifest.get('approval_receipt'):
        issues.append('frozen_roster_and_KL_rule_not_approved')
    assets=manifest.get('assets',{})
    for n in sorted(set(T)|set(R)):
        sha=(assets.get(n) or {}).get('archive_sha256')
        if not isinstance(sha,str) or not re.fullmatch('[a-f0-9]{64}',sha):
            issues.append('missing_archive_identity:'+n)
    return issues


def normalize_engine(value: Any) -> str | None:
    if not isinstance(value,str):
        return None
    # Existing DB exports use a full path. Farm exports append a source label;
    # this normalization does NOT independently attest the actual runtime.
    if value.endswith(' (farm_game.sbatch)'):
        value=value[:-len(' (farm_game.sbatch)')]
    return Path(value).name


def record_issues(r: Mapping, manifest: Mapping, *, legacy: bool=False) -> list[str]:
    issues=[]
    def bad(code):
        issues.append(code)
    if r.get('clean') is not True:
        bad('clean_not_true')
    if r.get('crn_rng') is not True:
        bad('crn_not_true')
    if normalize_engine(r.get('engine'))!=manifest.get('engine'):
        bad('engine_mismatch')
    if r.get('replay_verified') is not True:
        bad('replay_unverified')
    if r.get('status')!=['DONE','DONE']:
        bad('status_not_done_done')
    fb=r.get('fallbacks')
    if not isinstance(fb,list) or len(fb)!=2 or any(type(x) is not int or x!=0 for x in fb):
        bad('fallbacks_missing_or_nonzero')
    hard=r.get('hard_errors')
    if hard is None and isinstance(r.get('seat_stats'),list):
        hard=[x.get('hard_errors') if isinstance(x,dict) else None for x in r['seat_stats']]
    if hard is None and legacy:
        pass  # diagnostic only; analyze() blocks BEST in legacy mode
    elif not isinstance(hard,list) or len(hard)!=2 or any(type(x) is not int or x!=0 for x in hard):
        bad('hard_errors_unattested_or_nonzero')
    cash=r.get('cash')
    if not isinstance(cash,list) or len(cash)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in cash):
        bad('cash_missing_or_nonfinite')
    if type(r.get('world_seed')) is not int:
        bad('invalid_seed')
    a,b=r.get('seat0'),r.get('seat1')
    if not isinstance(a,str) or not isinstance(b,str) or a==b:
        bad('invalid_players')
    hashes=r.get('archive_sha256')
    if not isinstance(hashes,list) or len(hashes)!=2:
        bad('archive_identity_missing')
    else:
        for i,n in enumerate((a,b)):
            expected=(manifest.get('assets',{}).get(n) or {}).get('archive_sha256')
            if not expected or hashes[i]!=expected:
                bad('archive_identity_mismatch:'+str(n))
    return issues


def logical_key(r: Mapping) -> tuple:
    return r.get('world_seed'),r.get('seat0'),r.get('seat1')


def resolve_records(records: Iterable[Mapping]) -> tuple[list[dict],list[dict]]:
    """Resolve linear explicit corrections; quarantine duplicates, forks and ID reuse.
    A correction names a CURRENT record_id/key and needs a nonempty reason.
    Validation happens after resolution: a rejected replacement never resurrects
    the older favorable outcome. Audit issues remain blockers for human review.
    """
    current={}; aliases={}; heads={}; conflicts=set(); issues=[]
    relevant=('world_seed','seat0','seat1','archive_sha256','engine','crn_rng','cash','status',
              'fallbacks','hard_errors','seat_stats','replay_verified','clean')
    for i,raw in enumerate(records,1):
        r=dict(raw); key=logical_key(r)
        rid=r.get('record_id') or r.get('key') or f'line:{i}'
        parent=r.get('supersedes')
        if rid in aliases and aliases[rid]!=key:
            conflicts.add(key); conflicts.add(aliases[rid])
            issues.append({'line':i,'reason':'record_id_collision'}); continue
        if key in current and current[key]==r:
            aliases[rid]=key; heads[key].add(rid); continue
        if parent:
            root=aliases.get(parent)
            if (root is None or root!=key or parent not in heads.get(key,set())
                    or rid in aliases or not str(r.get('reason') or '').strip()):
                conflicts.add(key); issues.append({'line':i,'reason':'invalid_or_forked_correction','key':list(key)}); continue
            current[key]=r; conflicts.discard(key); aliases[rid]=key; heads[key]={rid}
        elif key not in current:
            current[key]=r; aliases[rid]=key; heads[key]={rid}
        else:
            old=current[key]
            if any(old.get(k)!=r.get(k) for k in relevant):
                conflicts.add(key); issues.append({'line':i,'reason':'conflicting_duplicate','key':list(key)})
            else:
                heads[key].add(rid)
            aliases[rid]=key
    return [r for k,r in current.items() if k not in conflicts],issues


def read_jsonl(path: Path) -> list[dict]:
    rows=[]
    with path.open() as f:
        for i,line in enumerate(f,1):
            if not line.strip():
                continue
            try:
                r=json.loads(line,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except ValueError as e:
                raise ValueError(f'{path}:{i}: invalid JSON ({e})') from e
            if not isinstance(r,dict):
                raise ValueError(f'{path}:{i}: expected object')
            rows.append(r)
    return rows


def build_pairs(records: Iterable[Mapping], manifest: Mapping, *, legacy: bool=False) -> tuple[dict,dict]:
    resolved,dups=resolve_records(records); pairs={}; reasons=Counter(); accepted=[]; quarantined=[]; off_roster=0
    all_names=set(manifest.get('T',[]))|set(manifest.get('R',[]))
    for r in resolved:
        if r.get('seat0') not in all_names or r.get('seat1') not in all_names:
            off_roster+=1; continue
        bad=record_issues(r,manifest,legacy=legacy)
        if bad:
            reasons.update(bad); quarantined.append({'key':r.get('key'),'reasons':bad}); continue
        a,b=r['seat0'],r['seat1']; A,B=sorted((a,b)); c0,c1=r['cash']
        pairs.setdefault((A,B),{}).setdefault(r['world_seed'],{})[a==A] = c0-c1 if a==A else c1-c0
        accepted.append(r)
    return pairs,{'accepted_records':len(accepted),'off_roster_records':off_roster,
                  'quarantine_reason_counts':dict(sorted(reasons.items())),
                  'quarantined':quarantined,'duplicate_or_correction_issues':dups,
                  'legacy_diagnostic_only':legacy}


def directional_worlds(pairs: Mapping, a: str, b: str) -> dict:
    A,B=sorted((a,b)); worlds=pairs.get((A,B),{})
    if a==A:
        return worlds
    # Both sign and seat meaning reverse. Recompute all sign statistics downstream.
    return {s:{not seat:-m for seat,m in w.items()} for s,w in worlds.items()}


def analyze(records: Iterable[Mapping], manifest: Mapping, *, legacy: bool=False) -> dict:
    issues=manifest_issues(manifest)
    pairs,audit=build_pairs(records,manifest,legacy=legacy)
    seeds=manifest.get('seeds',[]); T=manifest.get('T',[]); R=manifest.get('R',[])
    rows=[]
    for a in T:
        opponents=sorted((set(T)|set(R))-{a}); cells={}
        for b in opponents:
            cells[b]=summarize_pair(directional_worlds(pairs,a,b),seeds,sampling=manifest.get('sampling','selected_stress'))
        missing=[b for b in opponents if not cells[b]['complete']]
        observed=[cells[b]['lb'] for b in opponents if cells[b]['complete']]
        rows.append({'name':a,'S':math.fsum(observed)/len(opponents) if not missing else None,
                     'minLB':min(observed) if not missing else None,
                     'mean_points':statistics.fmean(c['expected_points'] for c in cells.values()) if not missing else None,
                     'completed_panels':len(opponents)-len(missing),'required_panels':len(opponents),
                     'missing':missing,'cells':cells})
    eligible=[r for r in rows if not r['missing']]
    ranked=sorted(eligible,key=lambda r:(-r['S'],-r['minLB'],r['name']))
    blocked=bool(issues or legacy or audit['duplicate_or_correction_issues'])
    best=ranked[0]['name'] if ranked and not blocked else None
    return {'schema':SCHEMA,'manifest_sha256':digest(manifest),'rubric':'frozen T8 + exact R11; exact declared worlds; mean conditional KL lower bound',
            'not_the_original_fixed_point_rubric':True,'original_rubric_best':None,
            'best':best,'blocked':blocked,'block_reasons':issues+(['legacy_records_not_qualified'] if legacy else [])+
            (['ambiguous_record_history'] if audit['duplicate_or_correction_issues'] else []),
            'eligible_by_panel_completion':[r['name'] for r in ranked],
            'rows':rows,'record_audit':audit,'upload_authorized':False,
            'population_warning':'Selected medoids/rare worlds support a stress-bank ranking, not a hosted-population 95% guarantee.'}


def make_plan(records: Iterable[Mapping], manifest: Mapping, *, cap: int|None=None,
              inflight: Iterable[tuple[int,str,str]]=()) -> dict:
    """Pure missing-job plan, both seats, no statistical early stopping.
    Missing archives are blockers, not grounds for shrinking the roster.
    Every pair's next world is offered before its next-next world. Existing
    one-seat results get the missing partner first. Inflight keys are exact
    manifest-identity jobs supplied by the dispatcher, never inferred from name.
    """
    if cap is not None and cap<0:
        raise ValueError('cap must be nonnegative')
    pairs,audit=build_pairs(records,manifest)
    T=manifest['T']; R=manifest['R']; seeds=manifest['seeds']; active=set(inflight)
    all_pairs=list(itertools.combinations(T,2))+[(a,b) for a in T for b in R]
    all_pairs=[tuple(sorted(p)) for p in all_pairs]
    all_pairs=list(dict.fromkeys(all_pairs))
    packets=[]
    for pair_index,(a,b) in enumerate(all_pairs):
        w=pairs.get((a,b),{})
        missing_world_index=0
        for seed in seeds:
            have=w.get(seed,{})
            jobs=[{'world_seed':seed,'seat0':x,'seat1':y,
                   'archive_sha256':[manifest.get('assets',{}).get(x,{}).get('archive_sha256'),
                                     manifest.get('assets',{}).get(y,{}).get('archive_sha256')]}
                  for order,x,y in [(True,a,b),(False,b,a)] if order not in have and (seed,x,y) not in active]
            if not jobs:
                continue
            packets.append((0 if len(have)==1 else 1,missing_world_index,pair_index,jobs))
            missing_world_index+=1
    jobs=[job for _,_,_,chunk in sorted(packets,key=lambda p:p[:3]) for job in chunk]
    total=len(jobs)
    selected=[]
    for _,_,_,chunk in sorted(packets,key=lambda p:p[:3]):
        if cap is None or len(selected)+len(chunk)<=cap:
            selected.extend(chunk)
    return {'schema':'panel-guard-plan-v2','manifest_sha256':digest(manifest),'required_pairs':len(all_pairs),
            'total_unfinished_not_inflight':total,'jobs':selected,
            'manifest_blockers':manifest_issues(manifest),'record_audit':audit,
            'automatic_dispatch':False,'stop_rule':'none inside T/R; both seats required',
            'ideal_hours_at_40_cores':[total*4/60/40,total*7/60/40],
            'core_hours':[total*4/60,total*7/60]}


def emit_tsv(plan: Mapping, manifest: Mapping, episode_base: int) -> str:
    if plan['manifest_sha256']!=digest(manifest):
        raise ValueError('plan/manifest identity mismatch')
    if manifest_issues(manifest):
        raise ValueError('manifest not approved and fully identified; cannot emit runnable TSV')
    if type(episode_base) is not int or episode_base<0:
        raise ValueError('explicit reserved nonnegative episode base required')
    lines=['a_name\ta_archive\tb_name\tb_archive\tworld_seed\tepisode_id']
    valid_pairs={frozenset(p) for p in itertools.combinations(manifest['T'],2)} | {frozenset((a,b)) for a in manifest['T'] for b in manifest['R']}
    seen=set()
    for i,j in enumerate(plan['jobs']):
        a,b=j['seat0'],j['seat1']
        key=(j['world_seed'],a,b)
        if key in seen or j['world_seed'] not in manifest['seeds'] or a==b or frozenset((a,b)) not in valid_pairs:
            raise ValueError('duplicate or out-of-manifest planned job')
        seen.add(key)
        if j.get('archive_sha256') != [manifest['assets'][a]['archive_sha256'],manifest['assets'][b]['archive_sha256']]:
            raise ValueError('planned archive identities do not match manifest')
        paths=[manifest['assets'][n].get('archive_path') for n in (a,b)]
        if any(not isinstance(p,str) or not p or '\t' in p or '\n' in p or '\r' in p for p in paths):
            raise ValueError('missing/invalid archive path')
        fields=[a,paths[0],b,paths[1],str(j['world_seed']),str(episode_base+i)]
        if any(any(c in s for c in '\t\r\n') for s in fields):
            raise ValueError('TSV control character in field')
        lines.append('\t'.join(fields))
    return '\n'.join(lines)+'\n'


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    for cmd in ('analyze','plan'):
        p=sub.add_parser(cmd); p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--games',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
        if cmd=='analyze':
            p.add_argument('--legacy-diagnostic',action='store_true')
        else:
            p.add_argument('--cap',type=int); p.add_argument('--inflight',type=Path)
    p=sub.add_parser('emit-tsv'); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--episode-base',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv)
    try:
        m=json.loads(args.manifest.read_text())
        # Draft manifests are useful for reporting blockers, but basic structural
        # errors cannot be interpreted safely.
        structural=[e for e in manifest_issues(m) if not e.startswith('missing_archive_identity:')
                    and e!='frozen_roster_and_KL_rule_not_approved']
        if structural:
            raise ValueError('; '.join(structural))
        if args.cmd=='analyze':
            result=analyze(read_jsonl(args.games),m,legacy=args.legacy_diagnostic); write_json(args.out,result)
        elif args.cmd=='plan':
            inflight=[]
            if args.inflight:
                state=json.loads(args.inflight.read_text())
                if state.get('manifest_sha256')!=digest(m):
                    raise ValueError('inflight registry belongs to another manifest')
                inflight=[(j['world_seed'],j['seat0'],j['seat1']) for j in state['jobs']]
            result=make_plan(read_jsonl(args.games),m,cap=args.cap,inflight=inflight); write_json(args.out,result)
        else:
            result=emit_tsv(json.loads(args.plan.read_text()),m,args.episode_base)
            args.out.parent.mkdir(parents=True,exist_ok=True)
            with args.out.open('x') as f:
                f.write(result)
        print(args.out)
        return 0
    except (OSError,ValueError,KeyError,TypeError) as e:
        ap.exit(2,f'ERROR: {e}\n')

if __name__=='__main__':
    raise SystemExit(main())
