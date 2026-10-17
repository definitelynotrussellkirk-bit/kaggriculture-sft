"""Behaviour cloning of the full-economic MoE policy on whole PLAN PROGRAMS.

Owner: full-economic runtime/throughput maintainer (same lane as
``scripts/train_full_economic_fast.py``). This is a supervised stage on the
resident CURRENT fast learner, exactly like ``full_economic_moe_lab.bc-current``
(:784) in its checkpoint plumbing, but the targets are different objects:

  * ``bc-current`` clones per-turn primitive PREFIXES through
    ``Graph.replay_prefix`` and the legacy entity packer.
  * this command clones the COMPLETE published plan program of a decision turn,
    scored through the same conditional factor chain the campaign sampler uses
    (fixed prior in ``Graph.factor`` + remaining-window ``sampling_mask``), over
    the full-plan v4 adapter input the fast collector actually encodes.

Every example is constructed the way the driver collects one: restore
``FullSeasonPlan`` from the recorded pre-proposal snapshot, pack with
``full_economic_plan_observation.pack(...)`` (mirror of ``pack_plan`` at
train_full_economic_fast.py:262) and batch with ``collate_bucket`` (:597), so
the encoder input is bit-identical to collection. Targets are the recorded
program's factors (``validate_program`` + ``encode_command``) wrapped in a typed
``farm-full-economic-moe-behavior-remaining-window-v2`` trace and replayed with
``fast.replay_padded`` -- the exact replay the learner trains with.

``selected_logps``/``joint_logp`` in those traces are the uniform-over-declared-
support REFERENCE required by the trace contract (same construction as the
repository fixture ``tests/test_train_full_economic_fast.py:stop_trace``). They
are NOT behaviour log-probabilities of any policy and are never used as PPO old
log-probabilities; only the fresh replayed logp enters the loss.

No auxiliary head or its Adam state is updated. The default loss is factor-normalised
NLL over the whole optimizer-step group (like ``full_economic_bc.bc_loss``
:444). ``--loss-reduction command`` instead averages factors within each command
and commands over the complete optimizer group, including STOP. It is an explicit
supervised objective change, not a changed joint likelihood or policy grammar.
A small fit probe establishes learning connectivity, not strength.

``--planned-optimizer-steps``, ``--learning-rate`` and ``--entropy-coefficient``
are optional owner overrides of the INHERITED schedule. Each is validated by
``economic_learning_v2.training.validate_settings`` and written into the
published child's ``optimizer.pt["settings"]`` and ``PLAN.json["schedule"]``, so
a later driver run whose plan schedule equals them validates. Entropy 0.0 is
accepted and is how the GRPO recovery phase runs with structural entropy off.

If a PINNED source file has been edited since the parent was published, this
command refuses by default and names the files. ``--rebind-source-pins PATH...``
with ``--rebind-reason`` is the explicit owner opt-in: only the listed files may
differ, the transition is recorded per file with old/new sha256 in
PLAN/ADMISSION/RECEIPT/RESULT, and the published child carries the CURRENT
attested pins and adapter identity. No validator is relaxed to achieve this.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "src"))

from farm_neat.economy import action_space as actions
from farm_neat.policy import fast_profile as fast
from farm_neat.policy import plan_observation as plan_features
from farm_neat.policy import network as policy
from farm_neat.economy.orders import FullSeasonPlan
from farm_neat.training import moe_lab as lab
from farm_neat.training import campaign as campaign
from farm_neat.training import sustain as io
from farm_neat.training.schedule import learning_rate_at, validate_settings
from farm_neat.tooling import checkpoint_migration as migration

SCHEMA = campaign.SUPERVISED_STAGE + "-whole-program-v1"
OWN = ("src/farm_neat/training/program_bc.py", "tests/contracts/test_program_bc.py")
CORPUS_SCHEMA = "farm-full-economic-decision-step-program-corpus-v1"
DEFAULT_PARENT = "work/checkpoints/program-bc-v1-01"
DEFAULT_CORPUS = "work/datasets/corpus/games-v1-plus-repairs.jsonl.gz"
SOURCE_TRANSITION_SCHEMA = SCHEMA + "-source-pin-rebind-v1"
LOSS_REDUCTIONS = {
    "factor": "FACTOR_NORMALISED_PROGRAM_NLL_OVER_THE_WHOLE_OPTIMIZER_STEP_GROUP",
    "command": "COMMAND_MEAN_OF_WITHIN_COMMAND_FACTOR_NLL_OVER_THE_WHOLE_OPTIMIZER_STEP_GROUP_INCLUDING_STOP_V1",
}
TARGET_SEMANTICS = ("RECORDED_PUBLISHED_PROGRAM_FACTORS; UNIFORM_OVER_DECLARED_SUPPORT_REFERENCE_LOGPS"
                    "_ARE_NOT_BEHAVIOR_LOGPS_AND_ARE_NEVER_PPO_OLD_LOGPS")


def own_sources():
    """Pins for THIS command only; the campaign closure stays campaign.sources()."""
    return {name: campaign.file_sha(ROOT/name) for name in OWN if (ROOT/name).exists()}


# -------------------------------------------------------------- schedule
def override_settings(inherited, *, planned_optimizer_steps=None, learning_rate=None, entropy_coefficient=None):
    """Optional owner overrides of the INHERITED schedule; returns (settings, block).

    Returns the inherited settings unchanged and ``None`` when nothing is given.
    Anything returned here is published verbatim as the child's
    ``optimizer.pt["settings"]`` and ``PLAN.json["schedule"]``, so a later driver
    run whose plan schedule equals it validates. ``entropy_coefficient`` 0.0 is
    accepted by ``validate_settings`` and is how the GRPO recovery phase after BC
    runs with structural entropy off -- the driver reads it from the checkpoint
    (train_full_economic_fast.py:1417 grpo_terms / :1425 legacy.ppo_terms), never
    from a flag of its own.
    """
    if planned_optimizer_steps is None and learning_rate is None and entropy_coefficient is None:
        return copy.deepcopy(inherited), None
    settings = copy.deepcopy(inherited)
    if planned_optimizer_steps is not None:
        if type(planned_optimizer_steps) is not int or planned_optimizer_steps < 1:
            raise ValueError("positive integer planned_optimizer_steps required")
        settings["planned_optimizer_steps"] = planned_optimizer_steps
    if learning_rate is not None:
        if type(learning_rate) is not float or not 0.0 < learning_rate:
            raise ValueError("positive float learning rate required")
        settings["learning_rate"] = learning_rate
    if entropy_coefficient is not None:
        if type(entropy_coefficient) is not float or not 0.0 <= entropy_coefficient:
            raise ValueError("nonnegative float entropy coefficient required")
        settings["entropy_coefficient"] = entropy_coefficient
    validate_settings(settings)
    changed = sorted(k for k in settings if settings[k] != inherited[k])
    if not changed:
        return settings, None
    return settings, {"kind": "EXPLICIT_OWNER_SCHEDULE_OVERRIDE_AT_BC_BOUNDARY",
                      "inherited": copy.deepcopy(inherited), "new": copy.deepcopy(settings),
                      "changed": changed}


# ----------------------------------------------------------- source pins
def _pin_difference(recorded, current):
    return sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))


def recorded_adapter(game):
    """The full-plan input adapter identity the parent actually trained against."""
    full_plan = game.get("full_plan")
    if isinstance(full_plan, dict) and isinstance(full_plan.get("adapter"), dict):
        return full_plan["adapter"]
    return None


def rebind_source_pins(parent_plan, parent_result, game, *, allowlist=(), reason=""):
    """Explicit opt-in transition across an edit to a PINNED source file.

    Default behaviour is refusal: if any pinned closure this stage depends on --
    ``campaign.sources()`` and the full-plan adapter's own ``source_pins`` -- has
    drifted from what the parent recorded, the fit stops and names the files.
    With ``--rebind-source-pins`` the owner asserts that the listed files, and
    only those, may differ. Weights, Adam, RNG and age are untouched; only the
    recorded source identity is rebound, and the published child always carries
    the CURRENT attested pins and the CURRENT adapter identity.

    This never relaxes ``validate_parent``; it is a stricter gate in front of it,
    plus the one in-memory rewrite the fit needs: the parent's recorded game
    execution names the OLD adapter, while ``Learner`` builds the CURRENT one,
    so the inherited game-contract equality check would otherwise refuse.
    """
    allowlist = list(allowlist or ())
    if len(set(allowlist)) != len(allowlist):
        raise ValueError("distinct rebind allowlist entries required")
    for name in allowlist:
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts or not (ROOT/candidate).is_file():
            raise ValueError("rebind allowlist must name existing repository-relative files: " + str(name))
    if allowlist and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("explicit rebind reason required with a rebind allowlist")
    if reason and not allowlist:
        raise ValueError("a rebind reason without an allowlist rebinds nothing")
    if parent_result["source_pins"] != parent_plan["source_pins"]:
        raise ValueError("parent PLAN/RESULT source pins disagree; refuse to rebind an inconsistent parent")
    allowed = set(allowlist)
    current_adapter = plan_features.identity()
    checks = [("campaign_sources", parent_plan["source_pins"], campaign.sources())]
    adapter = recorded_adapter(game)
    if adapter is not None:
        semantic = sorted(k for k in set(adapter) | set(current_adapter)
                          if k not in ("source_pins", "content_sha256")
                          and adapter.get(k) != current_adapter.get(k))
        if semantic:
            raise ValueError("full-plan adapter semantics changed, not only source pins: " + repr(semantic))
        checks.append(("plan_adapter_source_pins", adapter["source_pins"], current_adapter["source_pins"]))
    scopes, files = {}, {}
    for scope, recorded, current in checks:
        differing = _pin_difference(recorded, current)
        outside = [name for name in differing if name not in allowed]
        if outside:
            raise ValueError(scope + ": pinned sources changed outside the rebind allowlist: " + repr(outside))
        scopes[scope] = differing
        for name in differing:
            entry = {"parent_sha256": recorded.get(name), "current_sha256": current.get(name)}
            if files.setdefault(name, entry) != entry:
                raise ValueError("parent records inconsistent pins for " + name)
    if not files:
        return None
    rebound = adapter is not None and bool(scopes.get("plan_adapter_source_pins"))
    if rebound:
        game["full_plan"]["adapter"] = copy.deepcopy(current_adapter)
    return campaign.bound({"schema": SOURCE_TRANSITION_SCHEMA, "reason": reason,
        "allowlist": sorted(allowed), "unused_allowlist": sorted(allowed - set(files)),
        "files": files, "differing_by_scope": scopes,
        "parent_adapter_content_sha256": None if adapter is None else adapter["content_sha256"],
        "current_adapter_content_sha256": current_adapter["content_sha256"],
        "parent_execution_adapter_rebound": rebound,
        "child_source_pins": "CURRENT_ATTESTED_CAMPAIGN_SOURCES_AND_ADAPTER_IDENTITY",
        "weights_optimizer_rng_age": "UNCHANGED; ONLY_RECORDED_SOURCE_IDENTITY_IS_REBOUND",
        "semantics": "OWNER_ASSERTS_THE_LISTED_EDITS_DO_NOT_CHANGE_THIS_STAGE_INPUTS_OR_TARGETS",
        "validator_unchanged": "VALIDATE_PARENT_AND_CHECKPOINT_ARE_NOT_RELAXED_BY_THIS_BLOCK"})


# ------------------------------------------------------------------ targets
def target_trace(program, *, step, max_commands, architecture_sha256, temperature=1.0):
    """Typed target trace over the SAME conditional chain the sampler walks.

    ``sampling_mask`` is evaluated with the record's own ``as_of_step`` so the
    remaining-window deadline support is the one the collector would have used
    at that decision turn; ``stop_only`` reproduces the last-slot budget mask.
    """
    actions._integer(max_commands, 2, actions.MAX_COMMANDS, "command budget")
    commands = actions.validate_program(program)
    if len(commands) > max_commands:
        raise ValueError("recorded program exceeds the registered command budget")
    encoded = [actions.encode_command(command) for command in commands]
    masks = [[policy.sampling_mask(factor["path"], tuple(factor["choices"]),
                                   stop_only=(slot == max_commands-1 and factor["path"] == "op"),
                                   as_of_step=step)
              for factor in command["factors"]]
             for slot, command in enumerate(encoded)]
    reference = [-math.log(sum(mask)) for command in masks for mask in command]
    for command, command_masks in zip(encoded, masks, strict=True):
        for factor, mask in zip(command["factors"], command_masks, strict=True):
            if not mask[factor["index"]]:
                raise ValueError("recorded factor lacks declared support at its own as_of_step")
    return {**policy.trace_header(step), "architecture_sha256": architecture_sha256,
            "action_schema_sha256": actions.ACTION_SCHEMA_SHA256, "temperature": temperature,
            "max_commands": max_commands, "program": commands, "commands": encoded,
            "sampling_masks": masks, "selected_logps": reference,
            "joint_logp": math.fsum(reference), "training_admission": False}


# ------------------------------------------------------------------ examples
def read_corpus(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream):
            line = line.strip()
            if not line:
                continue
            game = json.loads(line)
            if "error" in game:
                raise ValueError(f"corpus game {ordinal} recorded an engine error; refuse partial labels")
            expected = {"program_id", "seed", "seat", "opponent", "cash", "configuration", "records"}
            if set(game) != expected:
                raise ValueError("closed recorded-corpus game required")
            yield ordinal, game


def build_examples(corpus, *, steps, max_commands, architecture_sha256, limit=0, progress=None):
    """Collector-identical inputs: restore snapshot -> observe if needed -> pack v4."""
    identity = plan_features.identity()
    examples = []
    games = 0
    observed = 0
    for ordinal, game in read_corpus(corpus):
        games += 1
        for record in game["records"]:
            if set(record) != {"step", "observation", "snapshot", "program"}:
                raise ValueError("closed recorded decision record required")
            if record["step"] not in steps:
                continue
            controller = FullSeasonPlan.restore(record["snapshot"])
            observation = record["observation"]
            if (controller.executor.player != observation["player"]
                    or controller.executor._observation_step(observation) != controller.executor.stage_step):
                controller.observe(observation)
                observed += 1
            packed = plan_features.pack(observation, controller, game["configuration"],
                                        expected_identity=identity)
            trace = target_trace(record["program"], step=record["step"], max_commands=max_commands,
                                 architecture_sha256=architecture_sha256)
            examples.append({"game_ordinal": ordinal, "program_id": game["program_id"], "seed": game["seed"],
                             "seat": game["seat"], "opponent": game["opponent"], "step": record["step"],
                             "pack": packed, "trace": trace, "factors": len(trace["selected_logps"]),
                             "commands": len(trace["program"])})
            if progress and len(examples) % progress == 0:
                print(json.dumps({"packed_examples": len(examples)}), flush=True)
        if limit and len(examples) >= limit:
            break
    if not examples:
        raise ValueError("no decision records matched the requested steps")
    return examples, {"games": games, "re_observed_snapshots": observed,
                      "max_target_commands_including_stop": max(e['commands'] for e in examples),
                      "sampling_command_budget": max_commands,
                      "forced_stop": "FINAL_SLOT_IS_STOP_ONLY; NO_TARGET_TRUNCATION"}


def split_holdout(examples, *, fraction, key, seed):
    """Deterministic group-level split; no example of a holdout group is trained on."""
    if type(fraction) is not float or not 0.0 <= fraction < 1.0:
        raise ValueError("holdout fraction in [0,1) required")
    if key not in ("program_id", "seed"):
        raise ValueError("holdout grouping key must be program_id or seed")
    groups = sorted({example[key] for example in examples})
    count = int(round(len(groups)*fraction))
    if fraction > 0.0 and count < 1:
        raise ValueError("requested holdout fraction selects no group")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    held = set(shuffled[:count])
    train = [e for e in examples if e[key] not in held]
    holdout = [e for e in examples if e[key] in held]
    if not train:
        raise ValueError("holdout consumed the whole corpus")
    return train, holdout, {"key": key, "groups": len(groups), "holdout_groups": sorted(held),
                            "fraction": fraction, "split_seed": seed}


# ------------------------------------------------------------------ scoring
def encode_batch(learner, batch):
    """Driver-identical collate (pad multiple 32) then the campaign fast encoder."""
    torch = learner.torch
    device = next(learner.graph.parameters()).device
    collated = campaign.collate_bucket([e["pack"] for e in batch])
    tensors = tuple(torch.as_tensor(collated[k], device=device) for k in ("rows", "token_mask", "knownness"))
    context, values, _ = learner.encoder(*tensors)
    return context, values


def program_logp(learner, batch):
    context, _ = encode_batch(learner, batch)
    logp, entropy = fast.replay_padded(learner.graph, context, [e["trace"] for e in batch],
                                       expected_identity=learner.fast_id)
    return context, logp, entropy


def command_logp(factor_logps, traces):
    """Sum within-command mean logps, including STOP, for each program.

    These are supervised scores, NOT joint program log probabilities. Padding
    has zero weight. The optimizer group later divides by its actual command
    count, so a short program or microbatch is not overweighted.
    """
    lengths = [[len(c["factors"]) for c in trace["commands"]] for trace in traces]
    if (not lengths or any(not row or any(n < 1 for n in row) for row in lengths)
            or tuple(factor_logps.shape) != (len(traces), max(map(sum, lengths)))):
        raise ValueError("one nonempty command factor trace per padded score row required")
    width = factor_logps.shape[1]
    weights = [[w for n in row for w in [1.0/n]*n] + [0.0]*(width-sum(row)) for row in lengths]
    return (factor_logps * factor_logps.new_tensor(weights)).sum(-1)


def supervised_scores(learner, batch, reduction):
    """Return original joint logps and the separately named optimization score."""
    if reduction == "factor":
        _, logp, _ = program_logp(learner, batch)
        return logp, logp
    if reduction != "command":
        raise ValueError("known supervised loss reduction required")
    context, _ = encode_batch(learner, batch)
    traces = [e["trace"] for e in batch]
    logp, _, factors = fast.replay_padded(learner.graph, context, traces,
        expected_identity=learner.fast_id, return_factor_logps=True)
    return logp, command_logp(factors, traces)


def packed_identity(pack):
    """Hash actual actor arrays, including their dtype and shape."""
    h = hashlib.sha256()
    for name in ("rows", "knownness", "token_mask"):
        a = pack[name]
        h.update(str((name, a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def attach_preferences(examples, data, corpus_sha256, architecture_sha256, max_commands, *, first_stop=False):
    """Admit observed orderings only for the exact positive corpus inputs."""
    if (data.get("schema") != "farm-native-program-preferences-v1"
            or data.get("corpus_sha256") != corpus_sha256
            or len(data.get("records", [])) != len(examples)):
        raise ValueError("complete corpus-bound preference records required")
    indexed = {(e["game_ordinal"], e["step"]): e for e in examples}
    if len(indexed) != len(examples): raise ValueError("distinct corpus decision identities required")
    seen = set()
    for row in data["records"]:
        key = (row["game_ordinal"], row["step"])
        if key in seen or key not in indexed: raise ValueError("duplicate or unknown preference input")
        seen.add(key); example = indexed[key]
        if (packed_identity(example["pack"]) != row["packed_input_sha256"]
                or example["trace"]["program"] != row["chosen"]):
            raise ValueError("preference input or chosen program differs from actual corpus")
        scores = [row["chosen_score"], *[a["score"] for a in row["alternatives"]]]
        if (not 1 <= len(row["alternatives"]) <= 32
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores)):
            raise ValueError("bounded complete empirical WDL scores required")
        traces = [example["trace"]]; inferior = []; programs = {lab.canonical(row["chosen"])}
        for alternative in row["alternatives"]:
            trace = target_trace(alternative["program"], step=example["step"],
                                 max_commands=max_commands, architecture_sha256=architecture_sha256)
            identity = lab.canonical(trace["program"])
            if identity in programs: raise ValueError("distinct preference programs required")
            programs.add(identity)
            if alternative["score"] > row["chosen_score"]:
                raise ValueError("chosen target is not empirically maximal")
            if alternative["score"] < row["chosen_score"]: inferior.append(len(traces))
            traces.append(trace)
        example["preference"] = {"traces": traces, "inferior": inferior}
        if first_stop:
            keep = [i for i, t in enumerate(traces) if t["program"] == [{"op": "STOP"}]]
            if len(keep) != 1 or (keep[0] != 0 and scores[0] <= scores[keep[0]]):
                raise ValueError("first-STOP supervision requires KEEP and a strictly better chosen edit, or chosen KEEP")
            example["preference"]["keep_index"] = keep[0]
    return {"examples": len(examples), "programs": sum(len(e["preference"]["traces"]) for e in examples),
            "strict_pairs": sum(len(e["preference"]["inferior"]) for e in examples),
            "states_with_strict_pairs": sum(bool(e["preference"]["inferior"]) for e in examples)}


def preference_logps(learner, context, batch):
    """Reuse each encoded state; alternatives never alter encoder batch membership."""
    indices = [i for i, e in enumerate(batch) for _ in e["preference"]["traces"]]
    traces = [t for e in batch for t in e["preference"]["traces"]]
    index = learner.torch.tensor(indices, device=context.device)
    logps, _ = fast.replay_padded(learner.graph, context.index_select(0, index), traces,
                                 expected_identity=learner.fast_id)
    return list(logps.split([len(e["preference"]["traces"]) for e in batch]))


def preference_terms(logps, references, inferior):
    """Mean strict-pair logistic loss per state; ties supply zero preference credit.

    Reference likelihoods cancel initial program-length and confidence offsets.
    They are frozen teacher-forced scores, never behavior/PPO likelihoods.
    """
    import torch
    if not (len(logps) == len(references) == len(inferior)) or not logps:
        raise ValueError("one current/reference/pair set per state required")
    losses = []
    for current, reference, ids in zip(logps, references, inferior, strict=True):
        reference = current.new_tensor(reference).detach()
        if current.shape != reference.shape or not bool(torch.isfinite(current).all() & torch.isfinite(reference).all()):
            raise ValueError("finite matching reference likelihoods required")
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 0 < i < len(current) for i in ids):
            raise ValueError("distinct inferior alternatives required")
        delta = (current[0] - current[ids]) - (reference[0] - reference[ids])
        losses.append(torch.nn.functional.softplus(-delta).mean() if ids else current.sum()*0)
    return torch.stack(losses)


def first_stop_terms(logps, keep_indices):
    """Per-state BCE of the existing first STOP decision; no new policy head.

    Index zero is the empirically selected positive program. If it is not KEEP,
    admission requires its observed score to strictly exceed KEEP. Native joint
    likelihood of the one-factor STOP program is exactly the first STOP logp.
    Other complete programs still receive ordinary BC and are never auto-edited.
    """
    import torch
    if not logps or len(logps) != len(keep_indices):
        raise ValueError("one KEEP index per complete state required")
    losses = []
    for current, keep in zip(logps, keep_indices, strict=True):
        if (current.ndim != 1 or type(keep) is not int or not 0 <= keep < len(current)
                or not bool(torch.isfinite(current).all()) or not bool((current <= 0).all())):
            raise ValueError("finite native log probabilities and valid KEEP index required")
        stop = current[keep]
        # Avoid evaluating the unused log(0) branch when the label is KEEP.
        losses.append(-stop if keep == 0 else -torch.log(-torch.expm1(stop)))
    return torch.stack(losses)


def normalize_accumulated_gradients(parameters, *, temporary_denominator, actual_denominator):
    """One actual denominator over all microbatches, including short tails."""
    if temporary_denominator <= 0 or actual_denominator <= 0:
        raise ValueError("positive accumulation denominators required")
    scale = temporary_denominator/actual_denominator
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def replay_agreement(learner, batch):
    """One shared context, both replays: within-profile identity of the scorer."""
    torch = learner.torch
    with torch.no_grad():
        context, _ = encode_batch(learner, batch)
        records = [e["trace"] for e in batch]
        padded, _ = fast.replay_padded(learner.graph, context, records, expected_identity=learner.fast_id)
        scalar, _ = policy.Graph.replay(learner.graph, context, records)
        return float((padded-scalar).abs().max()), padded.tolist(), scalar.tolist()


def first_op_input_groups(rows):
    """Group identical actor inputs, empty prefixes and legal masks; no causal claim."""
    groups, supports = {}, {}
    for row in rows:
        support = (tuple(row['choices']), tuple(row['mask']))
        key = (row['input_sha256'], support)
        group = groups.setdefault(key, {'input_sha256': row['input_sha256'],
                                       'choices': row['choices'], 'mask': row['mask'],
                                       'prefix': [], 'examples': 0, 'targets': {}})
        group['examples'] += 1
        target = str(row['target'])
        group['targets'][target] = group['targets'].get(target, 0)+1
        item = supports.setdefault(support, {'inputs': set(), 'min': list(row['probabilities']),
                                             'max': list(row['probabilities'])})
        item['inputs'].add(row['input_sha256'])
        item['min'] = [min(a,b) for a,b in zip(item['min'], row['probabilities'], strict=True)]
        item['max'] = [max(a,b) for a,b in zip(item['max'], row['probabilities'], strict=True)]
    for group in groups.values():
        group['target_entropy_nats'] = -sum((n/group['examples'])*math.log(n/group['examples'])
                                            for n in group['targets'].values())
    return {'schema': 'farm-first-op-input-groups-v1', 'groups': list(groups.values()),
            'distinct_input_groups': len(groups),
            'multimodal_target_groups': sum(len(g['targets']) > 1 for g in groups.values()),
            'aligned_coordinate_spreads': [
                {'choices': list(key[0]), 'mask': list(key[1]), 'distinct_inputs': len(item['inputs']),
                 'spread': [hi-lo for lo,hi in zip(item['min'],item['max'],strict=True)]
                           if len(item['inputs']) > 1 else None}
                for key,item in supports.items()],
            'interpretation': 'DESCRIPTIVE_SENSITIVITY; NOT_CONNECTIVITY_OR_ECONOMIC_USEFULNESS'}


def holdout_metrics(learner, examples, *, batch_size, opening_step=0):
    """Factor-normalised NLL plus first-op diagnostics on the opening examples.

    ``greedy_first_op_accuracy`` is a hard argmax match. On a corpus whose
    teacher openings all start with the SAME op it is a 0/1 step function that
    only flips once P(target) overtakes the incumbent argmax, so it can read
    0.0 through a large, real improvement. The graded companions --
    ``mean_target_first_op_probability``, ``mean_target_first_op_rank`` and the
    ``greedy_first_op_distribution`` -- describe target confidence. Target labels
    differ across examples, so their probability spread is NOT a conditioning
    test: a fixed head can have positive spread. Compare aligned coordinates
    on genuinely distinct actor inputs with identical prefixes and legal masks.
    Connectivity and economically useful conditioning require separate checks.
    """
    torch = learner.torch
    total_logp, total_factors, opening = 0.0, 0, 0
    correct, probabilities, ranks, argmax_ops = 0, [], [], {}
    input_rows = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start+batch_size]
            context, logp, _ = program_logp(learner, batch)
            total_logp += float(logp.sum()); total_factors += sum(e["factors"] for e in batch)
            for row, example in enumerate(batch):
                if example["step"] != opening_step:
                    continue
                opening += 1
                factor = example["trace"]["commands"][0]["factors"][0]
                choices = tuple(factor["choices"])
                scores, _, _, _, _ = learner.graph.factor(context[row], factor["path"], choices,
                                                          1.0, stop_only=False, as_of_step=example["step"])
                probs = scores.exp().tolist()
                input_rows.append({'input_sha256': example['pack']['content_sha256'],
                                   'choices': list(choices), 'mask': example['trace']['sampling_masks'][0][0],
                                   'probabilities': probs, 'target': choices[factor['index']]})
                order = sorted(range(len(probs)), key=lambda i: -probs[i])
                correct += int(order[0] == factor["index"])
                probabilities.append(probs[factor["index"]])
                ranks.append(order.index(factor["index"])+1)
                name = str(choices[order[0]])
                argmax_ops[name] = argmax_ops.get(name, 0)+1
    result = {"examples": len(examples), "factors": total_factors, "factor_nll": -total_logp/total_factors,
              "opening_examples": opening, "greedy_first_op_accuracy": None,
              "mean_target_first_op_probability": None, "target_first_op_probability_spread": None,
              "mean_target_first_op_rank": None, "greedy_first_op_distribution": None,
              "first_op_input_diagnostics": first_op_input_groups(input_rows)}
    if opening:
        result.update(greedy_first_op_accuracy=correct/opening,
                      mean_target_first_op_probability=sum(probabilities)/opening,
                      target_first_op_probability_spread=max(probabilities)-min(probabilities),
                      mean_target_first_op_rank=sum(ranks)/opening,
                      greedy_first_op_distribution=dict(sorted(argmax_ops.items())))
    return result


# ------------------------------------------------------------------ the fit
def fit(output, *, parent_dir, model_sha256, optimizer_sha256, corpus, updates, batch_size, accumulation,
        max_commands, holdout_fraction=0.1, holdout_key="program_id", shuffle_seed=None,
        steps=(0, 144, 288, 432, 576), examples_limit=0, metric_every=25, planned_optimizer_steps=None,
        learning_rate=None, entropy_coefficient=None, gpu_fraction=0.25, reserve_bytes=450_000_000, verify_child=True,
        agreement_atol=1e-5, agreement_examples=4, rebind_source_pins_allowlist=(), rebind_reason="",
        loss_reduction="factor", preferences_path=None, preferences_sha256=None, preference_coefficient=None,
        preference_mode=None):
    import torch
    if loss_reduction not in LOSS_REDUCTIONS:
        raise ValueError("known supervised loss reduction required")
    preference_config = None
    if any(v is not None for v in (preferences_path, preferences_sha256, preference_coefficient, preference_mode)):
        if any(v is None for v in (preferences_path, preferences_sha256, preference_coefficient)):
            raise ValueError("preference data, hash and coefficient must be explicit together")
        if accumulation != 1 or holdout_fraction != 0.0 or examples_limit != 0 or loss_reduction != "factor":
            raise ValueError("preference stage requires full corpus, no holdout, factor BC and accumulation1")
        preferences_path = Path(preferences_path).absolute()
        if lab.sha(preferences_path) != preferences_sha256: raise ValueError("preference data hash mismatch")
        preference_config = {"schema": "farm-native-program-preferences-v1", "data_sha256": preferences_sha256,
                             "reference_model_sha256": model_sha256, "coefficient": preference_coefficient}
        if preference_mode is not None:
            if preference_mode not in ("program", "first_stop"):
                raise ValueError("known native preference mode required")
            preference_config["mode"] = preference_mode
    for name, value, upper in (("updates", updates, 4096), ("batch", batch_size, 64),
                               ("accumulation", accumulation, 128), ("max_commands", max_commands, actions.MAX_COMMANDS)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError("bounded positive " + name + " required")
    if type(gpu_fraction) is not float or not 0.05 <= gpu_fraction <= 1.0:
        raise ValueError("explicit GPU memory fraction in [.05,1] required")
    corpus = Path(corpus).absolute()
    parent = io.checkpoint(ROOT, parent_dir)
    if parent["model_sha256"] != model_sha256 or parent["optimizer_sha256"] != optimizer_sha256:
        raise ValueError("exact current parent model/Adam hashes required")
    storage = campaign._storage_check(reserve_bytes)
    if not storage.get("admissible"):
        raise ValueError("program BC checkpoint output headroom refused")
    output = lab.output_directory(output)
    torch.cuda.set_per_process_memory_fraction(gpu_fraction)

    # ---- resident learner, constructed exactly the way bc-current does
    parent_plan = io.read(io.path(ROOT, parent_dir)/"PLAN.json")
    parent_result = io.read(io.path(ROOT, parent_dir)/"RESULT.json")
    game = copy.deepcopy(parent_plan["execution"].get("resume_game_execution", parent_plan["execution"]))
    # Refuse a drifted pinned closure unless the owner rebound it explicitly; may
    # rewrite only the recorded adapter identity inside this in-memory game copy.
    source_transition = rebind_source_pins(parent_plan, parent_result, game,
                                           allowlist=rebind_source_pins_allowlist, reason=rebind_reason)
    clock = game.get("economic_clock", {})
    learner = campaign.Learner(parent_dir, parent, checkpoint_blocks=True,
        opening_only=clock.get("publication") == "ATOMIC_OPENING_ONLY_NO_LATER_ECONOMIC_REVISION",
        plan_learning=game.get("plan_learning"), shop_revision_every=clock.get("shop_revision_every", 0),
        full_plan=bool(game.get("full_plan")))
    if learner.execution != game:
        raise ValueError("program BC must retain the exact parent game execution contract")
    learner.execution = campaign.supervised_execution(game, loss_reduction=loss_reduction, preferences=preference_config)
    original = torch.load(io.path(ROOT, parent_dir)/"optimizer.pt", map_location="cpu", weights_only=True)
    entry = {"model": lab._state_digest(learner.graph.state_dict()),
             "main_adam": lab._state_digest(learner.optimizer.state_dict()),
             "cpu_rng": lab._state_digest(torch.get_rng_state()),
             "cuda_rng": lab._state_digest(torch.cuda.get_rng_state_all())}
    # A source-migrated parent keeps its ORIGINAL pins inside optimizer.pt; the recorded
    # transition says what they must be, and without one this is the same comparison.
    joined, _ = migration.joined_identity(original, parent_plan, parent_result, learner.manifest,
                                          live_pins=campaign.sources())
    if original["source_pins"] != joined["source_pins"]:
        raise ValueError("parent optimizer/PLAN source pins disagree; refuse to rebind an inconsistent parent")
    if (entry["main_adam"] != lab._state_digest(original["optimizer"])
            or entry["cpu_rng"] != lab._state_digest(original["torch_rng"])
            or entry["cuda_rng"] != lab._state_digest(original["cuda_rng"])):
        raise ValueError("exact optimizer/RNG entry restoration required")
    auxiliary = None if learner.auxiliary is None else {
        "heads": lab._state_digest(learner.auxiliary.state_dict()),
        "adam": lab._state_digest(learner.aux_optimizer.state_dict())}
    if auxiliary is not None and auxiliary != {
            "heads": lab._state_digest(original["plan_learning_state"]["heads"]),
            "adam": lab._state_digest(original["plan_learning_state"]["optimizer"])}:
        raise ValueError("exact auxiliary entry restoration required")
    del original

    # ---- optional explicit schedule override, written into the published child
    inherited = copy.deepcopy(learner.settings)
    settings, schedule_override = override_settings(inherited, planned_optimizer_steps=planned_optimizer_steps,
                                                    learning_rate=learning_rate, entropy_coefficient=entropy_coefficient)
    if schedule_override is not None:
        learner.settings = settings
    validate_settings(learner.settings)
    if learner.age + updates > learner.settings["planned_optimizer_steps"]:
        raise ValueError("declared updates must fit the effective schedule; no silent extension")

    architecture_sha256 = policy.architecture(learner.graph.config)["content_sha256"]
    shuffle_seed = learner.settings["seed"] if shuffle_seed is None else shuffle_seed
    if type(shuffle_seed) is not int or shuffle_seed < 0:
        raise ValueError("nonnegative integer shuffle seed required")

    started = time.monotonic()
    examples, corpus_stats = build_examples(corpus, steps=tuple(steps), max_commands=max_commands,
                                            architecture_sha256=architecture_sha256, limit=examples_limit,
                                            progress=500)
    train, holdout, split = split_holdout(examples, fraction=holdout_fraction, key=holdout_key, seed=shuffle_seed)
    order = list(range(len(train)))
    random.Random(shuffle_seed+1).shuffle(order)
    preference_receipt = None
    if preference_config is not None:
        if len(train) % batch_size: raise ValueError("complete fixed preference encoder batches required")
        data = json.loads(preferences_path.read_text())
        preference_receipt = attach_preferences(train, data, lab.sha(corpus), architecture_sha256, max_commands,
                                                first_stop=preference_mode == "first_stop")
        reference_rows = []; reference_start = time.monotonic()
        with torch.no_grad():
            for offset in range(0, len(order), batch_size):
                batch = [train[i] for i in order[offset:offset+batch_size]]
                context, _ = encode_batch(learner, batch)
                for example, logps in zip(batch, preference_logps(learner, context, batch), strict=True):
                    if not bool(torch.isfinite(logps).all()): raise ValueError("nonfinite entry reference")
                    example["preference"]["reference"] = logps.cpu().tolist()
                    reference_rows.append({"game_ordinal": example["game_ordinal"], "step": example["step"],
                                           "program_logps": logps.cpu().tolist()})
        reference = {"reference_model_sha256": model_sha256, "rows": reference_rows,
                     "batch_order": order, "batch_size": batch_size, "seconds": time.monotonic()-reference_start,
                     "scope": "ENTRY_PARENT_TEACHER_FORCED_SCORES; NO_SAMPLING_OR_UPDATES"}
        lab.write(output/"REFERENCE.json", reference)
        preference_receipt.update(config=preference_config, path=str(preferences_path.relative_to(ROOT)),
                                  reference_sha256=lab.sha(output/"REFERENCE.json"))
    counts = {"corpus_sha256": lab.sha(corpus), "corpus_path": str(corpus.relative_to(ROOT)),
              "corpus_schema": CORPUS_SCHEMA, **corpus_stats, "examples": len(examples),
              "train_examples": len(train), "holdout_examples": len(holdout),
              "train_factors": sum(e["factors"] for e in train),
              "holdout_factors": sum(e["factors"] for e in holdout),
              "examples_by_step": {str(s): sum(1 for e in examples if e["step"] == s) for s in sorted(set(steps))},
              "split": split}

    loss_description = ("FACTOR_NORMALISED_BC_PLUS_COEFFICIENT_TIMES_MEAN_STATE_REFERENCE_RELATIVE_STRICT_WDL_PREFERENCE_V1"
                        if preference_config is not None else LOSS_REDUCTIONS[loss_reduction])
    if preference_mode == "first_stop":
        loss_description = "FACTOR_NORMALISED_BC_PLUS_COEFFICIENT_TIMES_MEAN_STATE_NATIVE_FIRST_STOP_BCE_V1"
    plan = {"schema": SCHEMA, "execution": learner.execution, "source_pins": campaign.sources(),
            "program_bc_source_pins": own_sources(), "parent": parent,
            "environment": {**learner.profile, "bitsandbytes": str(learner.bnb.__version__)},
            "schedule": learner.settings, "inherited_schedule": inherited,
            "schedule_override": schedule_override,
            "start_update": learner.age, "requested_updates": updates,
            "corpus": counts, "targets": TARGET_SEMANTICS,
            "target_trace_schema": policy.REMAINING_TRACE_SCHEMA, "max_commands": max_commands,
            "decision_steps": sorted(set(steps)),
            "input_adapter": plan_features.identity(), "batch_mode": campaign.BATCH_MODE,
            "pad_multiple": campaign.PAD_MULTIPLE,
            "scorer": "FAST_PROFILE_PADDED_POSITION_REPLAY_WITH_FIXED_PRIOR_AND_REMAINING_WINDOW_MASKS",
            "loss": loss_description, "loss_reduction": loss_reduction,
            "batch_size": batch_size, "accumulation": accumulation,
            "examples_per_optimizer_step": batch_size*accumulation,
            "gpu_memory_fraction": gpu_fraction, "reserve_bytes": reserve_bytes,
            "entry_restoration": entry, "preserved_auxiliary": auxiliary,
            "resume_binding": learner.origin, "shuffle_seed": shuffle_seed,
            "source_transition": source_transition,
            "sample_order": "DETERMINISTIC_SEEDED_PERMUTATION_THEN_CYCLING",
            "no_optimizer_reset": True, "no_auxiliary_updates": True, "selfplay_games": 0,
            "probe_scope": "HELDOUT_GROUP_FACTOR_NLL_AND_GREEDY_FIRST_OP; NOT_STRENGTH"}
    if preference_receipt is not None:
        plan["preferences"] = preference_receipt
    lab.write(output/"PLAN.json", plan)
    lab.write(output/"STORAGE.json", storage)
    admission = campaign.bound({"schema": SCHEMA+"-admission", "corpus_sha256": counts["corpus_sha256"],
        "corpus_path": counts["corpus_path"], "corpus_games": counts["games"],
        "train_examples": len(train), "holdout_examples": len(holdout),
        "holdout": split, "targets": TARGET_SEMANTICS,
        "terminal_training_labels": 0 if preference_receipt is None else preference_receipt["strict_pairs"],
        "selfplay_games": 0, "source_transition": source_transition,
        "input": "COLLECTOR_IDENTICAL_RESTORED_SNAPSHOT_FULL_PLAN_V4_PACK_PAD32"})
    lab.write(output/"ADMISSION.json", admission)

    before = {"graph": {k: v.detach().cpu().clone() for k, v in learner.graph.state_dict().items()},
              "optimizer": campaign.legacy._cpu_copy(learner.optimizer.state_dict()), "age": learner.age,
              "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
    history, index = [], 0
    try:
        probe = train[:min(agreement_examples, len(train))]
        agreement, padded_logp, scalar_logp = replay_agreement(learner, probe)
        if not (agreement <= agreement_atol):
            raise ValueError(f"fast/scalar replay disagreement {agreement} above {agreement_atol}")
        initial = holdout_metrics(learner, holdout, batch_size=batch_size) if holdout else None
        for _ in range(updates):
            tick = time.monotonic()
            learner.optimizer.zero_grad(set_to_none=True)
            total_logp, total_score, total_factors, total_commands, examples_used = 0.0, 0.0, 0, 0, 0
            for _ in range(accumulation):
                batch = [train[order[(index+j) % len(order)]] for j in range(batch_size)]
                index += batch_size
                preference_loss = None
                if preference_config is None:
                    logp, score = supervised_scores(learner, batch, loss_reduction)
                else:
                    context, logp, _ = program_logp(learner, batch)
                    alternatives = preference_logps(learner, context, batch)
                    preference_loss = (first_stop_terms(alternatives, [e["preference"]["keep_index"] for e in batch])
                        if preference_mode == "first_stop" else preference_terms(alternatives,
                        [e["preference"]["reference"] for e in batch], [e["preference"]["inferior"] for e in batch]))
                    score = logp - preference_coefficient * (sum(e["factors"] for e in batch)/len(batch)) * preference_loss
                if not bool(torch.isfinite(score).all()):
                    raise ValueError("nonfinite program log-probability")
                ((-score.sum())/(batch_size*accumulation)).backward()
                total_logp += float(logp.detach().sum())
                total_score += float(score.detach().sum())
                total_factors += sum(e["factors"] for e in batch)
                total_commands += sum(e["commands"] for e in batch)
                examples_used += len(batch)
            denominator = total_factors if loss_reduction == "factor" else total_commands
            normalize_accumulated_gradients(learner.graph.parameters(),
                temporary_denominator=batch_size*accumulation, actual_denominator=denominator)
            loss = -total_score/denominator
            norm = torch.nn.utils.clip_grad_norm_(learner.graph.parameters(),
                                                  learner.settings["gradient_clip"], error_if_nonfinite=True)
            lr = learning_rate_at(learner.age+1, learner.settings)
            for group in learner.optimizer.param_groups:
                group["lr"] = lr
            learner.optimizer.step(); torch.cuda.synchronize(); learner.age += 1
            row = {"global_update": learner.age, "factor_nll": -total_logp/total_factors,
                   "objective_loss": loss, "loss_reduction": loss_reduction,
                   "loss_denominator": denominator, "commands": total_commands, "gradient_norm": float(norm),
                   "learning_rate": lr, "examples": examples_used, "factors": total_factors,
                   "seconds": time.monotonic()-tick, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())}
            if preference_loss is not None:
                row["first_stop_loss" if preference_mode == "first_stop" else "preference_loss"] = float(preference_loss.detach().mean())
                row["preference_programs"] = sum(len(e["preference"]["traces"]) for e in batch)
                row["strict_preference_pairs"] = sum(len(e["preference"]["inferior"]) for e in batch)
            if metric_every and holdout and len(history)+1 < updates and (len(history)+1) % metric_every == 0:
                row["holdout"] = holdout_metrics(learner, holdout, batch_size=batch_size)
            history.append(row); print(lab.canonical(row), flush=True)
        final = holdout_metrics(learner, holdout, batch_size=batch_size) if holdout else None
        if auxiliary is not None and auxiliary != {
                "heads": lab._state_digest(learner.auxiliary.state_dict()),
                "adam": lab._state_digest(learner.aux_optimizer.state_dict())}:
            raise ValueError("program BC unexpectedly changed auxiliary heads or Adam")
        if lab.sha(corpus) != counts["corpus_sha256"] or campaign.sources() != plan["source_pins"]:
            raise ValueError("corpus or source closure changed before publication")
        if preference_config is not None and lab.sha(preferences_path) != preferences_sha256:
            raise ValueError("preference labels changed before publication")
        changed = [name for name, tensor in learner.graph.state_dict().items()
                   if not torch.equal(tensor.detach().cpu(), before["graph"][name])]
        if not changed:
            raise ValueError("no-op fit is not a committed training update")
        receipt = {"schema": SCHEMA+"-receipt", "global_optimizer_steps_before": before["age"],
                   "global_optimizer_steps_after": learner.age, "optimizer_updates": updates,
                   "corpus": counts, "replay_agreement_max_abs": agreement,
                   "replay_agreement_atol": agreement_atol, "replay_agreement_examples": len(probe),
                   "replay_agreement_padded_logp": padded_logp, "replay_agreement_scalar_logp": scalar_logp,
                   "initial_holdout": initial, "final_holdout": final, "history": history,
                   "changed_tensor_names": changed, "auxiliary_updates": 0, "selfplay_games": 0,
                   "schedule_override": schedule_override, "source_transition": source_transition,
                   "loss": loss_description, "loss_reduction": loss_reduction,
                   "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "seconds": time.monotonic()-started}
        manifest = learner.publish(output, receipt, lab.sha(output/"PLAN.json"), admission, [])
        result = {"schema": SCHEMA, "status": io.BC_STATUS, "plan_sha256": lab.sha(output/"PLAN.json"),
                  "source_pins": campaign.sources(), "program_bc_source_pins": own_sources(),
                  "before_model_sha256": parent["model_sha256"], "after_model_sha256": lab.sha(output/"model.pt"),
                  "optimizer_sha256": lab.sha(output/"optimizer.pt"), "manifest": manifest["content_sha256"],
                  "optimizer_updates": updates, "global_optimizer_updates": learner.age, "selfplay_games": 0,
                  "corpus": counts, "auxiliary_preserved": auxiliary, "receipt": receipt,
                  "source_transition": source_transition,
                  "population_strength_established": False, "rollout_iteration_unchanged": True}
        lab.write(output/"RESULT.json", result)
        lab.write(output/"RECEIPT.json", receipt)
        checked = io.checkpoint(ROOT, output)
        validation = {"sustain_checkpoint": checked}
        if verify_child:
            del learner
            torch.cuda.empty_cache()
            child = campaign.Learner(io.relative(ROOT, output), checked, checkpoint_blocks=True,
                opening_only=clock.get("publication") == "ATOMIC_OPENING_ONLY_NO_LATER_ECONOMIC_REVISION",
                plan_learning=game.get("plan_learning"), shop_revision_every=clock.get("shop_revision_every", 0),
                full_plan=bool(game.get("full_plan")))
            validation.update(driver_learner_resume={"age": child.age, "settings": child.settings,
                "execution_target": child.execution, "profile_origin": child.origin,
                "auxiliary_continuation": child.auxiliary_continuation,
                "validate_parent": "PASSED"})
            del child
            torch.cuda.empty_cache()
        lab.write(output/"VALIDATION.json", validation)
        result["validation"] = validation
        return result
    except BaseException as error:
        learner.optimizer.zero_grad(set_to_none=True)
        learner.graph.load_state_dict(before["graph"], strict=True)
        learner.optimizer.load_state_dict(before["optimizer"])
        torch.set_rng_state(before["cpu_rng"]); torch.cuda.set_rng_state_all(before["cuda_rng"])
        learner.age = before["age"]
        lab.write(output/"FAILURE.json", {"error": str(error), "attempted_updates": len(history),
                                          "committed_updates": 0, "rollback_parent": parent, "history": history})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", default=DEFAULT_PARENT)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--optimizer-sha256", required=True)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--loss-reduction", choices=tuple(LOSS_REDUCTIONS), default="factor",
                        help="factor: global factor mean; command: global command mean of within-command factor NLL, including STOP")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--preferences", dest="preferences_path")
    parser.add_argument("--preferences-sha256")
    parser.add_argument("--preference-coefficient", type=float)
    parser.add_argument("--preference-mode", choices=("program", "first_stop"))
    parser.add_argument("--holdout-key", choices=("program_id", "seed"), default="program_id")
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--max-commands", type=int, required=True,
                        help="explicit slots including final STOP; saved comparison profile uses 64")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 144, 288, 432, 576])
    parser.add_argument("--examples-limit", type=int, default=0)
    parser.add_argument("--metric-every", type=int, default=25)
    parser.add_argument("--planned-optimizer-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coefficient", type=float, default=None,
                        help="schedule override written into the child's settings; 0.0 turns structural entropy off")
    parser.add_argument("--gpu-fraction", type=float, default=0.25)
    parser.add_argument("--reserve-bytes", type=int, default=450_000_000)
    parser.add_argument("--no-verify-child", dest="verify_child", action="store_false")
    parser.add_argument("--rebind-source-pins", nargs="*", default=[], metavar="PATH",
                        help="repository-relative pinned files whose sha256 may differ from the parent's record")
    parser.add_argument("--rebind-reason", default="",
                        help="required with --rebind-source-pins; recorded in PLAN/RECEIPT/RESULT")
    args = parser.parse_args(argv)
    result = fit(args.output, parent_dir=args.parent, model_sha256=args.model_sha256,
                 optimizer_sha256=args.optimizer_sha256, corpus=args.corpus, updates=args.updates,
                 batch_size=args.batch_size, accumulation=args.accumulation, loss_reduction=args.loss_reduction,
                 holdout_fraction=args.holdout_fraction, holdout_key=args.holdout_key,
                 shuffle_seed=args.shuffle_seed, max_commands=args.max_commands, steps=tuple(args.steps),
                 examples_limit=args.examples_limit, metric_every=args.metric_every,
                 planned_optimizer_steps=args.planned_optimizer_steps, learning_rate=args.learning_rate,
                 entropy_coefficient=args.entropy_coefficient,
                 gpu_fraction=args.gpu_fraction, reserve_bytes=args.reserve_bytes,
                 verify_child=args.verify_child,
                 rebind_source_pins_allowlist=args.rebind_source_pins, rebind_reason=args.rebind_reason,
                 preferences_path=args.preferences_path, preferences_sha256=args.preferences_sha256,
                 preference_coefficient=args.preference_coefficient, preference_mode=args.preference_mode)
    print(lab.canonical({k: v for k, v in result.items() if k != "receipt"}), flush=True)
    return result


if __name__ == "__main__":
    main()
