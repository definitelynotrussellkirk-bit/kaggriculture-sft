"""Prepare a parent-checked GRPO plan; execute only with EXECUTE=1.

Preparation writes PLAN.json and LAUNCH.json. The objective and command budget
are required arguments; a prepared plan is not evidence of a running trainer.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SRC))
from farm_neat.training import campaign as c
from farm_neat.tooling import runtime


def launch_spec(output, plan_sha, owner, prepared, auxiliary_transition):
    """Literal argv and only the declared environment overrides go into the receipt."""
    spec = runtime.lock()
    py = str(runtime.interpreter(spec, section='training'))
    argv = [py, '-B', 'RUN.py', 'storage', 'run', '--reserve-bytes', '850000000',
            '--timeout', '4500', '--', py, '-B', 'src/farm_neat/training/campaign.py',
            'run', '--output', str(output), '--plan-sha256', plan_sha, '--owner', owner, '--execute']
    return {'schema': 'farm-grpo-launch-v2', 'status': 'PREPARED', 'argv': argv,
            'environment': spec['training']['environment'], 'cwd': str(_SRC.parent),
            'plan_file_sha256': plan_sha, 'parent': prepared['parent'],
            'execution': prepared['execution'], 'command_budget': prepared['command_budget'],
            'forced_stop': 'FINAL_SLOT_IS_STOP_ONLY; BUDGET_INCLUDES_STOP',
            'decision_steps': prepared['decision_steps'], 'temperature': 1.0,
            'bc_rehearsal': False, 'auxiliary_continuation': auxiliary_transition,
            'gradient_recipients': 'ACTOR_AND_SHARED_REPRESENTATION_FROM_ACTOR_CRITIC_FUTURE_LOSSES',
            'retention': {'purpose': 'resolved launch and owned supervisor identity',
                          'remove_when': 'no continuation or experiment evidence dependency'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--group-offset', type=int, default=2000)
    parser.add_argument('--command-budget', type=int, required=True)
    parser.add_argument('--plan-learning', type=Path, required=True,
                        help='explicit objective JSON, checked against the parent before preparation')
    args = parser.parse_args(argv)
    if not 2 <= args.command_budget <= c.actions.MAX_COMMANDS:
        parser.error('unsupported command budget including STOP')
    if not 1 <= args.rounds <= 1024 or args.group_offset < 0:
        parser.error('positive bounded rounds and nonnegative group offset required')
    os.chdir(_SRC.parent)
    parent, output = args.parent, args.output
    child = c.io.checkpoint(_SRC.parent, parent)
    learning = c.validate_plan_learning(c.io.read(args.plan_learning))
    # Inspect real retained auxiliary state on CPU; never fabricate a join from prose.
    import torch
    saved = torch.load(parent/'optimizer.pt', map_location='cpu', weights_only=True)
    transition = c.auxiliary_resume(saved.get('plan_learning_state'), saved['execution'],
                                    learning, saved['updates'])
    del saved
    owner = 'farmneat-pass-grpo'
    prepared = c.prepare(output, parent=str(parent), model_sha256=child['model_sha256'],
        optimizer_sha256=child['optimizer_sha256'], owner=owner,
        seed_start=int(os.environ.get('SEED_START', 6000000)),
        sampler_start=int(os.environ.get('SAMPLER_START', 8000000)),
        games_per_round=200, processes=2, rounds=args.rounds,
        exclusive_seeds=True, updates_per_round=1, accumulation_cohorts=50, stack_cohorts=2,
        metric_every=0, canary_cohorts=2, command_budget=args.command_budget,
        stage_seconds=1200, max_seconds=4200, reserve_bytes=450000000, witness_workers=2,
        mode='LOCAL', retain_children=1, worlds_per_process=20, overlap=False,
        opening_only=False, plan_learning=learning, shop_revision_every=2, full_plan=True,
        group_offset_start=args.group_offset)
    launch = launch_spec(output, prepared['plan_file_sha256'], owner, prepared['plan'], transition)
    print(json.dumps(launch, indent=2)); print(shlex.join(launch['argv']))
    c.io.write(output/'LAUNCH.json', launch)
    if os.environ.get('EXECUTE') == '1':
        with (output/'launch.log').open('ab') as log:
            process = subprocess.Popen(launch['argv'], env={**os.environ, **launch['environment']},
                                       cwd=launch['cwd'], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        launch.update(status='SUPERVISOR_STARTED', supervisor_pid=process.pid,
                      progress_verified=False, completion_verified=False)
        staged = output/'LAUNCH.json.tmp'
        c.io.write(staged, launch)
        os.replace(staged, output/'LAUNCH.json')
        print(json.dumps({'status': launch['status'], 'supervisor_pid': process.pid,
                          'progress_verified': False}))


if __name__ == '__main__':
    main()
