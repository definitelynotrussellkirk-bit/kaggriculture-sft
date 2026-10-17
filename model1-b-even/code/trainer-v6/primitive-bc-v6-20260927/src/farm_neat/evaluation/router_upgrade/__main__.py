"""Explicit-output Router-upgrade builder, diagnostics, and bound D06 panel decision.

The historical D06 maintainer panel binds router_upgrade.__init__ byte-for-byte,
so its post-panel consumer lives here rather than mutating that audited builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from farm_neat.evaluation import router_upgrade as upgrade
from farm_neat.evaluation.common import equal, file_sha, read, write_new
from farm_neat.evaluation.router_upgrade import PATCH, build, probes, smoke

ROOT = upgrade.ROOT
BASE_SELECTION_CLAIM = '2fc085811f0bc32c13b2252c40539bef5fcab9a1786d61cc6f229b42c1e1fa66'
PANEL_CLAIM = '3627d274e6b029d24ae76fdc244b0b6d990daa600a16ea1a1a6db041a10216e9'
PANEL_SUMMARY = ROOT / 'work/runs/reviewer-play/router-upgrade-panel/panel/summary.json'
PANEL_MANIFEST = ROOT / 'work/runs/reviewer-play/router-upgrade-panel/panel/manifest.json'
PANEL_BUILD = ROOT / 'work/runs/reviewer-play/router-upgrade-panel/d06_candidate.build.json'
BASE_CLAIM_PATH = ROOT / 'project/registries/search-ledger' / f'{BASE_SELECTION_CLAIM}.json'
PANEL_CLAIM_PATH = ROOT / 'project/registries/search-ledger' / f'{PANEL_CLAIM}.json'
PANEL_SUMMARY_SHA256 = 'c248931470bdc07d9d8b2d26d01c3359efae70d40d8e46b42171673c47771085'
PANEL_MANIFEST_SHA256 = 'ecd8eddba9961f0fb0a3ab8b66da87a112f60b6a6175a69ffe10f94e0ee41078'
PANEL_BUILD_SHA256 = 'fd2d1b38a953542de648b582ed834bd8da880f3788620b296adda850bd56fe02'
HISTORICAL_BUILDER_SHA256 = '0c660adfdd38cff454cdf9a4037b8b1bca9aa1505531ed80d879efa592c96236'
D06_ARTIFACT_SHA256 = 'eac0f504fb8d8fc10fd7b6a5e5fc6874e1e88e8ad01257b68a1e47c1f3f0fb54'
D06_ACTOR_IDENTITY_SHA256 = 'cff17549b7cb18a325d9226dd0f8c655bca6f10ff07857362c2166f7a9ce33c2'
DECISION_SCHEMA = 'farm-router-upgrade-panel-decision-v1'


def _pinned(path: Path, expected: str, label: str, *, binding: bool = True) -> dict:
    equal(file_sha(path), expected, label)
    return read(path, binding=binding)


def _claim_evidence(claim: dict) -> dict[str, str]:
    return {row['path']: row['sha256'] for row in claim.get('evidence', [])
            if 'path' in row and 'sha256' in row}


def verified_panel_receipt() -> dict:
    """Verify the completed maintainer panel and return D06's keep/drop receipt.

    This launches no games. The keep rule is a strictly positive parent-paired
    interval; an interval containing zero or an entirely negative interval drops
    the patch.
    """
    equal(file_sha(Path(upgrade.__file__)), HISTORICAL_BUILDER_SHA256, 'historical D06 builder')
    base_claim = _pinned(BASE_CLAIM_PATH, BASE_SELECTION_CLAIM,
                         'K0006 base-selection claim', binding=False)
    panel_claim = _pinned(PANEL_CLAIM_PATH, PANEL_CLAIM, 'D06 panel claim', binding=False)
    summary = _pinned(PANEL_SUMMARY, PANEL_SUMMARY_SHA256, 'panel summary')
    manifest = _pinned(PANEL_MANIFEST, PANEL_MANIFEST_SHA256, 'panel manifest')
    built = _pinned(PANEL_BUILD, PANEL_BUILD_SHA256, 'panel candidate build')

    equal(base_claim['key'], 'K-ROUTER-FAMILY-K0006-PANEL', 'base selection key')
    equal(panel_claim['key'], 'K-ROUTER-D06-NO-POSITIVE-INTERVAL', 'panel claim key')
    evidence = _claim_evidence(panel_claim)
    equal(evidence[str(PANEL_SUMMARY.relative_to(ROOT))], PANEL_SUMMARY_SHA256,
          'claim -> panel summary')
    equal(evidence[str(PANEL_MANIFEST.relative_to(ROOT))], PANEL_MANIFEST_SHA256,
          'claim -> panel manifest')
    equal(evidence[str(PANEL_BUILD.relative_to(ROOT))], PANEL_BUILD_SHA256,
          'claim -> candidate build')

    equal(summary['status'], 'COMPLETE', 'panel completion')
    equal(summary['games_planned'], 1232, 'panel planned games')
    equal(summary['physical_games_attempted'], 1232, 'panel attempts')
    equal(summary['physical_games_completed'], 1232, 'panel completed games')
    equal(summary['synthetic_attempts'], 0, 'panel synthetic attempts')
    equal(summary['legacy_gate'],
          {'matched': 80, 'required': 80, 'status': 'EXACT_80_ROW_REPLAY'}, 'legacy gate')
    equal(summary['min_rival_cash'], 3000.0, 'minimum rival cash')
    equal(manifest['games_planned'], 1232, 'manifest planned games')
    equal(manifest['candidate'], upgrade.CANDIDATE_REF, 'manifest candidate')
    equal(manifest['parent'], 'k0006', 'manifest parent')
    equal(manifest['rivals'], ['router', 'k0006', 'tape0', 'pass'], 'manifest rivals')

    actor = manifest['actors'][upgrade.CANDIDATE_REF]
    declared = actor['declared']
    equal(declared['kind'], 'EXACT_PACKAGED_SOURCE_EXEC', 'candidate execution kind')
    equal(declared['patches'], [upgrade.PATCH], 'candidate selected patch')
    equal(declared['map_row'], 'D06', 'candidate map row')
    equal(declared['claim'], upgrade.CLAIM, 'candidate tuning claim')
    equal(declared['parent_sha256'], upgrade.PARENT_SHA256, 'candidate parent')
    equal(declared['artifact_sha256'], D06_ARTIFACT_SHA256, 'candidate artifact identity')
    equal(built['artifact_sha256'], D06_ARTIFACT_SHA256, 'candidate build artifact')
    equal(built['builder_sha256'], HISTORICAL_BUILDER_SHA256, 'candidate builder identity')
    equal(built['patches'], [upgrade.PATCH], 'candidate build patch')
    equal(summary['identity_hashes'][upgrade.CANDIDATE_REF], D06_ACTOR_IDENTITY_SHA256,
          'candidate actor identity')
    equal(actor['content_sha256'], D06_ACTOR_IDENTITY_SHA256, 'manifest actor identity')

    reproduced, metadata = upgrade.render((upgrade.PATCH,))
    equal(hashlib.sha256(reproduced).hexdigest(), D06_ARTIFACT_SHA256,
          'reproduced audited artifact')
    equal(metadata['patches'], [upgrade.PATCH], 'reproduced patch metadata')

    paired = summary['paired']['k0006']
    equal(paired['identity_hashes'][upgrade.CANDIDATE_REF], D06_ACTOR_IDENTITY_SHA256,
          'paired candidate identity')
    equal(paired['identity_hashes']['k0006'], summary['identity_hashes']['k0006'],
          'paired parent identity')
    equal(paired['interpretation'], 'NO_POSITIVE_FIXED_WORLD_PAIRED_INTERVAL',
          'parent-paired interpretation')
    equal(paired['worlds_count'], 72, 'registered world count')
    pair64 = {row['world'] for row in paired['worlds'] if row.get('cohort') == 'pair64'}
    equal(len(pair64), 64, 'pair64 world count')
    seat_widths = sorted({len(row['paired_margins_by_seat']) for row in paired['worlds']})
    equal(seat_widths, [2], 'paired seat coverage')
    seats, = seat_widths
    low, high = paired['world_bootstrap_ci95']
    if low > 0:
        decision = 'KEEP_POSITIVE_INTERVAL'
        kept, dropped = [upgrade.PATCH], []
    elif high < 0:
        decision = 'DROP_NEGATIVE_INTERVAL'
        kept, dropped = [], [upgrade.PATCH]
    else:
        decision = 'DROP_INTERVAL_INCLUDES_ZERO'
        kept, dropped = [], [upgrade.PATCH]
    equal(decision, 'DROP_INTERVAL_INCLUDES_ZERO', 'accepted D06 keep/drop decision')

    return {
        'schema': DECISION_SCHEMA,
        'patch': upgrade.PATCH,
        'map_row': 'D06',
        'candidate_quantity': 6,
        'base': 'k0006',
        'base_selection_claim': BASE_SELECTION_CLAIM,
        'panel_claim': PANEL_CLAIM,
        'decision': decision,
        'kept_patches': kept,
        'dropped_patches': dropped,
        'paired_mean': paired['paired_mean'],
        'world_bootstrap_ci95': [low, high],
        'registered_worlds': paired['worlds_count'],
        'pair64_worlds': len(pair64),
        'seats': seats,
        'rivals': list(manifest['rivals']),
        'games_completed': summary['physical_games_completed'],
        'synthetic_attempts': summary['synthetic_attempts'],
        'legacy_gate': summary['legacy_gate']['status'],
        'minimum_rival_cash': summary['min_rival_cash'],
        'candidate_execution_kind': declared['kind'],
        'candidate_artifact_sha256': D06_ARTIFACT_SHA256,
        'identity_hashes': dict(summary['identity_hashes']),
        'source_hashes': {
            str(PANEL_SUMMARY.relative_to(ROOT)): PANEL_SUMMARY_SHA256,
            str(PANEL_MANIFEST.relative_to(ROOT)): PANEL_MANIFEST_SHA256,
            str(PANEL_BUILD.relative_to(ROOT)): PANEL_BUILD_SHA256,
            str(PANEL_CLAIM_PATH.relative_to(ROOT)): PANEL_CLAIM,
            str(BASE_CLAIM_PATH.relative_to(ROOT)): BASE_SELECTION_CLAIM,
            'src/farm_neat/evaluation/router_upgrade/__init__.py': HISTORICAL_BUILDER_SHA256,
        },
        'promotion_authority': False,
        'interpretation': ('D06 is a negative/inconclusive patch and remains disabled; '
                           'the frozen K0006 parent is retained.'),
    }


def build_final(output) -> dict:
    """Build the post-panel artifact and a durable decision sidecar."""
    decision = verified_panel_receipt()
    path = upgrade._output_path(output)
    sidecar = path.with_suffix('.decision.json')
    upgrade._output_path(sidecar)
    if sidecar.exists():
        raise FileExistsError(str(sidecar))
    built = upgrade.build(path, patches=tuple(decision['kept_patches']))
    equal(built['patches'], [], 'final kept patch set')
    expected, _ = upgrade.render(())
    equal(built['artifact_sha256'], hashlib.sha256(expected).hexdigest(), 'final artifact bytes')
    return write_new(sidecar, {
        **decision,
        'final_artifact': str(path),
        'final_artifact_sha256': built['artifact_sha256'],
        'final_build': built,
        'final_behavior': 'FROZEN_K0006_WITH_D06_DROPPED',
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build')
    builder.add_argument('--output', required=True)
    builder.add_argument('--patch', action='append', default=[], choices=[PATCH])
    probe = commands.add_parser('probe')
    probe.add_argument('--output', required=True)
    audit = commands.add_parser('smoke', help='four complete games maximum; diagnostic only, no promotion')
    audit.add_argument('--output', required=True)
    audit.add_argument('--artifact', required=True)
    audit.add_argument('--rival', choices=['router', 'k0006'], default='router')
    audit.add_argument('--timeout', type=float, default=300.)
    commands.add_parser('verify-panel', help='verify the completed maintainer D06 keep/drop receipt; launches no games')
    final = commands.add_parser('build-final', help='build the post-panel artifact with only kept patches')
    final.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            result = build(args.output, patches=args.patch)
        elif args.command == 'probe':
            result = probes(args.output)
        elif args.command == 'smoke':
            result = smoke(args.output, artifact=args.artifact, rival=args.rival, timeout=args.timeout)
        elif args.command == 'verify-panel':
            result = verified_panel_receipt()
        else:
            result = build_final(args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'REFUSED: {type(error).__name__}: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False), flush=True)
    return 1 if result.get('status') == 'FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
