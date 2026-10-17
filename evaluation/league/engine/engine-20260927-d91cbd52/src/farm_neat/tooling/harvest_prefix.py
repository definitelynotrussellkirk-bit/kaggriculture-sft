"""Read-only harvest-prefix diagnostic for one source-bound own observation/jobs.

No world, policy rollout, actor, optimizer, or command is launched. The producer
must supply the exact current dispatcher job list, not a forecast or edited list.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from farm_neat.economy.heuristic.state import View
from farm_neat.evaluation.harvest_prefix import harvest_prefix_bound

ROOT = Path(__file__).resolve().parents[3]
ORDERS = 'src/farm_neat/economy/orders.py'
ORDERS_SHA256 = '4bb0401c1bdc075d8f3195b421f10ab6742524fbb2270f649f5f47ecca2ec97c'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(path: Path, expected_sha256: str) -> dict:
    """Read an explicitly pinned {observation, jobs, orders_sha256} JSON object."""
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError('input bytes differ from required SHA256')
    data = json.loads(raw)
    if data.get('orders_sha256') != ORDERS_SHA256 or sha(ROOT / ORDERS) != ORDERS_SHA256:
        raise ValueError('dispatcher version changed; revalidate, do not replace the pin')
    b = harvest_prefix_bound(View(data['observation']), data['jobs'])
    return {'schema': 'farm-harvest-prefix-receipt-v1', 'input_sha256': actual,
            'source_sha256': {ORDERS: ORDERS_SHA256,
                'src/farm_neat/evaluation/harvest_prefix.py': sha(ROOT / 'src/farm_neat/evaluation/harvest_prefix.py')},
            'bound': b, 'full_games': 0,
            'limit': 'Conditional on producer supplying the exact current job list; not a legal-action feasibility certificate.'}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    result = diagnose(a.input, a.sha256)
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
