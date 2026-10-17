"""Explicit offline CLI for funded_capacity; zero games and no executor writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from farm_neat.evaluation.funded_capacity import funded_action_bound


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cash-cents', type=int, required=True)
    parser.add_argument('--committed-cents', type=int, default=0)
    parser.add_argument('--days', type=int, required=True)
    parser.add_argument('--hand-cap', type=int, default=8)
    parser.add_argument('--floor', type=int, default=0, help='Declared required hires EVERY day, not HANDS.target')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        result = funded_action_bound(args.cash_cents, args.committed_cents, args.days,
                                     args.hand_cap, minimum_hands=[args.floor]*args.days)
        text = json.dumps(result, indent=2, sort_keys=True)+'\n'
        if args.out:
            with args.out.open('x') as stream:
                stream.write(text)
        else:
            sys.stdout.write(text)
    except (ValueError, OSError) as error:
        parser.exit(2, f'funded-capacity: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
