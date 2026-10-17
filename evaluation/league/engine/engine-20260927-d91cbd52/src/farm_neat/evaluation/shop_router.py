"""Frozen external opponent: yhay81's Shop Router 0909, unchanged Apache-2.0 runtime.

Responsive scripted policy with 13 complete 719-turn tapes, NOT an observation-blind
tape, a neural policy, or a training-data producer. It selects among its tapes from the
observed shops at step 144, repairs weeds, advances eligible sales and liquidates at 718.
Sale logic is credited by its author to aurax7. LICENSE.txt, main.py, actions.json and
PROVENANCE.json in shop_router_0909/ are the exact downloaded bytes and are never edited.

`identity()` verifies all three member hashes before returning the frozen identity the
checkpoints record. Use `ShopRouterAdapter` as the opponent; each match gets a fresh
instance, so no game memory is shared.
"""
from __future__ import annotations
from pathlib import Path

from farm_neat.evaluation.common import bound, canonical, equal, file_sha, read
from farm_neat.evaluation import opponents

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-shop-router-0909-frozen-responsive-opponent-v1'
FOLDER = Path(__file__).resolve().parent/'shop_router_0909'
MEMBERS = {
    'main.py': 'd6d74997dc5b483db63d8e39cafa1afeec0f366824e75107e109123f111e866b',
    'actions.json': '17d503f2fd20d59f9c0f14024d1e74a8add8bb9b5561d4d908b45deecb5495ef',
    'LICENSE.txt': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'}


def identity():
    for name, sha in MEMBERS.items(): equal(file_sha(FOLDER/name), sha, 'frozen opponent asset '+name)
    provenance = read(FOLDER/'PROVENANCE.json')
    equal(provenance['members'], MEMBERS, 'imported asset provenance')
    return bound({'schema': SCHEMA, 'kind': 'FROZEN_RESPONSIVE_SCRIPTED_POLICY',
        'snapshot': 'yhay81/shop-router-0909; exact downloaded2026-09-09 source',
        'assets': {str((FOLDER/name).relative_to(ROOT)): sha for name, sha in MEMBERS.items()},
        'provenance_sha256': provenance['content_sha256'], 'adapter_sha256': file_sha(__file__),
        'loader_sha256': file_sha(opponents.__file__), 'selection': 'DETERMINISTIC_OBSERVATION_BASED',
        'opponent_rng': None, 'shop_selection_step': 144, 'shared_terminal_plan_step': 648,
        'tape_count': 13, 'tape_turns': 719, 'responsive_repairs': ['WEED_DIG_QUEUE', 'ADVANCE_SALES', 'TERMINAL_LIQUIDATION'],
        'observation_blind_tape': False, 'training_admission': False})


class ShopRouterAdapter:
    """Existing callable loader, fresh instance rather than shared global game memory."""
    pending = False

    def __init__(self):
        self.identity = identity()
        kernel = opponents._tape_module_agent(FOLDER/'main.py')
        self.actor = kernel.__globals__['Policy'](FOLDER)

    def act(self, observation, configuration=None):
        return self.actor.act(observation)

    def observe(self, observation):
        return None

    def snapshot(self):
        return {'controller': SCHEMA, 'identity': self.identity['content_sha256'],
            'players': {str(seat): {**vars(state), 'queues': {str(worker): list(queue)
                for worker, queue in state.queues.items()}} for seat, state in self.actor.players.items()}}


def sources():
    return {**benchmark.sources(), 'src/farm_neat/evaluation/shop_router.py': file_sha(__file__),
        'scripts/train_semantic_joint.py': file_sha(joint.__file__), 'src/farm_neat/evaluation/opponents.py': file_sha(opponents.__file__),
        **identity()['assets']}




def sources():
    return {'src/farm_neat/evaluation/shop_router.py': file_sha(__file__),
        'src/farm_neat/evaluation/opponents.py': file_sha(opponents.__file__),
        'src/farm_neat/evaluation/common.py': file_sha(Path(__file__).resolve().parent/'common.py'),
        **identity()['assets']}


if __name__ == '__main__':
    print(canonical(identity()), flush=True)
