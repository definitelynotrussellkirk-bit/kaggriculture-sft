"""Frozen external Router-family agent: jaxa623's K0006, exact notebook-built bytes.

K0006 is Ahmed Berat Ozer's V43 (Apache-2.0, a descendant of yhay81's Shop Router with
upstream notices retained) plus four market wrappers: a step-0 wheat round trip, a
24-turn sale-reservation horizon, a two-turn sale advance and lockstep front-loading.
main.py and PROVENANCE.json in k0006_v43/ are never edited. `K0006Adapter` executes the
source into a fresh namespace per instance, as the Kaggle runner does, so no game memory
is shared between matches. Evaluation stratum and upgrade base, not training data.
"""
from __future__ import annotations
from pathlib import Path

from farm_neat.evaluation.common import bound, canonical, equal, file_sha, read

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'farm-k0006-v43-frozen-responsive-opponent-v1'
FOLDER = Path(__file__).resolve().parent/'k0006_v43'
MEMBERS = {'main.py': '4757f3f5b28db8a2f4614bb08993a8a567a7d49bae1ac324fbcfbcc1af60e95e'}


def identity():
    for name, sha in MEMBERS.items(): equal(file_sha(FOLDER/name), sha, 'frozen opponent asset '+name)
    provenance = read(FOLDER/'PROVENANCE.json')
    equal(provenance['members'], MEMBERS, 'imported asset provenance')
    return bound({'schema': SCHEMA, 'kind': 'FROZEN_RESPONSIVE_SCRIPTED_POLICY',
        'snapshot': 'jaxa623/2780-beyond-48-0-128-128-worlds-with-95-cis; exact notebook-built main.py',
        'assets': {str((FOLDER/name).relative_to(ROOT)): sha for name, sha in MEMBERS.items()},
        'provenance_sha256': provenance['content_sha256'], 'adapter_sha256': file_sha(__file__),
        'selection': 'DETERMINISTIC_OBSERVATION_BASED', 'opponent_rng': None,
        'observation_blind_tape': False, 'training_admission': False})


class K0006Adapter:
    pending = False

    def __init__(self):
        self.identity = identity()
        path = FOLDER/'main.py'
        namespace = {'__name__': 'k0006_' + self.identity['content_sha256'][:16]}
        exec(compile(path.read_bytes().decode('utf-8'), str(path), 'exec'), namespace)
        self.actor = namespace['agent']

    def act(self, observation, configuration=None):
        return self.actor(observation, configuration)


if __name__ == '__main__':
    print(canonical(identity()), flush=True)
