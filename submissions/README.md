# Final submissions

The two archives exactly as uploaded to Kaggle, plus each one unpacked (minus `policy.pt`) so the inference code can be read here.

| Archive | sha256 | Bytes | Model | Weights |
|---|---|---|---|---|
| [`n1-main-sprinkle/n1-main-sprinkle.tar.gz`](n1-main-sprinkle/) | `932bb45d7fe9f0ebb486356285b43c5d079761c3905ce650d2e99c9c77986952` | 94,080,075 | [Model 1, B-even](../model1-b-even/) | fp16 EMA, 54,639,683 params |
| [`nflow-main-wash8014/nflow-main-wash8014.tar.gz`](nflow-main-wash8014/) | `7661782af4ca815585945b019963b86586dbae490ba2554b592fdb080f0f4b3e` | 13,605,819 | [Model 2, N flow](../model2-n-flow/) | fp16, 7,252,244 params |

The `.tar.gz` files are stored with Git LFS (`git lfs pull`).

## How an archive runs

- Entry point `main.py:agent(observation, configuration)`. It loads the policy lazily on the first call, on CPU with 1 thread, and returns PASS on any exception.
- Decoding is greedy (temperature 0, seed 0). `MANIFEST.json` declares these settings and hashes every member. `step_search.py` (score whole steps by joint log-probability and pick the best of k) ships in both archives but is not called in greedy mode.
- `agent.py` turns the raw Kaggle observation into the actor's view, computes features with the same code used in training, runs the model and decodes worker commands → orders → STOP.

| File | What |
|---|---|
| `main.py` | Kaggle entry point |
| `MANIFEST.json` | declared settings, model spec and sha256 of every member |
| `BUILD.json` | build record (builder sha, cold-verify games, timings) |
| `work/runs/maintainer/<harness>/agent.py` | observation → features → model → action |
| `work/runs/maintainer/<harness>/model.py` / `nf_model.py` | the network as served |
| `work/runs/maintainer/<harness>/v8*.py`, `calc_features.py`, `flip.py`, ... | feature code, identical to the training copies |
| `src/farm_neat/` | game constants, reference engine, observation helpers |

To play the two against each other on shared worlds, see [`../evaluation/`](../evaluation/).
