# Figures

Scripts and input data for every figure in the write-up. Rendered images are in [`../images/`](../images/), numbered in reading order. Each script writes next to itself (`model_options.py` into `options/`, `workflow_fig.py` into the current directory). They need matplotlib.

| Image | Script | Input data |
|---|---|---|
| 01 abandoned macro theorems | [`theorem_graph.py`](theorem_graph.py) | [`theorem-graph.json`](theorem-graph.json) (311 claims, 919 linked pairs) |
| 02 resources, 15 feature table, 16 feature drop test, 17 data mix, 18 recipe, 20 paired games | [`make_final_figures.py`](make_final_figures.py) | values in the script, [`model-params.json`](model-params.json), [`model-series.json`](model-series.json) |
| 04, 09 architectures | [`arch_figs.py`](arch_figs.py) | values in the script |
| 05-08, 10-14 spec sheets, signatures, parameter maps, timelines, evidence | [`model_options.py`](model_options.py) | [`model-params.json`](model-params.json) (checkpoint tensors by module), [`model-series.json`](model-series.json) (loss curves), [`b-vs-p1-spread.json`](b-vs-p1-spread.json) |
| 19 workflow | [`workflow_fig.py`](workflow_fig.py) | values in the script |
