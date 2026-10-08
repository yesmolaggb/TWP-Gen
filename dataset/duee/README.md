# DuEE controlled subset

This directory contains the controlled 10-class subset used for the module-level
clustering analysis:

- `duee_top10_manifest.csv`: 1,011 instance IDs and reference labels.
- `top10_comparison_arrays.npz`: semantic-only and TWP-Gen multidimensional
  representations used in the paper experiment.
- `top10_metrics.json`: the reported clustering metrics.

The original sentences are not redistributed. Download DuEE 1.0 from the
[official Baidu AI Studio dataset page](https://aistudio.baidu.com/aistudio/competition/detail/32?isFromCcf=true)
and match the original sentences by `sentence_id`. Access may require a Baidu
account and acceptance of the dataset's terms.
