# DuEE clustering analysis

The scripts in this directory support the controlled clustering analysis reported
in the paper.

- `clustering_metrics.py` implements ARI, NMI, Hungarian-matched ACC, and B-cubed F1.
- `compare_clustering_algorithms.py` compares K-Means, GMM, and Bisecting K-Means on a fixed multidimensional representation.
- `plot_representation_tsne.py` generates the two-panel t-SNE comparison between semantic-only and multidimensional representations used for the paper figure.
- `prepare_controlled_subset.py` normalizes the prepared labels and exports the
  1,011-instance manifest distributed in `dataset/duee/`.

The plotting script expects an NPZ file containing `labels`,
`semantic_embeddings`, and `gesi_embeddings`, plus the corresponding metrics JSON.
The historical key name `gesi_embeddings` is retained for compatibility with the
experiment artifact; it denotes the TWP-Gen multidimensional representation in
the paper.

Example:

```bash
python experiments/duee/plot_representation_tsne.py \
  --input /path/to/top10_comparison_arrays.npz \
  --metrics /path/to/top10_metrics.json \
  --output-dir /path/to/figures
```

This script reproduces the t-SNE visualization from prepared representations. It
does not by itself construct the multidimensional embeddings from raw DuEE data.
