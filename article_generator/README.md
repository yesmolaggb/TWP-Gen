# Article Generator

This package implements the outline-to-article stage of TWP-Gen. The canonical
pipeline entry point is:

```text
src/post_outline/run_postoutline_experiment.py
```

It reads the outline produced by `outline_generator`, retrieves section-specific
evidence from the collected source package, verifies supported outline items,
generates the document section by section, aligns claims with citations, and writes:

```text
../output/post_outline/article/<topic>.md
../output/post_outline/references/<topic>.json
../output/post_outline/diagnostics/<topic>.json
```

The end-to-end runner `../scripts/run_twpgen_pipeline.sh` calls this same entry
point. `src/run_from_outline.py` remains for backward compatibility but is not the
paper-reproduction path.

Configure the model once through the repository-root `.env` and
`twpgen_config.json`. See the [main README](../README.md) and
[post-outline documentation](src/post_outline/README.md).
