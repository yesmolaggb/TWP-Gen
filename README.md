# TWP-Gen

TWP-Gen is the reference implementation for **Technical White Paper Generation via Multidimensional Feature Fusion**. It provides a lightweight technical white-paper generation pipeline built around corpus preparation, SVO/event extraction, multidimensional feature fusion, latent topic clustering, evidence retrieval, and outline synthesis.

## Included

The repository keeps only the main pipeline code, a root-level configuration template, topic examples, utility modules, and launch scripts. Large datasets, generated resources, logs, model checkpoints, and runtime outputs are not shipped. Runtime folders are created automatically when needed.

## Main Entry Points

```text
scripts/run_twpgen_pipeline.sh   Full topic pipeline runner
run_twpgen.py                    Main clustering entry point
generate_whitepaper_outline.py   Outline synthesis entry point
retrieve_outline_evidence.py     Evidence matching entry point
article_generator/src/post_outline/   Outline-to-article generation (evidence-grounded path)
twpgen_config.py                 Runtime config loader
twpgen_config.example.json       Config template without secrets
.env.example                     Environment variable template
```

## Configuration

```bash
cp .env.example .env
```

Then edit `.env` locally. Keep real API keys out of Git.

## Run

```bash
bash scripts/run_twpgen_pipeline.sh
```

Workspace wrapper:

```bash
bash /workspace/run_twpgen_pipeline.sh
```

## Data

Prepare your input under the configured dataset root, defaulting to `./dataset/<topic>/`. The open-source package intentionally does not include private corpora, generated `.pk` features, checkpoints, or large dictionaries.
## Project Layout

TWP-Gen is organized as a three-stage pipeline:

- `knowledge_collector/`: collect web knowledge and write `dataset/<topic>/corpus.txt`.
- `outline_generator/`: cluster and fuse evidence, then write `dataset/<topic>/outline.txt`.
- `article_generator/`: generate final Markdown/HTML articles from the outline.
- `scripts/run_twpgen_pipeline.sh`: one-command pipeline runner.
