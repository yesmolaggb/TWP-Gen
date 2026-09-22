# TWP-Gen

TWP-Gen is the reference implementation for **Technical White Paper Generation via Multidimensional Feature Fusion**. It provides a lightweight technical white-paper generation pipeline built around corpus preparation, SVO/event extraction, multidimensional feature fusion, latent topic clustering, evidence retrieval, and outline synthesis.

## Included

The repository keeps only the main pipeline code, a root-level configuration template, topic examples, utility modules, and launch scripts. Large datasets, generated resources, logs, model checkpoints, and runtime outputs are not shipped. Runtime folders are created automatically when needed.

## Main Entry Points

```text
twpgen_settings.py                     Central configuration (single source of truth)
twpgen_config.example.json             Configuration template without secrets
twpgen_config.json                     Your local configuration (created from the template)
scripts/run_twpgen_pipeline.sh         Full topic pipeline runner
outline_generator/run_twpgen.py        Main clustering entry point
outline_generator/generate_whitepaper_outline.py
                                       Outline synthesis entry point
outline_generator/retrieve_outline_evidence.py
                                       Evidence matching entry point
article_generator/src/post_outline/    Outline-to-article generation (evidence-grounded path)
```

## Configuration

```bash
cp .env.example .env
cp twpgen_config.example.json twpgen_config.json
```

Both files live at the repository root. `twpgen_config.json` holds every path,
API key, model name, dictionary location and topic list; `.env` only supplies
secrets and per-machine overrides. Keep real API keys out of Git, and edit
`twpgen_config.json` instead of scattering paths across scripts.

Every entry point reads the same values through `twpgen_settings.py`:

```bash
python twpgen_settings.py           # print the resolved configuration as JSON
python twpgen_settings.py --shell   # print export KEY=VALUE lines for shell scripts
```

The resolution order is: environment variable, then `twpgen_config.json`, then
the built-in defaults in `twpgen_settings.py`. Relative paths in the config file
are resolved against the repository root, so the same file works on any machine
after you copy the repository.

To point at a different config file, set `TWPGEN_CONFIG`:

```bash
TWPGEN_CONFIG=./my_config.json python twpgen_settings.py
```

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
