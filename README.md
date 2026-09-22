# TWP-Gen

English | [中文](README.zh-CN.md)

Reference implementation for **Technical White Paper Generation via Multidimensional
Feature Fusion**.

![Overall architecture of TWP-Gen](assets/method_overview.png)

*Overall architecture of TWP-Gen. The framework consists of five stages: (A) input
and retrieval for collecting evidence and knowledge units; (B) feature extraction for
entity-verb, semantic, event, and graph features; (C) feature fusion and clustering
for topic clusters; (D) outline induction for evidence-grounded structure; and (E)
citation-supported white paper generation.*

TWP-Gen is a bottom-up framework for technical white-paper generation. It retrieves
topic-related materials, converts them into traceable predicate-object knowledge
units, represents each unit through four complementary views (entity-verb relation,
sentence semantics, event function, and graph structure), fuses and clusters them,
maps the clusters to a corpus-derived section taxonomy to induce an evidence-grounded
outline, and finally generates each section with its associated source records.

## Included

The repository keeps only the main pipeline code, a root-level configuration template, topic examples, utility modules, and launch scripts. Large datasets, generated resources, logs, model checkpoints, and runtime outputs are not shipped. Runtime folders are created automatically when needed.

## Main Entry Points

```text
twpgen_settings.py                     Central configuration (single source of truth)
twpgen_config.example.json             Configuration template without secrets
twpgen_config.json                     Your local configuration (created from the template)
dataset/whitepaper_topics.json          The 60 generation tasks of the paper
dataset/evaluation/                    Metric definitions, rubrics and scoring scripts
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

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. create the local configuration
cp .env.example .env
cp twpgen_config.example.json twpgen_config.json
#    then edit twpgen_config.json and put your API key in .env

# 3. sanity check: print every resolved path and model setting
python twpgen_settings.py

# 4. run the full pipeline for every topic in the task list
bash scripts/run_twpgen_pipeline.sh
```

The runner exports the same configuration values used by the Python entry points,
so the shell stages and the Python stages never disagree about paths or models.

## Step-by-step run

The pipeline has three stages. Run them in order; each stage reads the central
configuration, so no paths need to be passed on the command line.

### Step 0 - configure once

```bash
cp .env.example .env
cp twpgen_config.example.json twpgen_config.json
```

`twpgen_config.json` holds every path, model name, dictionary location and topic
list. `.env` holds the secret:

```ini
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Verify what the code will use before running anything expensive:

```bash
python twpgen_settings.py           # resolved configuration as JSON
python twpgen_settings.py --shell   # export KEY=VALUE lines for shell scripts
```

### Step 1 - collect knowledge

```
knowledge_collector/collect_references.py
```

Reads the topic list (`dataset/whitepaper_topics.txt`) and writes one corpus per
topic under `dataset/<topic>/corpus.txt`. Requires the search credentials from
`.env` (`TAVILY_API_KEY` and, when used, the Jina key).

### Step 2 - cluster and induce the outline

```
outline_generator/run_twpgen.py
outline_generator/generate_whitepaper_outline.py
outline_generator/retrieve_outline_evidence.py
```

Builds knowledge units, extracts the four feature views, fuses and clusters them
(`k = 60` in the paper), maps the clusters onto the section taxonomy and writes
`dataset/<topic>/outline.txt`.

### Step 3 - generate the white paper

```
article_generator/src/post_outline/run_postoutline_experiment.py
```

Generates each section from its own outline node plus that section's evidence and
source records, then assembles the final document with traceable citations:

```bash
python article_generator/src/post_outline/run_postoutline_experiment.py \
  --outline-root dataset \
  --source-root  knowledge_collector/result \
  --output-root  output/post_outline \
  --limit 5          # drop --limit to run every topic
```

Outputs:

```
output/post_outline/generated/article/<topic>.md    generated white paper
output/post_outline/final/article/<topic>.md        citation-anchored version
output/post_outline/final/references/<topic>.json   per-source citation records
```

### Step 4 - evaluate

```bash
python dataset/evaluation/evaluate_topics.py \
  --article-dir output/post_outline/final/article \
  --output      output/evaluation/scores.json \
  --repeats     3
```

Ten 0--5 metrics over three groups, one call per group, three runs averaged per
metric. To summarise existing per-method result files instead:

```bash
python dataset/evaluation/run_summary.py --results-root path/to/eval --backbone 32b
```

## Data

Prepare your input under the configured dataset root, defaulting to `./dataset/<topic>/`. The open-source package intentionally does not include private corpora, generated `.pk` features, checkpoints, or large dictionaries.

## Dataset

The 60 generation tasks used in the paper and the evaluation code live in `dataset/`:

| Path | What it is |
|---|---|
| `dataset/whitepaper_topics.json` | The 60 tasks with `id`, `title` and `domain` |
| `dataset/whitepaper_topics.txt` | The same titles as a plain one-per-line list |
| `dataset/evaluation/` | The evaluation code: metrics, rubrics, scoring and summarisation |
| `dataset/<topic>/` | Generated per-topic runtime data (git-ignored) |

Their locations are part of the central configuration, so any stage can read them
through `twpgen_settings.py` (`topic_dataset_file`, `evaluation_code_dir`, ...).

## Project Layout

TWP-Gen is organized as a three-stage pipeline:

- `knowledge_collector/`: collect web knowledge and write `dataset/<topic>/corpus.txt`.
- `outline_generator/`: cluster and fuse evidence, then write `dataset/<topic>/outline.txt`.
- `article_generator/`: generate final Markdown/HTML articles from the outline.
- `scripts/run_twpgen_pipeline.sh`: one-command pipeline runner.

## Reuse of the configuration

Only `twpgen_settings.py` reads `twpgen_config.json`/`.env`; every other module
imports the resolved values from it. `outline_generator/twpgen_config.py` is kept
as a thin re-export so the existing `import twpgen_config as args` call sites keep
working, and `article_generator/src/post_outline/llm_settings.py` is the adapter
used by the outline-to-article path. To add a new path or credential, add it once
in `twpgen_settings.py` and use it everywhere.
