# Evidence-grounded article generation

`run_postoutline_experiment.py` is the canonical outline-to-article entry point
used by `scripts/run_twpgen_pipeline.sh`.

## Workflow

| Step | Operation |
|---|---|
| 1 | Parse the per-topic outline without changing its supported hierarchy |
| 2 | Retrieve evidence for each section from the collected source package |
| 3 | Remove outline items that lack direct evidence support |
| 4 | Generate each section under structural, citation, and factuality constraints |
| 5 | Align claims with source IDs and run the grounding quality gate |
| 6 | Assemble the document and export traceable reference records and diagnostics |

The generator accepts either source Markdown directories or the text report written
by `knowledge_collector/collect_references.py`.

## Inputs

```text
<outline-root>/<topic>/outline.txt
<source-root>/<topic>.txt
```

The source root may alternatively contain `<topic-directory>/sources/*.md`.

## Run

Execute from the repository root:

```bash
python article_generator/src/post_outline/run_postoutline_experiment.py \
  --outline-root dataset \
  --source-root knowledge_collector/result \
  --output-root output/post_outline \
  --topic "示例技术白皮书"
```

Batch options:

| Option | Meaning |
|---|---|
| `--topic <title>` | Select an exact topic; may be supplied more than once |
| `--start-index N` | Start from index `N` in the configured topic list |
| `--limit N` | Process at most `N` topics; `0` (default) means all |
| `--overwrite` | Regenerate an existing article |
| `--no-thinking` | Disable model reasoning |
| `--model`, `--base-url`, `--api-key` | Override the central configuration for this run |

## Outputs

```text
<output-root>/
├── article/<topic>.md
├── references/<topic>.json
├── diagnostics/<topic>.json
├── run_manifest.json
└── run_summary.json
```

`references/<topic>.json` preserves the citation ID, source title, URL, and an
original supporting excerpt. `diagnostics/<topic>.json` records section-level
coverage, invalid citations, heading checks, numeric-claim checks, and grounding
audits used during generation.

## Configuration

The entry point uses the repository-wide configuration adapter
`llm_settings.py`. Resolution order is:

1. explicit command-line options;
2. `ARTICLE_LLM_*` and `OPENAI_*` environment variables loaded from `.env`;
3. root `twpgen_config.json` through `twpgen_settings.py`;
4. secret-free fallback defaults.

Keep real credentials in the repository-root `.env`, never in source files.
