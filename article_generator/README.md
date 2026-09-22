# Article Generator

Lightweight outline-to-article generation module for the TWP-Gen pipeline.

The integrated pipeline calls `src/run_from_outline.py` with `--no-search`, so it generates the final article directly from the outline created by `outline_generator`. Runtime outputs go to `../output/article` and `../output/references`.

Configure the LLM through the project `.env` file: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `TWPGEN_LLM_MODEL`, or the article-specific `ARTICLE_LLM_*` variables.

## Alternative generation path

`src/post_outline/` provides a second implementation of the same stage (outline → article).
It keeps the same inputs and outputs, and adds section-level evidence retrieval, claim-level
fact verification and passage-level citation records. See `src/post_outline/README.md`.
