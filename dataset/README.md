# TWP-Gen Evaluation Dataset

English | [中文](README.zh-CN.md)

This folder holds the evaluation-side resources of the TWP-Gen paper: **Technical
White Paper Generation via Multidimensional Feature Fusion**.

## Files

| File | What it is |
|---|---|
| `whitepaper_topics.json` | The 60 generation tasks. Each entry has `id`, `title` (the input topic) and `domain`. |
| `whitepaper_topics.txt` | The same 60 titles as one title per line, for scripts that read a plain text list. |
| `evaluation/` | The evaluation code: metric definitions, 0--5 rubrics, the scoring protocol and the summary scripts. |

## The 60 tasks

The dataset is a set of generation prompts, not a supervised training set. Each
task is a Chinese technical white-paper title; the system is asked to produce the
complete white paper from that title alone. The 60 titles span eight domains:

| Domain | Documents | Share |
|---|---:|---:|
| Networking and Internet protocols | 16 | 26.7% |
| AI and intelligent computing | 11 | 18.3% |
| Mobile communications (5G/6G) | 7 | 11.7% |
| Cybersecurity and trust | 7 | 11.7% |
| Industrial and sector applications | 6 | 10.0% |
| Cloud, data center and storage | 5 | 8.3% |
| Multimedia, sensing and XR | 4 | 6.7% |
| Energy and sustainable infrastructure | 4 | 6.7% |

The domain label is only used for grouping and reporting; it is never given to the
generator or evaluator. The task file contains titles and domain labels only, and
the evaluation protocol does not provide a reference white paper to the evaluator.
Because the pipeline is unsupervised, no training, validation or test split is
defined.

This evaluation set is separate from the larger 892-document corpus that is used
only to derive the white-paper section taxonomy.

## The evaluation code

`evaluation/` implements the protocol described in the paper.

| File | What it does |
|---|---|
| `metrics.py` | The ten metrics, the three metric groups and the three 0--5 rubrics. Single source for what is measured. |
| `config.py` | Resolves the evaluator endpoint, key and model from the repository configuration; provides the task-list and output paths. |
| `scorer.py` | Anonymisation, document loading, the group-wise scoring prompt, score parsing and repeated runs. |
| `evaluate_topics.py` | Scores a directory of generated white papers and writes a result JSON. |
| `aggregate.py` | Averages the per-document scores, computes the group and overall averages, writes JSON/CSV/Markdown summaries. |
| `run_summary.py` | Command-line wrapper around `aggregate.py`. |

Scoring a directory of generated white papers:

```bash
python dataset/evaluation/evaluate_topics.py \
  --article-dir    output/post_outline/article \
  --reference-dir  output/post_outline/references \
  --diagnostic-dir output/post_outline/diagnostics \
  --output      output/evaluation/scores.json \
  --repeats     3
```

Summarising existing per-method result files:

```bash
python dataset/evaluation/run_summary.py \
  --results-root path/to/eval \
  --backbone     32b
```

The evaluator endpoint and key come from the repository configuration
(`twpgen_config.json` or `.env`), so nothing has to be repeated on the command line.
Pass `--model`, `--base-url` or `--api-key` only to override it for a single run.

## Reading the task list

```python
import json

dataset = json.load(open("whitepaper_topics.json", encoding="utf-8"))
for task in dataset["topics"]:
    print(task["id"], task["title"], task["domain"])
```

Each `title` can be used directly as the input of the pipeline; `id` gives a stable
shorthand such as `T01` for reporting. The evaluation code reads the same file, so
the task list is defined once.

## Configuration

The folder is registered in the central configuration at the repository root, so no
script needs to hard-code it:

| Setting | Value |
|---|---|
| `evaluation_dataset_dir` | `./dataset` |
| `topic_dataset_file` | `./dataset/whitepaper_topics.json` |
| `topic_dataset_txt` | `./dataset/whitepaper_topics.txt` |
| `evaluation_code_dir` | `./dataset/evaluation` |

```python
import twpgen_settings as cfg
print(cfg.topic_dataset_file)
```
