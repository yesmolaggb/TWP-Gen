# TWP-Gen Evaluation Dataset

This folder contains the topic set and the evaluation protocol used for the TWP-Gen
paper: **Technical White Paper Generation via Multidimensional Feature Fusion**.

## Files

| File | What it is |
|---|---|
| `whitepaper_topics.json` | The 60 generation tasks. Each entry has `id`, `title` (the input topic) and `domain`. |
| `whitepaper_topics.txt` | The same 60 titles as one title per line, for scripts that read a plain text list. |
| `whitepaper_domain_classification.csv` | The same 60 titles with their domain assignment and the basis of that assignment. |
| `whitepaper_domain_summary.csv` | Domain counts and percentages. |
| `evaluation_method.json` | The full evaluation protocol in machine-readable form. |
| `evaluation_method.md` | The same protocol in prose, for readers of the paper or the code. |

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
generator. The documents themselves are used to analyse section structure and to
serve as the real-world reference point of the task. Because the pipeline is fully
unsupervised, no training, validation or test split is defined.

This evaluation set is separate from the larger 892-document corpus that is used
only to derive the white-paper section taxonomy.

## Reading the files

```python
import json

dataset = json.load(open("whitepaper_topics.json", encoding="utf-8"))
for task in dataset["topics"]:
    print(task["id"], task["title"], task["domain"])
```

```python
protocol = json.load(open("evaluation_method.json", encoding="utf-8"))
print(protocol["metric_order"])
```

Each `title` can be used directly as the input of the pipeline; `id` gives a stable
shorthand such as `T01` for reporting.

## Configuration

The folder is registered in the central configuration at the repository root, so no
script needs to hard-code it:

| Setting | Value |
|---|---|
| `evaluation_dataset_dir` | `./数据集` |
| `topic_dataset_file` | `./数据集/whitepaper_topics.json` |
| `topic_dataset_txt` | `./数据集/whitepaper_topics.txt` |
| `evaluation_protocol_file` | `./数据集/evaluation_method.json` |
| `evaluation_protocol_doc` | `./数据集/evaluation_method.md` |

```python
import twpgen_settings as cfg
print(cfg.topic_dataset_file)
```
