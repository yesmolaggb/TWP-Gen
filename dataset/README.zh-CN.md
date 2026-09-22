# TWP-Gen 评估数据集

[English](README.md) | 中文

本目录存放 TWP-Gen 论文 **Technical White Paper Generation via Multidimensional
Feature Fusion** 的评估侧资源。

## 目录内容

| 文件 | 说明 |
|---|---|
| `whitepaper_topics.json` | 60 个生成题目，每条含 `id`、`title`（生成输入）和 `domain` |
| `whitepaper_topics.txt` | 同样的 60 个题目，每行一个，便于脚本读取 |
| `evaluation/` | 评估代码：指标定义、0--5 评分标准、打分协议与汇总脚本 |

## 60 个题目

本数据集是一组生成提示，而不是有监督训练集。每个题目是一个中文技术白皮书标题，
系统需要仅依据该标题生成完整白皮书。60 个题目覆盖八个领域：

| 领域 | 文档数 | 占比 |
|---|---:|---:|
| 网络与互联网协议 | 16 | 26.7% |
| 人工智能与智能计算 | 11 | 18.3% |
| 移动通信（5G/6G） | 7 | 11.7% |
| 网络安全与信任 | 7 | 11.7% |
| 工业与行业应用 | 6 | 10.0% |
| 云、数据中心与存储 | 5 | 8.3% |
| 多媒体、感知与 XR | 4 | 6.7% |
| 能源与可持续基础设施 | 4 | 6.7% |

领域标签只用于分组和统计，不会提供给生成模型。白皮书原文用于分析章节结构，并作为
任务真实场景的参照。由于整个流程是无监督的，因此不划分训练集、验证集和测试集。

本评估集与用于归纳章节分类体系的 892 篇文档语料是分开的。

## 评估代码

`evaluation/` 实现了论文中描述的评估协议。

| 文件 | 作用 |
|---|---|
| `metrics.py` | 十个指标、三个指标组、三套 0--5 评分标准的唯一出处 |
| `config.py` | 从仓库配置解析评估器地址、密钥与模型，并提供题目表与输出路径 |
| `scorer.py` | 匿名化、文档加载、按组打分提示词、分数解析与重复运行 |
| `evaluate_topics.py` | 对某个目录下的生成白皮书逐篇打分，写出结果 JSON |
| `aggregate.py` | 汇总每篇分数，计算组均值与总分，导出 JSON/CSV/Markdown |
| `run_summary.py` | `aggregate.py` 的命令行入口 |

对某个目录下的白皮书打分：

```bash
python dataset/evaluation/evaluate_topics.py \
  --article-dir output/article \
  --output      output/evaluation/scores.json \
  --repeats     3
```

汇总已有的各方法结果文件：

```bash
python dataset/evaluation/run_summary.py \
  --results-root path/to/eval \
  --backbone     32b
```

评估器地址与密钥来自仓库配置（`twpgen_config.json` 或 `.env`），无需在命令行重复填写。
只有在单次运行需要覆盖时才传 `--model`、`--base-url` 或 `--api-key`。

## 读取题目清单

```python
import json

dataset = json.load(open("whitepaper_topics.json", encoding="utf-8"))
for task in dataset["topics"]:
    print(task["id"], task["title"], task["domain"])
```

每个 `title` 可以直接作为流程的输入，`id` 提供 `T01` 这样的稳定简写，便于在结果中引用。
评估代码读取同一个文件，因此题目清单只定义一次。

## 配置

本目录已登记在仓库根目录的配置中心里，任何脚本都不需要写死路径：

| 配置项 | 取值 |
|---|---|
| `evaluation_dataset_dir` | `./dataset` |
| `topic_dataset_file` | `./dataset/whitepaper_topics.json` |
| `topic_dataset_txt` | `./dataset/whitepaper_topics.txt` |
| `evaluation_code_dir` | `./dataset/evaluation` |

```python
import twpgen_settings as cfg
print(cfg.topic_dataset_file)
```
