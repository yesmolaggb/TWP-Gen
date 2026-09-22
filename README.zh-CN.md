# TWP-Gen

[English](README.md) | 中文

**Technical White Paper Generation via Multidimensional Feature Fusion** 的参考实现。

![TWP-Gen 总体架构](assets/method_overview.png)

*TWP-Gen 总体架构。框架包含五个阶段：(A) 输入与检索，收集证据并构建知识单元；(B) 特征抽取，得到实体—动词关系、句子语义、事件功能与图结构四类特征；(C) 特征融合与聚类，形成主题簇；(D) 大纲归纳，生成有证据支撑的章节结构；(E) 引用支撑的白皮书生成。*

TWP-Gen 是一个自底向上的技术白皮书生成框架。它检索主题相关材料，将其转换为可溯源
的谓词—宾语知识单元，用四类互补视角（实体—动词关系、句子语义、事件功能、图结构）
表示每个单元，融合并聚类后映射到从真实语料归纳出的章节分类体系，从而生成有证据支撑
的大纲，最后结合对应来源记录逐章节生成正文。

## 目录结构

```text
twpgen_settings.py                     配置中心（唯一配置来源）
twpgen_config.example.json             不含密钥的配置模板
twpgen_config.json                     本地配置（从模板复制得到）
dataset/whitepaper_topics.json         论文使用的 60 个生成题目
dataset/whitepaper_topics.txt          同样的 60 个题目，每行一个
dataset/evaluation/                    评估代码：指标、评分标准、打分与汇总
assets/method_overview.png             论文主方法图
scripts/run_twpgen_pipeline.sh         全流程一键运行脚本
outline_generator/run_twpgen.py        聚类主入口
outline_generator/generate_whitepaper_outline.py
                                       大纲生成入口
outline_generator/retrieve_outline_evidence.py
                                       证据匹配入口
article_generator/src/post_outline/    大纲到正文的生成路径
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成本地配置
cp .env.example .env
cp twpgen_config.example.json twpgen_config.json
#    然后编辑 twpgen_config.json，并把 API Key 写进 .env

# 3. 自检：打印所有解析后的路径与模型设置
python twpgen_settings.py

# 4. 对题目清单中的每个题目运行完整流程
bash scripts/run_twpgen_pipeline.sh
```

## 分步运行

整个流程分三个阶段，按顺序执行即可。每一步都读取配置中心，因此命令行不需要再传路径。

### 第 0 步：配置一次

```bash
cp .env.example .env
cp twpgen_config.example.json twpgen_config.json
```

`twpgen_config.json` 存放所有路径、模型名、词典位置和题目清单；`.env` 只放密钥：

```ini
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

在开始跑昂贵流程之前，先确认代码实际会用到什么：

```bash
python twpgen_settings.py           # 以 JSON 打印解析后的配置
python twpgen_settings.py --shell   # 以 export KEY=VALUE 形式打印，供 shell 使用
```

### 第 1 步：采集知识

```
knowledge_collector/collect_references.py
```

读取题目清单（`dataset/whitepaper_topics.txt`），为每个题目写入
`dataset/<题目>/corpus.txt`。需要 `.env` 中的检索凭据（`TAVILY_API_KEY`，用到时还有 Jina Key）。

### 第 2 步：聚类并生成大纲

```
outline_generator/run_twpgen.py
outline_generator/generate_whitepaper_outline.py
outline_generator/retrieve_outline_evidence.py
```

构建知识单元、抽取四类特征视图、融合并聚类（论文中 `k = 60`）、把簇映射到章节
分类体系，最终写出 `dataset/<题目>/outline.txt`。

### 第 3 步：生成正文

```
article_generator/src/post_outline/run_postoutline_experiment.py
```

按大纲节点结合该章节的证据与来源记录逐章节生成，再拼装成带可溯源引用的完整文档：

```bash
python article_generator/src/post_outline/run_postoutline_experiment.py \
  --outline-root dataset \
  --source-root  knowledge_collector/result \
  --output-root  output/post_outline \
  --limit 5          # 去掉 --limit 即处理全部题目
```

产物：

```
output/post_outline/generated/article/<题目>.md     生成的白皮书
output/post_outline/final/article/<题目>.md         引用锚定后的版本
output/post_outline/final/references/<题目>.json    每条来源的引用记录
```

### 第 4 步：评估

```bash
python dataset/evaluation/evaluate_topics.py \
  --article-dir output/post_outline/final/article \
  --output      output/evaluation/scores.json \
  --repeats     3
```

十个 0--5 分指标分为三组，每组一次调用，每个指标取三次运行的均值。若要汇总已有的
各方法结果文件：

```bash
python dataset/evaluation/run_summary.py --results-root path/to/eval --backbone 32b
```

## 配置说明

```bash
cp .env.example .env
cp twpgen_config.example.json twpgen_config.json
```

两个文件都放在仓库根目录。`twpgen_config.json` 存放所有路径、API Key、模型名、
词典位置和题目清单；`.env` 只放密钥和本机覆盖项。真实密钥不要提交到 Git。

取值优先级：环境变量 → `twpgen_config.json` → `twpgen_settings.py` 中的内置默认值。
配置文件里的相对路径以仓库根目录为基准，因此换机器后复制仓库即可直接使用。

## 数据集

论文使用的 60 个生成题目与评估代码位于 `dataset/`：

| 路径 | 说明 |
|---|---|
| `dataset/whitepaper_topics.json` | 60 个题目，含 `id`、`title`、`domain` |
| `dataset/whitepaper_topics.txt` | 同样的题目，每行一个，便于脚本读取 |
| `dataset/evaluation/` | 评估代码：指标、评分标准、打分与汇总 |
| `dataset/<题目>/` | 每个题目的运行时数据（已在 Git 中忽略） |

这些路径都登记在配置中心，任何阶段都可以通过 `twpgen_settings.py` 读取
（`topic_dataset_file`、`evaluation_code_dir` 等），无需在脚本里写死。

## 配置复用

只有 `twpgen_settings.py` 会读取 `twpgen_config.json` 与 `.env`，其他模块都从它导入
解析后的值。`outline_generator/twpgen_config.py` 作为薄转发层保留，使已有的
`import twpgen_config as args` 调用点无需修改；`article_generator/src/post_outline/llm_settings.py`
则在「大纲到正文」这条路径上做同样的适配。需要新增路径或凭据时，只需在
`twpgen_settings.py` 中加一次。
