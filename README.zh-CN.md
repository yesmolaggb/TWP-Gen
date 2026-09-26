# TWP-Gen

[English](README.md) | 简体中文

论文 **Technical White Paper Generation via Multidimensional Feature Fusion**
的官方实现。

TWP-Gen 是一个自底向上的技术白皮书生成框架：先把检索证据转换为可溯源的谓词—宾语
知识单元，再从四个互补视角进行表示和聚类，依据 11 类白皮书章节功能归纳证据支撑的
大纲，最后逐章节生成带来源引用的正文。

![TWP-Gen 总体架构](assets/method_overview.png)

*五个阶段分别为：（A）证据检索与知识单元构建；（B）多维特征提取；（C）特征融合与
聚类；（D）证据支撑的大纲归纳；（E）带引用的技术白皮书生成。*

## 项目概览

| 模块 | 主要功能 | 主要产物 |
|---|---|---|
| 知识采集 | 检索主题相关材料并保存来源元数据 | `dataset/<题目>/corpus.txt` 与来源记录 |
| 知识单元建模 | 提取谓词—宾语单元与四类概念特征 | 各题目的特征文件 |
| 证据聚类 | 融合归一化特征并细化 60 个主题簇 | `dataset/<题目>/clusters_/` |
| 大纲归纳 | 把有证据支撑的簇映射到 11 类章节体系 | `dataset/<题目>/outline.txt` |
| 正文生成 | 按章节证据生成正文并绑定引用 | 正文、参考来源与诊断文件 |
| 自动评价 | 按三组十项指标重复评分并汇总 | JSON、CSV、Markdown 与图 |

## 论文与代码对应关系

论文定义的是四个**概念视图**。第一个视图在代码中拆成两个内部子分支，因此实现层面
使用五个张量，但方法层面仍与论文的四视图定义一致。

| 论文中的视图 | 代码中的特征 | 作用 |
|---|---|---|
| 实体—动词关系 | `vs_emb` + `oh_emb` | 谓词义项与宾语中心词两个子分支共同构成第一视图 |
| 句子语义 | `summarized_emb` | 表示压缩后证据句的上下文语义 |
| 事件功能 | `event_emb` | 表示功能与事件层信息 |
| 图结构 | `sents_graph_emb` | 表示证据图中的结构关系 |

论文与代码共用的其他设置如下：

| 项目 | 论文设置 | 代码位置 |
|---|---:|---|
| 技术白皮书聚类数量 | `k = 60` | `outline_generator/run_twpgen.py` |
| 聚类损失权重 | `gamma = 5` | `outline_generator/run_twpgen.py` |
| 融合方式 | 逐元素求和 | `outline_generator/spherical_topic_clustering.py` |
| 概念特征视图 | 4 类 | 代码内部的五个张量按上表映射 |
| 章节分类体系 | 11 类功能 | `outline_generator/generate_whitepaper_outline.py` |
| 自动评价 | 10 项指标、3 个分组、重复 3 次 | `dataset/evaluation/` |

### 11 类章节体系

概述、背景、结论与展望为固定核心章节；其余章节仅在存在对应证据簇时生成。

| ID | 章节功能 | 生成方式 |
|---:|---|---|
| 1 | 概述 | 固定核心章节 |
| 2 | 背景 | 固定核心章节 |
| 3 | 解决方案与目标 | 由证据决定 |
| 4 | 架构设计 | 由证据决定 |
| 5 | 方法原理 | 由证据决定 |
| 6 | 应用场景 | 由证据决定 |
| 7 | 技术实现 | 由证据决定 |
| 8 | 评测与实验 | 由证据决定 |
| 9 | 安全与合规 | 由证据决定 |
| 10 | 结论与展望 | 固定核心章节 |
| 11 | 附录 | 由证据决定 |

无法可靠分配到证据依赖章节的簇不会单独生成章节，但其摘要仍作为概述、背景和结论章节
的文档级上下文使用，与论文中的方法描述一致。

## 仓库结构

```text
TWP-Gen/
├── assets/                         README 使用的主方法图
├── dataset/
│   ├── whitepaper_topics.json      60 个题目及八个领域标签
│   ├── whitepaper_topics.txt       批量运行用的逐行题目清单
│   └── evaluation/                 十项指标的 LLM 评价实现
├── experiments/
│   ├── cluster_number/             聚类数量敏感性实验
│   ├── duee/                       DuEE 指标与聚类算法比较
│   ├── run_feature_ablation.py     特征消融运行入口
│   ├── ablation_analysis.py        消融结果汇总
│   ├── statistical_significance.py 配对检验、Holm 校正与置信区间
│   └── domain_analysis.py          八领域结果汇总
├── knowledge_collector/            检索与来源采集
├── outline_generator/              知识单元、特征融合、聚类与大纲生成
├── article_generator/              基于证据的正文生成
├── scripts/run_twpgen_pipeline.sh  端到端运行脚本
├── twpgen_settings.py              统一配置读取器
└── twpgen_config.example.json      不含密钥的配置模板
```

大型语料、模型权重、中间特征、日志和生成结果未随仓库发布，对应运行目录均已加入
`.gitignore`。

## 安装与配置

推荐使用 Linux、Python 3.10 及以上版本。完整神经特征与聚类流程需要 CUDA。

```bash
git clone https://github.com/yesmolaggb/TWP-Gen.git
cd TWP-Gen
pip install -r requirements.txt

cp .env.example .env
cp twpgen_config.example.json twpgen_config.json
```

密钥只写入 `.env`；`.env` 和 `twpgen_config.json` 都不会被 Git 跟踪。统一配置的取值
优先级为：

```text
环境变量 > twpgen_config.json > 代码内置默认值
```

长时间运行前，建议先检查最终解析出的路径和模型设置：

```bash
python twpgen_settings.py
python twpgen_settings.py --shell
```

### 主要环境变量

| 变量 | 用途 | 何时需要 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI 兼容接口的密钥 | 大纲、正文生成或自动评价 |
| `OPENAI_BASE_URL` | OpenAI 兼容接口地址 | 使用云端或本地兼容服务 |
| `TWPGEN_LLM_MODEL` | 接口提供的默认模型名 | 所有 LLM 阶段 |
| `TAVILY_API_KEY` / `TAVILY_API_KEYS` | 单个检索密钥或密钥池 | 启用 Tavily 检索 |
| `TWPGEN_TAVILY_KEY_FILE` | 每行一个 Tavily 密钥的文件 | 从文件读取密钥池 |
| `ARTICLE_LLM_MODEL` | 单独覆盖正文生成模型 | 正文模型与默认模型不同时 |
| `ARTICLE_ENABLE_THINKING` | 在模型支持时启用推理 | 复现开启推理的生成设置 |
| `TWPGEN_TOPIC_FILE` | 批量题目清单 | 替换默认 60 题清单时 |

模型路径、运行环境、资源词典、输出路径和超参数只需在根目录的
`twpgen_config.json` 中配置一次。

## 运行方法

### 端到端运行

默认题目文件为 `dataset/whitepaper_topics.txt`，包含论文中的全部 60 个题目。脚本为
每个题目保存独立状态和日志，中断后可以继续。

```bash
bash scripts/run_twpgen_pipeline.sh
```

若只运行部分题目，可让 `TWPGEN_TOPIC_FILE` 指向另一个逐行题目文件。

### 从已有大纲生成正文

已完成检索、聚类和大纲生成时，可直接运行：

```bash
python article_generator/src/post_outline/run_postoutline_experiment.py \
  --outline-root dataset \
  --source-root knowledge_collector/result \
  --output-root output/post_outline \
  --limit 5
```

`--limit 0`（默认值）处理全部配置题目；可以重复添加 `--topic "<题目>"` 精确选择
题目；使用 `--overwrite` 重新生成已有正文。

### 输出目录

```text
output/post_outline/
├── article/<题目>.md           生成的技术白皮书
├── references/<题目>.json      实际引用的来源标题、URL 与原文摘录
├── diagnostics/<题目>.json     引用与证据绑定诊断
├── run_manifest.json            各题目运行状态
└── run_summary.json             模型、模式、耗时与汇总信息
```

## 数据集

`dataset/whitepaper_topics.json` 保存论文使用的 60 个中文技术白皮书生成题目。领域标签
只用于分组统计，不参与生成。

| 领域 | 题目数 |
|---|---:|
| 网络与互联网协议 | 16 |
| 人工智能与智能计算 | 11 |
| 移动通信（5G/6G） | 7 |
| 网络安全与可信技术 | 7 |
| 工业与行业应用 | 6 |
| 云、数据中心与存储 | 5 |
| 多媒体、感知与 XR | 4 |
| 能源与可持续基础设施 | 4 |
| **合计** | **60** |

运行时检索证据、受限或大型语料、模型权重和生成文档未随仓库重新分发。字段结构见
[dataset/README.zh-CN.md](dataset/README.zh-CN.md)。

## 自动评价

自动评价包含三组十项 0--5 分指标。评价器接收匿名化正文；在证据可信度评价中，还会
接收引用统计、引用段落与来源原文配对、来源标题与 URL，以及未引用正文摘录。

| 指标组 | 指标 |
|---|---|
| 内容质量 | 相关性、广度、深度、新颖性 |
| 白皮书适应性 | 技术具体性、可理解性、结构性 |
| 证据可信度 | 引用充分性、引用有效性、事实一致性 |

```bash
python dataset/evaluation/evaluate_topics.py \
  --article-dir output/post_outline/article \
  --reference-dir output/post_outline/references \
  --diagnostic-dir output/post_outline/diagnostics \
  --output output/evaluation/scores.json \
  --repeats 3 \
  --resume

python dataset/evaluation/run_summary.py \
  --results-root output/evaluation \
  --backbone 32b
```

当 `article`、`references` 和 `diagnostics` 为同级目录时，后两个目录可以自动识别。

## 复现论文实验

下表中的脚本都是可直接运行的实验入口。部分实验需要体积较大或因许可原因未发布的
中间特征和已评分基线结果。

| 实验 | 命令 | 本地所需输入 | 主要输出 |
|---|---|---|---|
| 聚类数量敏感性 | `python experiments/cluster_number/run_sensitivity.py --dataset-root dataset --topics-file dataset/whitepaper_topics.json --k 20 30 40 50 60 70 80` | 各题目的 `clusters_/embed_0.pt` | CSV/JSON 与图 |
| DuEE 聚类算法比较 | `python experiments/duee/compare_clustering_algorithms.py --input <arrays.npz> --output-dir output/duee_algorithms` | 含 `labels` 和 `gesi_embeddings` 的 NPZ | ARI/NMI/ACC/B³ F1 表 |
| 特征消融运行 | `python experiments/run_feature_ablation.py --topic <题目> --dataset-root dataset` | 已准备的题目特征文件 | 各变体运行清单 |
| 消融结果汇总 | `python experiments/ablation_analysis.py --results-root <评价目录> --backbone 32b` | 已完成的各变体分数 | JSON/CSV/Markdown |
| 统计显著性 | `python experiments/statistical_significance.py --results-root <评价目录> --backbone 32b` | 逐题目配对分数 | Wilcoxon/Holm 表与置信区间图 |
| 跨领域分析 | `python experiments/domain_analysis.py --results-root <评价目录> --topics dataset/whitepaper_topics.json --backbone 32b` | 逐题目分数 | 八领域结果表 |

DuEE 实验实现标准 ARI、NMI、基于 Hungarian 最优一一匹配的 ACC，以及 B³ F1。四项
指标均乘以 100 后报告，数值越高表示预测聚类与参考类别的一致性越强。

## 论文报告结果

下表与论文总体结果保持一致，`Avg.` 为十项指标的非加权平均值。

| 生成模型 | 评价模型 | TWP-Gen Avg. | 最强基线 Avg. | 提升 |
|---|---|---:|---:|---:|
| Qwen3-14B，关闭推理 | Qwen3-32B | **4.39** | 4.09 | +0.30 |
| Qwen3-32B，开启推理 | Qwen3-32B | **4.49** | 4.14 | +0.35 |
| Qwen3-32B，开启推理 | DeepSeek-V3 | **4.43** | 4.11 | +0.32 |

在包含 1,011 个实例的 DuEE 受控子集上，多维表示取得 50.55 ARI、73.94 NMI、
58.19 ACC 和 67.21 B³ F1；仅使用句子语义时分别为 34.03、56.33、47.54 和
53.82。对 60 个白皮书题目的成对比较得到 Holm 校正后的 Wilcoxon `p < 0.001`；
与最强基线相比，平均提升为 0.354，95% 置信区间为 `[0.275, 0.428]`。

## 复现说明

- 随机种子和实验参数均由实验脚本暴露，可使用 `--help` 查看完整接口。
- 检索、大纲、正文生成和评价都读取同一份根目录配置，请勿在模块中写入机器路径或密钥。
- 生成结果与大型中间文件不会提交到 Git，其预期位置已在上文和脚本帮助中说明。
- 源代码地址：<https://github.com/yesmolaggb/TWP-Gen>。
