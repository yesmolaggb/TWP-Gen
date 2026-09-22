# Post-outline Article Generation（另一条正文生成路径）

本模块是 TWP-Gen 流水线中「大纲 → 正文」这一步的另一种实现，与 `article_generator` 既有实现并存：

- 既有实现：`src/run_from_outline.py`（读大纲、按章节生成、输出 `output/article` 与 `output/references`）
- 本模块：`src/post_outline/`（同样读大纲，但加入章节证据检索、论断级事实核验与可溯源引用）

两者输入输出一致，可以互相替换。选择哪一条由 `scripts/run_twpgen_pipeline.sh` 或手动调用决定。

## 1. 输入

### 1.1 大纲

支持两种布局，若同时存在则按下面的顺序优先：

```
<outline-root>/<主题>/outline.txt     编号大纲（一主题一目录）
<outline-root>/<主题>/outline.md
<outline-root>/<主题>.md              与 outline_generator / batch_from_outline 的输出一致
```

编号大纲示例：

```
题目：示例技术白皮书

1. 概述
   1.1 系统定位与核心能力
      - 覆盖设备接入、状态监测与统一运维
   1.2 适用范围
      - 适用于园区级与边缘节点部署
```

### 1.2 章节证据

每个主题一个目录，目录名包含主题名即可（例如 `01_示例技术白皮书`），内部放 `sources/`：

```
<source-root>/<主题目录>/sources/*.md
```

来源文件格式（元信息 + `## Search Snippet` 之后的正文）：

```markdown
# 页面标题

- Source ID: `SRC-XXXX`
- URL: https://example.com/page
- Relevance score: 0.87

## Search Snippet

正文内容……
```

### 1.3 主题列表

一个文本文件，每行一个主题名，与上面两级目录名对应：

```
示例技术白皮书
另一个主题
```

## 2. 运行

### 2.1 一键运行

在**仓库根目录**执行（下面的路径都以仓库根为基准；`dataset/` 换成你自己的数据目录）：

```bash
MOD=article_generator/src/post_outline

bash $MOD/run_pipeline.sh \
  --topics-file  topics.txt \
  --outline-root dataset/outline \
  --source-root  dataset/sources \
  --env-file     .env \
  --output-root  output/post_outline
```

产物：

```
output/post_outline/generated/article/<主题>.md    生成正文
output/post_outline/generated/diagnostics/        每章的证据检索与核验记录
output/post_outline/final/article/<主题>.md       引用锚定后的正文
output/post_outline/final/references/<主题>.json  引用记录（来源标题、URL、原文片段）
```

脚本依次执行两步：

| 步骤 | 做什么 | 产物 |
|---|---|---|
| ① 生成 | 检索章节证据 → 核验大纲条目 → 逐章生成 → 事实核验与候选稿择优 → 全局编辑 | `<output-root>/generated/article/<主题>.md` |
| ② 引用 | 为正文引用编号建立可溯源记录（来源标题、URL、逐句原文片段） | `<output-root>/final/article/`、`<output-root>/final/references/` |

### 2.2 分步运行

同样在仓库根目录执行：

```bash
MOD=article_generator/src/post_outline

# ① 单个主题（也可用 --index N 取内置主题列表中的第 N 个）
python $MOD/run_one_topic.py --topic "示例技术白皮书" \
  --outline-root dataset/outline \
  --source-root  dataset/sources \
  --output-root  output/post_outline/generated \
  --env-file     .env

# ② 构建引用记录
python $MOD/passage_level_citations.py \
  --input-article-dir output/post_outline/generated/article \
  --source-root dataset/sources \
  --output-root output/post_outline/final \
  --reference-granularity source --excerpt-chars 900 \
  --no-ground-uncited --no-text-edit

# ③ 可选：再执行一轮编辑（跨章节去重 / 长句拆分 / 术语首现解释）
python $MOD/editorial_polish.py \
  --input-article  "output/post_outline/final/article/示例技术白皮书.md" \
  --output-article "output/post_outline/edited/article/示例技术白皮书.md" \
  --diagnostic     output/post_outline/edited/diagnostics.json \
  --env-file       .env --focus dedup
```

## 3. 与既有实现的对应关系

| 环节 | 既有实现 | 本模块 |
|---|---|---|
| 章节证据 | `--search <json>` 注入检索结果 | 直接从 `<source-root>` 读取来源文件并做 BM25 检索 + 主题相关性过滤 |
| 大纲核验 | — | 逐条核验二级条目是否有连续原文支持，无证据的条目不写 |
| 生成约束 | `src/prompts/generate/generate.py` | 本模块 `SYSTEM_PROMPT`（引用克制、数值逐字来自证据、案例只展开一次） |
| 事实核验 | — | 论断—证据对齐 + 无依据数值检测 + 候选稿择优 |
| 引用记录 | `output/references/*.json` | 同样输出 `references/*.json`，`content` 为该来源的逐句原文片段 |

## 4. 参数

`run_pipeline.sh`：

| 参数 | 说明 |
|---|---|
| `--topics-file` | 主题列表文件 |
| `--outline-root` / `--source-root` | 大纲目录、来源目录 |
| `--env-file` | 可选；默认读取**仓库根目录**的 `.env` |
| `--output-root` | 输出目录 |
| `--model` | 可选；不填则按下面的优先级自动解析 |
| `--python` | 指定解释器路径（例如 conda 环境） |

## 5. 配置：一处定义，各处复用

全仓库（检索、大纲、正文生成）共用仓库根目录的同一个配置源：

| 文件 | 作用 |
|---|---|
| `twpgen_settings.py` | 配置中心，唯一读取与解析入口 |
| `twpgen_config.json` | 实际取值：所有路径、API Key、模型名、主题清单、字典路径 |
| `twpgen_config.example.json` | 无密钥模板，首次使用时复制成上面那个文件 |
| `.env` | 只放密钥与本机覆盖项 |

本模块的 `src/post_outline/llm_settings.py` 与 `article_generator/src/run_from_outline.py`、
`batch_from_outline.py`、`agent/generate.py` 以及 `outline_generator/*` 都读取配置中心，
不再各自写死路径或读取密钥。生成、责任编辑、引用构建三个入口统一调用
`resolve_settings()` 与 `build_client()`。

取值优先级（前者优先）：

1. 命令行显式参数：`--api-key` / `--base-url` / `--model`
2. 环境变量（含从 `.env` 载入的值）

   | 用途 | 变量 |
   |---|---|
   | API Key | `ARTICLE_LLM_API_KEY` → `OPENAI_API_KEY` |
   | 接口地址 | `ARTICLE_LLM_BASE_URL` → `OPENAI_BASE_URL` → `OPENAI_API_BASE` |
   | 模型 | `ARTICLE_LLM_MODEL` → `TWPGEN_LLM_MODEL` |

3. 配置中心 `twpgen_settings.py`（取值来自仓库根目录的 `twpgen_config.json`）
4. 模块内置默认值：模型 `qwen3-32b`、地址 `https://dashscope.aliyuncs.com/compatible-mode/v1`、超时 360 秒

也就是说：**在仓库根目录的配置里写一次，整条流水线（检索、大纲、正文生成）都能用上**，
无需在每个脚本里重复配置。`twpgen_config.json` 与 `.env` 已在 `.gitignore` 中，不会被提交。

想先看看当前解析出的路径和模型设置：

```bash
python twpgen_settings.py
```

`passage_level_citations.py`：

| 参数 | 说明 |
|---|---|
| `--reference-granularity` | `source`：每条来源一个编号（默认）；`claim`：每个论断一个编号 |
| `--excerpt-chars` | 引用片段长度上限，默认 900 |
| `--no-text-edit` | 只重建引用记录，不改动正文（推荐） |
| `--no-ground-uncited` | 不额外给未引用的句子补引用 |

`editorial_polish.py`：`--focus dedup`（去重）或 `--focus readability`（可读性）。

## 6. 依赖

```bash
pip install openai        # 其余依赖见仓库根目录 requirements.txt
```

生成过程会多次调用大模型接口，注意费用。
