# Codex 单文件提示词（一）：Bench-CoE 动态专家研究的 ModelScope 资源下载、预处理与缓存

> 使用方式：把本文件全文复制给 Codex。  
> 本提示词只负责资源审计、ModelScope 下载、数据预处理、模型可加载性检查与资源锁定；不运行正式论文实验。  
> 若本提示词与仓库已有、经过验证的协议冲突，先保留已有实现并在审计报告中列出差异，不得静默覆盖。

---

你是一名负责可重复机器学习基础设施的高级研究工程师。请在当前 Bench-CoE / Subject-Bert-Bench-CoE 仓库中完成“动态异构 LLM/VLM 专家池”扩展研究所需的全部资源准备工作。

## 0. 总目标

构建一个只从 **ModelScope（魔搭）** 下载数据集和模型、可断点续传、可审计、可复现的资源流水线，为后续 Improve5（局部可靠性）与修正后的 Improve6（有向失败—修复图）实验提供统一输入。

必须交付：

1. 可执行的 ModelScope 资源发现、下载、校验脚本；
2. 明确分层的 `smoke`、`core`、`full` 下载配置；
3. 文本和多模态数据集的统一样本格式；
4. 模型、数据、图片资源和许可证状态的锁文件；
5. 源域校准集、源域内部验证集、目标测试集之间不可交叉的角色清单；
6. 后续实验可直接消费的不可变资源清单与数据索引；
7. 资源准备报告，包括成功项、缺失项、受限项、磁盘估计、重复项和潜在污染风险。

本阶段不得：

- 运行正式方法比较、显著性检验或论文主表；
- 使用目标测试标签选择模型、超参数、提示词、答案解析器或数据子集；
- 从 Hugging Face、GitHub Release、网盘或其他镜像静默补下载；
- 把一个相似但不同的数据集/模型当作原资源的替代品；
- 把模型的不同提示模板或不同随机种子伪装成独立专家；
- 修改、删除或覆盖用户已有输出；
- 编造 ModelScope ID、revision、哈希、许可证、数据规模或下载成功状态。

## 1. 开始前的仓库与环境审计

先执行只读检查，再修改代码：

1. 阅读仓库根目录及父目录适用的 `AGENTS.md`、README、现有实验说明、环境文件和数据脚本。
2. 检查 `git status --short`、当前分支和已有未提交修改。用户修改必须保留；只编辑本任务相关文件。
3. 搜索现有的：
   - Bench-CoE、Improve5、Improve6、RepairChain、DARE、FARG、CRAFT、RIFT；
   - 数据注册表、模型注册表、缓存格式、答案解析器；
   - MMLU-Pro、BBH、GPQA、MMStar、CMMMU、MathVista、MMMU-Pro；
   - Hugging Face、ModelScope、EvalScope 或 lm-evaluation-harness 适配代码。
4. 记录 Python、PyTorch、CUDA、Transformers、ModelScope、EvalScope、vLLM 版本及 GPU、CPU、RAM、可用磁盘。
5. 不要擅自升级核心依赖。若依赖缺失，建立单独、最小化且有版本约束的环境文件。
6. 如果当前目录不是目标仓库，先在工作区中查找仓库；仍找不到则停止，给出阻塞报告，不要新建一个冒充原仓库的空项目。

优先复用仓库已有的数据结构和推理入口；新代码应通过适配层接入，不能破坏已有实验。

## 2. 存储位置与强制环境变量

实现以下约定：

```bash
BENCHCOE_ASSET_ROOT=/absolute/path/to/benchcoe_assets
MODELSCOPE_CACHE=/absolute/path/to/modelscope_cache
BENCHCOE_PROFILE=smoke|core|full
MODELSCOPE_API_TOKEN=optional_token
```

规则：

- 不得使用 `$HOME`、`~` 或系统根目录作为批量下载目标。
- 若 `BENCHCOE_ASSET_ROOT` 未设置，可在仓库外层的安全工作区创建一个明确目录，但必须在报告中给出绝对路径。
- 将模型、原始数据、处理后数据、输出缓存分目录存放：

```text
${BENCHCOE_ASSET_ROOT}/
  models/
  datasets_raw/
  datasets_processed/
  image_assets/
  locks/
  manifests/
  logs/
  quarantine/
```

- 所有大文件目录写入 `.gitignore`，代码、配置、manifest 和小型统计报告可以纳入 Git。
- 下载前估计所选 profile 的总空间；要求“可用空间 ≥ 预计下载量 × 1.25”。不足则停止下载，生成空间报告。
- 不得删除现有文件释放空间。临时文件只能放在任务专用临时目录中。

## 3. 资源分层

### 3.1 Smoke 层

用于验证下载、预处理和推理接口：

- 文本模型：Qwen3-1.7B；
- 视觉模型：InternVL3.5-2B；
- 每个必测数据集最多抽取 8 个仅用于管线检查的样本；
- smoke 子集只能按稳定样本 ID 或固定哈希选择，不能依据模型答对/答错选择。

### 3.2 Core 层

用于主实验的推荐最低资源：

文本同家族规模梯度：

- Qwen3-1.7B；
- Qwen3-4B；
- Qwen3-8B；
- Qwen3-14B。

文本领域专家：

- Qwen2.5-Math-7B-Instruct；
- Qwen2.5-Coder-7B-Instruct。

视觉同家族规模梯度：

- InternVL3.5-2B；
- InternVL3.5-4B；
- InternVL3.5-8B；
- InternVL3.5-14B。

视觉跨家族：

- Qwen2.5-VL-3B-Instruct；
- Qwen2.5-VL-7B-Instruct。

数据集使用第 5 节中标记为 `core` 的全部资源。

### 3.3 Full 层

在 Core 基础上增加：

- Qwen3-32B 强锚点；
- InternVL3.5-38B 强视觉锚点；
- Gemma-3-12B-IT 或等价的官方 12B 指令版本；
- Mistral-Nemo-Instruct-2407；
- Llama-3.1-8B-Instruct，仅在 ModelScope 上能合法访问且许可证已确认时启用；
- Gemma-3-4B-IT、Gemma-3-12B-IT 视觉能力对照；
- 第 5 节标记为 `full` 或 `optional` 的数据集。

32B/38B 和受限模型不得由 `smoke` 或 `core` 隐式触发。只有显式设置 `BENCHCOE_PROFILE=full` 才允许下载。

## 4. 模型注册表

建立 `configs/modelscope_models.yaml`。下列是要解析的逻辑资源及 **候选** ModelScope ID；下载脚本必须调用 ModelScope 当前 API 验证 ID、类型、revision、文件列表和可访问性，不能仅凭字符串存在就认定成功。

### 4.1 文本模型

| 逻辑名 | 首选候选 ID | 层级 | 角色 |
|---|---|---|---|
| qwen3_1_7b | `Qwen/Qwen3-1.7B` | smoke/core | 同家族规模 |
| qwen3_4b | `Qwen/Qwen3-4B` | core | 同家族规模 |
| qwen3_8b | `Qwen/Qwen3-8B` | core | 同家族规模 |
| qwen3_14b | `Qwen/Qwen3-14B` | core | 同家族规模 |
| qwen3_32b | `Qwen/Qwen3-32B` | full | 强锚点 |
| qwen25_math_7b | `Qwen/Qwen2.5-Math-7B-Instruct` | core | 数学专家 |
| qwen25_coder_7b | `Qwen/Qwen2.5-Coder-7B-Instruct` | core | 代码专家 |
| gemma3_12b_it | 通过 ModelScope 官方搜索解析 | full | 跨家族 |
| mistral_nemo_12b | 通过 ModelScope 官方搜索解析 | full | 跨家族 |
| llama31_8b_instruct | 通过 ModelScope 官方搜索解析 | full/restricted | 跨家族 |

### 4.2 视觉语言模型

| 逻辑名 | 首选候选 ID | 层级 | 角色 |
|---|---|---|---|
| internvl35_2b | `OpenGVLab/InternVL3_5-2B` | smoke/core | 同家族规模 |
| internvl35_4b | `OpenGVLab/InternVL3_5-4B` | core | 同家族规模 |
| internvl35_8b | `OpenGVLab/InternVL3_5-8B` | core | 同家族规模 |
| internvl35_14b | `OpenGVLab/InternVL3_5-14B` | core | 同家族规模 |
| internvl35_38b | `OpenGVLab/InternVL3_5-38B` | full | 强锚点 |
| qwen25_vl_3b | `Qwen/Qwen2.5-VL-3B-Instruct` | core | 跨家族 |
| qwen25_vl_7b | `Qwen/Qwen2.5-VL-7B-Instruct` | core | 跨家族 |
| gemma3_4b_it | 通过 ModelScope 官方搜索解析 | full | 跨家族/规模 |
| gemma3_12b_it_vl | 通过 ModelScope 官方搜索解析 | full | 跨家族/规模 |

### 4.3 模型解析规则

1. namespace 大小写或官方 HF-format 后缀不同的情况，必须通过 ModelScope 搜索/API 解析并写入锁文件。
2. 优先官方发布者或可信组织；若只有个人镜像，标记 `untrusted_mirror`，默认不下载。
3. 不允许把 FP8/AWQ/GPTQ 版本自动替代 BF16/原始权重。量化版是独立资源。
4. 主实验优先 BF16。若某模型只能量化运行，必须将量化方式作为实验变量，不能混入 scale-only 主比较。
5. 记录：
   - `logical_name`；
   - `resolved_modelscope_id`；
   - `revision` 或 commit；
   - `publisher`；
   - `parameter_count`；
   - `architecture`；
   - `modality`；
   - `instruction_tuned`；
   - `thinking_capability`；
   - `license`；
   - `gated`；
   - `precision`；
   - `download_bytes`；
   - `local_path`；
   - `status`。
6. Qwen3 主实验固定 `enable_thinking=False`；thinking 模式只能作为后续独立实验，不能成为另一个“独立专家”。
7. 模型加载必须使用本地路径和 `local_files_only=True`；正式运行前设置离线模式，防止 Transformers 回源下载。

## 5. 数据集注册表与角色

建立 `configs/modelscope_datasets.yaml`。每个数据集均需包含：

- `logical_name`；
- `candidate_modelscope_ids`；
- `task_type`；
- `modality`；
- `allowed_role`；
- `allowed_splits`；
- `answer_type`；
- `license`；
- `revision`；
- `raw_hash_or_snapshot_id`；
- `status`。

### 5.1 文本源域校准池

这些数据的标签可用于源域能力、局部邻域和修复边估计，但同一 split 不得同时作为目标测试。

| 数据集 | 候选 ModelScope ID | 层级 | 用途 |
|---|---|---|---|
| MMLU-Pro | `TIGER-Lab/MMLU-Pro`，备选 `modelscope/MMLU-Pro` | core | 主源域 |
| ARC-Challenge | 通过 ModelScope 搜索解析官方/可信镜像 | core | 科学推理源域 |
| LogiQA 2.0 | 通过 ModelScope 搜索解析官方/可信镜像 | core | 逻辑推理源域 |

在固定总预算时，需要支持 1、2、4 个源 benchmark 的组合，但组合逻辑留给实验阶段。

### 5.2 文本目标与外部验证

| 数据集 | 候选 ModelScope ID | 层级 | 角色 |
|---|---|---|---|
| BBH | 通过 ModelScope 搜索解析 | core | 已有 OOD 目标 |
| GPQA | 通过 ModelScope 搜索解析 | core | 已有高难 OOD 目标 |
| LiveBench | `Gen-Verse/LiveBench`；各子集可另解析 `AI-ModelScope/livebench_*` | core | 版本锁定的外部验证 |
| MuSR | `AI-ModelScope/MuSR` | core | 多步软推理外部验证 |
| HLE | `AI-ModelScope/hle` | full/secondary | 高难度次要压力测试 |
| HLE-Verified | `lmms-lab/HLE-Verified` 或经验证的对应镜像 | optional/separate | 只能作为独立敏感性版本 |

HLE 与 HLE-Verified 是不同实验资源，不得静默互换。HLE 主分析只按预先定义的 `text-only`、`multiple-choice/closed-form`、语言等元数据筛选，禁止根据模型是否答对来筛题。

### 5.3 视觉源域校准池

| 数据集 | 候选 ModelScope ID | 层级 | 用途 |
|---|---|---|---|
| MMMU-Pro | 通过 ModelScope 搜索解析 | core | 主视觉源域 |
| ScienceQA | 通过 ModelScope 搜索解析 | core | 科学图文源域 |
| AI2D | 通过 ModelScope 搜索解析 | core | 图解推理源域 |
| ChartQA | 通过 ModelScope 搜索解析 | core | 图表源域 |

### 5.4 视觉目标与外部验证

| 数据集 | 候选 ModelScope ID | 层级 | 角色 |
|---|---|---|---|
| MMStar | 通过 ModelScope 搜索解析 | core | 已有目标 |
| CMMMU | 通过 ModelScope 搜索解析 | core | 已有中文目标 |
| MathVista | 通过 ModelScope 搜索解析 | core | 已有数学视觉目标 |
| BLINK | `evalscope/BLINK` | core | 核心视觉感知外部验证 |
| NaturalBench | 通过 ModelScope 搜索解析 | core；找不到则阻塞该项 | 图像依赖真实性验证 |
| CV-Bench | `comefly/cvbench` | core | 2D/3D 空间关系 |
| CharXiv | 通过 ModelScope 搜索解析 | optional | 开放答案补充，不进主表 |
| HLE multimodal | `AI-ModelScope/hle` | optional/secondary | 高难视觉压力测试 |

### 5.5 数据集缺失规则

- 对没有可验证 ModelScope 资源的数据集，记录：

```json
{
  "status": "not_available_on_modelscope",
  "searched_at": "...",
  "queries": ["..."],
  "candidates": [],
  "impact": "...",
  "manual_action_required": true
}
```

- 不得自动去 Hugging Face、GitHub、Kaggle 或作者网盘下载。
- 不得使用相似名称的数据集代替。
- 继续处理其他可用资源；最后生成阻塞清单。

## 6. ModelScope 下载器

实现一个统一 CLI，例如：

```bash
python tools/modelscope_assets.py resolve --profile core
python tools/modelscope_assets.py estimate --profile core
python tools/modelscope_assets.py download --profile core
python tools/modelscope_assets.py verify --profile core
python tools/modelscope_assets.py lock --profile core
```

要求：

1. 使用当前安装版本的 ModelScope 官方 SDK/API；先通过最小 API 探测确认调用方式。
2. 支持断点续传、重试、指数退避、下载超时、代理环境继承和清晰日志。
3. 同一逻辑资源使用文件锁，防止多进程重复下载。
4. 下载完成前写临时状态，只有校验通过后才原子发布到最终路径。
5. 失败下载移入 `quarantine/` 或保留为带状态的临时目录，不得被验证器当作成功。
6. 支持 `--dry-run`、`--only`、`--exclude`、`--profile`、`--max-workers`、`--revision`。
7. 大模型默认串行下载；小型元数据解析可并行。
8. 记录每个资源的开始/结束时间、字节数、重试次数和最终状态。
9. 下载图片型数据集时，必须验证实际图像资产存在且能解码；只有 parquet/json 元数据而无图片不能算成功。
10. 下载脚本返回非零状态当且仅当必需资源失败；可选资源失败应在报告中标黄但不伪装为整体成功。

## 7. 统一数据格式

不得把不同 benchmark 粗暴拼成只含 `question/answer` 的表。建立版本化 schema，例如 `benchcoe_unified_v1`。

每条样本至少包含：

```json
{
  "schema_version": "benchcoe_unified_v1",
  "sample_id": "stable-dataset-split-nativeid-or-hash",
  "dataset": "logical_name",
  "dataset_revision": "resolved_revision",
  "split": "train|validation|test|...",
  "role": "source_calibration|source_validation|target_locked_test|secondary_test",
  "modality": "text|vision_language",
  "task_type": "multiple_choice|exact_match|open_ended|code",
  "question": "...",
  "context": "...",
  "choices": [{"label": "A", "text": "..."}],
  "answer_canonical": "A",
  "answer_raw": "...",
  "images": [
    {
      "index": 0,
      "relative_path": "...",
      "sha256": "...",
      "width": 0,
      "height": 0,
      "mime": "image/png"
    }
  ],
  "category": "...",
  "language": "...",
  "native_metadata": {},
  "content_hash": "...",
  "license": "..."
}
```

规则：

- `sample_id` 在重复运行中稳定，且不能依赖本地绝对路径。
- 保留原始 choices 顺序；不得重排后忘记同步答案。
- 保存 `answer_raw` 与 `answer_canonical`，答案归一化必须可逆审计。
- 多图样本保持原始顺序。
- 所有图片使用相对路径，生成图像 SHA-256、尺寸、格式和解码状态。
- 原始数据不可修改；清洗结果写新目录。
- 对开放答案、代码题、无法客观评分的样本单独标记，默认不进入 Improve5/6 主实验。
- 每个数据集生成：
  - `samples.jsonl` 或分片 parquet；
  - `dataset_manifest.json`；
  - `qa_report.json`；
  - `id_map.jsonl`；
  - `README.generated.md`。

## 8. 角色隔离与无污染协议

建立 `configs/data_roles.yaml`，强制执行：

1. 源域标签只用于：
   - 全局源域准确率；
   - Improve5 局部可靠性；
   - Improve6 一跳修复边；
   - 源域内部交叉验证和超参数选择。
2. 目标数据仅在最终评估时读取标签。推理缓存生成阶段不向方法暴露目标标签。
3. 目标标签不得用于：
   - 选择专家池；
   - 选择 `k`；
   - 选择边阈值或置信水平；
   - 选择源 benchmark；
   - 修改 prompt 或答案解析器；
   - 选择要报告的数据子集；
   - 决定模型加入顺序。
4. 如果某 benchmark 只有一个带标签 split：
   - 作为源域时，不得再作为目标；
   - 作为目标时，全部锁定为测试；
   - 不得从其中抽出“开发集”调参。
5. 创建 `protocol_lock.yaml`，在生成任何目标结果之前写入：
   - 数据角色；
   - 模型列表；
   - prompt 版本；
   - 解码参数；
   - 答案解析器版本；
   - 源域超参数搜索空间；
   - 主要/次要数据集；
   - 排除规则。
6. `protocol_lock.yaml` 生成哈希；后续修改必须产生新版本和原因，不能覆盖原版本。

## 9. 数据质量检查

对每个数据集自动检查并报告：

- 样本总数、各 split 数量、类别和语言分布；
- 缺失问题、缺失答案、choices 重复、非法答案标签；
- 重复 native ID、重复 content hash；
- 源域与目标域之间 exact hash、规范化文本 hash、近重复 MinHash；
- 图片缺失、损坏、全黑、尺寸异常、路径越界；
- 选项数量分布和随机猜测基线；
- 开放答案比例；
- 训练/验证/测试之间重复；
- 原始许可和再分发限制；
- 是否需要执行代码或 LLM judge；
- 是否存在答案字段命名不一致。

发现跨 split 或源—目标精确重复时：

- 不得自行删除后继续；
- 将样本列入 `overlap_report.jsonl`；
- 按预注册规则隔离；
- 在最终报告中说明数量和影响。

## 10. 模型可加载性与最小推理验证

下载完成后只执行最小 smoke 检查，不跑正式 benchmark：

1. tokenizer/processor/model 能从本地目录加载；
2. `local_files_only=True` 下无联网行为；
3. 文本模型能对 2 个固定非 benchmark 句子生成；
4. VLM 能读取一张程序生成的非 benchmark 测试图并回答；
5. GPU/CPU dtype、attention backend、device map 记录完整；
6. Qwen3 主入口明确关闭 thinking；
7. 每个模型记录最小峰值显存和加载时长；
8. 38B 模型若当前硬件不足，只做文件完整性检查并标记 `downloaded_not_runtime_verified`；
9. 不把 smoke 输出纳入专家缓存。

不要用真实目标测试题来调模型模板。若模型官方聊天模板不同，建立模型适配器，并将最终序列化 prompt 的哈希记录到 manifest。

## 11. 资源锁与实验输入契约

生成：

```text
artifacts_or_repo_metadata/
  asset_lock.json
  asset_lock.sha256
  model_registry.resolved.json
  dataset_registry.resolved.json
  protocol_lock.yaml
  protocol_lock.sha256
  resource_status.csv
  disk_report.json
  licenses_report.md
  overlap_report.jsonl
  preparation_report.md
```

`asset_lock.json` 必须列出每个资源的：

- 逻辑名和解析后的 ModelScope ID；
- revision/snapshot；
- 本地绝对路径与可移植相对路径；
- 上游文件清单/大小；
- 关键配置文件哈希；
- 数据处理代码 Git commit；
- 处理后数据 manifest 哈希；
- 下载与 QA 状态；
- 许可证状态；
- 是否允许进入 core/full 实验。

后续实验只接受：

```text
asset_lock 验证通过
+ protocol_lock 验证通过
+ 所需资源状态为 ready
```

若锁文件与磁盘内容不一致，实验必须失败并指出具体资源，不得偷偷重下或忽略。

## 12. 推荐代码结构

尽量适配原仓库；若没有等价结构，可新增：

```text
configs/
  modelscope_models.yaml
  modelscope_datasets.yaml
  data_roles.yaml
tools/
  modelscope_assets.py
  preprocess_benchmarks.py
  verify_assets.py
benchcoe/
  assets/
    registry.py
    modelscope_backend.py
    locking.py
    validation.py
  data/
    schema.py
    adapters/
  models/
    adapters/
tests/
  test_asset_registry.py
  test_unified_schema.py
  test_split_isolation.py
  test_answer_mapping.py
  test_image_integrity.py
docs/
  resource_preparation.md
```

所有脚本均需：

- `--help` 可用；
- 日志清晰；
- 支持重复运行；
- 路径通过配置或环境变量传入；
- 不依赖开发者机器的绝对路径；
- 对失败返回正确退出码。

## 13. 测试

至少实现并运行：

1. 注册表 schema 测试；
2. ModelScope ID 解析器的 mock 测试；
3. 已下载资源不重复下载测试；
4. 中断下载恢复测试（可用小文件模拟）；
5. 源/目标角色冲突必然失败测试；
6. choices 与答案映射测试；
7. 多图顺序和图片 hash 测试；
8. 重复样本检测测试；
9. 锁文件被篡改后验证失败测试；
10. 本地离线加载测试；
11. smoke profile 端到端测试。

测试不得依赖 full 模型全部下载完成。

## 14. 执行顺序

严格按以下阶段执行，每阶段保存报告：

### Phase A：审计与实现

- 审计仓库和环境；
- 建注册表、下载器、预处理器、校验器和测试；
- 运行单元测试。

### Phase B：资源解析

- 查询并验证 ModelScope ID；
- 记录候选、官方性、许可证、revision 和大小；
- 生成 dry-run 空间估计；
- 禁止下载 unresolved 资源。

### Phase C：Smoke

- 下载 smoke 资源；
- 预处理每个数据集的稳定 8 样本管线切片；
- 完成离线加载和 schema QA。

### Phase D：Core

- 只有 Smoke 全部通过且空间足够时执行；
- 下载 core 资源；
- 处理 core 全量数据；
- 完成 QA、角色隔离和锁定。

### Phase E：Full

- 仅当 `BENCHCOE_PROFILE=full` 时执行；
- 不得因为 full 可选资源失败而破坏已通过的 core；
- 对受限或不可用项输出人工处理清单。

## 15. 完成标准

只有满足以下条件才可宣称资源准备完成：

- 所选 profile 的所有必需 ModelScope 资源均已解析；
- 下载成功且验证通过；
- 数据已转换为统一 schema；
- 图片完整性检查通过；
- 模型可从本地离线加载，或被明确标为仅下载未运行验证；
- 源域/目标域角色无冲突；
- `asset_lock` 和 `protocol_lock` 均生成并校验；
- 单元测试和 smoke E2E 通过；
- 不存在把可选失败写成成功的情况。

若未达到，结论必须是 `PARTIAL` 或 `BLOCKED`，并列出：

- 已完成项；
- 阻塞项；
- 需要的人工授权/许可证；
- 缺少的磁盘或环境条件；
- 后续可安全执行的精确命令。

## 16. 最终回复格式

最终向用户汇报：

1. 当前分支与修改文件；
2. ModelScope 下载 profile；
3. 成功/缺失/受限的模型数量；
4. 成功/缺失/受限的数据集数量；
5. 数据和模型的实际存储路径；
6. 磁盘占用；
7. 测试结果；
8. `asset_lock`、`protocol_lock`、`preparation_report.md` 路径；
9. 是否可以进入实验提示词；
10. 若不能，给出唯一必要的下一步。

不要只给建议或伪代码。需要实际实现、测试并在条件允许时完成所选 profile 的下载与预处理；不得虚构下载或测试结果。

