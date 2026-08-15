# Codex 单文件提示词（二）：Bench-CoE Improve5 / 修正 Improve6 的完整动态专家实验

> 使用方式：先完成“ModelScope 资源下载、预处理与缓存”提示词，再把本文件全文交给 Codex。  
> 本提示词只使用第一阶段生成的本地锁定资源；不得在实验过程中临时下载数据/模型或用目标标签调参。  
> 目标是完成代码、缓存、分阶段实验、统计检验、图表和事实性报告；不得编造结果或在失败门禁后强行给出正向结论。

---

你是一名负责机器学习方法、实验工程和可重复性审计的高级研究员。请在当前 Bench-CoE / Subject-Bert-Bench-CoE 仓库中，实现并运行 Improve5（局部行为可靠性）与修正后的 Improve6（高置信有向失败—修复图）的动态专家池研究。

## 0. 研究问题

本实验必须区分以下因素：

\[
\text{源域证据规模 }N_s
\times
\text{专家数量 }M
\times
\text{专家结构（规模、能力、谱系与多样性）}.
\]

分别回答：

### Improve5 主问题

当专家池的行为空间维度扩大时，局部能力估计需要多少源域样本才能保持稳定？同家族规模增长与跨家族多样性，哪一种能以更少源样本形成可利用的局部互补性？

### Improve6 主问题

有向失败—修复关系在多少失败样本下能够可靠形成？这些关系能否跨域迁移，并在专家加入、退出、超时、源域漂移和标签噪声下保持有效？

### 共同问题

新增专家的收益究竟来自：

1. 更强的单模型能力；
2. raw instance oracle 自然上升；
3. 可被方法稳定识别的失败互补性；
4. 重复模型或随机猜测造成的假增益？

## 1. 硬约束

1. 先读取仓库 `AGENTS.md`、现有实现、先前 Improve1–6 实验结果、数据/模型锁和 protocol。
2. 保留用户已有修改和结果，禁止覆盖。
3. 本提示词不下载任何数据集或模型。缺少资源时停止相应阶段并输出缺失清单。
4. 所有模型输出只生成一次并缓存；后续 \(N_s\)、\(M\)、\(k\)、边阈值、专家顺序和随机种子实验必须基于缓存离线计算。
5. 目标测试标签不能参与任何模型池、超参数、prompt、解析器或子集选择。
6. 不允许继续把 \(C^2\) 解释成“两跳连续修复概率”。
7. 不允许把不同 prompt、不同 seed、不同 thinking 状态当作独立模型家族。
8. 不允许在看到目标结果后删除负迁移数据集或更改主要指标。
9. 不允许把未运行、失败或部分运行写成完成。
10. 若现有 Improve5/Improve6 定义与本文不同，建立版本化新实现并保留旧版本作为 legacy，不得静默替换。

## 2. 运行前门禁

验证第一阶段产物：

- `asset_lock.json` 和哈希；
- `protocol_lock.yaml` 和哈希；
- 模型、本地数据、图片路径；
- 所需资源状态；
- 源域/目标域角色；
- 仓库 Git commit；
- 已有模型输出缓存。

任何锁不一致时：

- 不要自动重新下载；
- 不要修改 protocol；
- 输出具体差异并停止受影响阶段。

建立本实验的新 `experiment_protocol.yaml`，在读取任何目标测试标签前锁定：

- 主要与次要数据集；
- 专家池；
- 模型推理设置；
- 方法版本；
- 超参数搜索空间；
- 源域内部验证策略；
- 主指标和次指标；
- 统计方法；
- seed；
- 失败与排除规则；
- 运行层级。

生成协议哈希。后续任何修改必须创建新版本并记录理由。

## 3. 三层执行配置

支持：

```bash
EXPERIMENT_TIER=smoke|core|full
```

### 3.1 Smoke

- 每个数据集固定 8–16 个由稳定 ID 选出的样本；
- 文本最多 3 个模型；
- 视觉最多 3 个模型；
- 2 个 seed；
- 只验证推理、缓存、解析、方法和聚合流程；
- smoke 结果不得进入论文主表。

### 3.2 Core

- 已锁定的 core 数据集；
- 文本/视觉分别使用可用的核心专家池；
- \(N_s\in\{64,128,256,512,1024\}\)；
- \(M\in\{3,5,8,11,14\}\)，若当前池不足则只运行合法值；
- 10 个预注册 seed；
- 主要负对照、动态加入/退出和成本实验；
- 生成论文主要表图。

### 3.3 Full

在 Core 基础上：

- \(N_s\) 增加 2048；
- 专家压力测试 \(M\in\{18,20\}\)，仅在实际有足够真实专家时；
- 32B/38B 强锚点；
- 完整冷启动、标签噪声、超时、源域多样性实验；
- 关键结论使用 20 seed；
- secondary/optional 数据集；
- 完整附录表图。

若资源只支持较小专家池，不得复制真实模型凑 \(M\)。Duplicate Expert 只能作为明确标注的负对照。

## 4. 专家元数据与专家池

建立或扩展 `expert_registry.yaml`，每个真实专家记录：

- 模型 ID、revision、本地路径；
- 家族、参数量、架构；
- 文本/视觉模态；
- instruction tuning；
- thinking/non-thinking；
- 精度与量化方式；
- 上下文和输出 token 上限；
- 源域准确率；
- 推理成本；
- 许可证状态；
- 可用数据集。

至少形成三类池：

### A. Scale-only

同家族、同代际、同推理条件，仅改变规模：

- 文本：Qwen3 1.7B / 4B / 8B / 14B / 32B（按实际资源）；
- 视觉：InternVL3.5 2B / 4B / 8B / 14B / 38B（按实际资源）。

### B. Lineage-diverse

在相近源域能力或 7B–14B 量级下选择不同家族。必须报告能力是否匹配，不能把明显更强的模型造成的增益误写为多样性增益。

### C. Hybrid

由同家族尺寸、跨家族模型、领域专家和一个强锚点组成。负对照专家不计入真实池大小。

## 5. 统一推理和输出缓存

### 5.1 推理条件

主实验固定：

- Instruct/chat 版本；
- temperature \(=0\)；
- greedy 或等价确定性解码；
- 相同的最大输出 token 预算；
- BF16 优先；
- 相同任务使用语义等价的 prompt；
- Qwen3 `enable_thinking=False`；
- thinking 实验若运行，作为单独敏感性分析并固定推理 token；
- 禁止 self-consistency 多次采样混入主实验。

记录：

- 最终序列化 prompt 哈希；
- 模型 chat template 哈希；
- tokenizer/processor revision；
- dtype、量化、attention backend；
- GPU 型号；
- 输入/输出 token；
- 串行时延、批处理时延；
- 峰值显存；
- 错误、超时、拒答和格式错误。

### 5.2 缓存 schema

每个 `sample × expert` 只允许一个主实验输出：

```json
{
  "schema_version": "benchcoe_prediction_v1",
  "run_id": "...",
  "sample_id": "...",
  "dataset": "...",
  "split": "...",
  "role": "...",
  "expert_id": "...",
  "model_revision": "...",
  "prompt_hash": "...",
  "decode_config_hash": "...",
  "raw_output": "...",
  "parsed_answer": "...",
  "parse_status": "ok|invalid|refusal|timeout|runtime_error",
  "input_tokens": 0,
  "output_tokens": 0,
  "latency_serial_s": 0.0,
  "latency_batch_s": 0.0,
  "peak_memory_bytes": 0,
  "created_at": "...",
  "record_hash": "..."
}
```

正确性标签单独生成，目标推理时不写入预测缓存。解析器版本锁定，失败、拒答和格式错误统一计错；禁止手工逐题修答案。

### 5.3 客观评分

- 主实验优先多项选择和 exact match；
- choices 必须通过标签与文本双重归一化；
- 开放答案数据（如 CharXiv）只能作补充；
- 如果必须用 LLM judge，单独报告 judge 模型、prompt、复核协议与一致性，绝不混入主指标；
- 代码题只有在可执行沙箱和官方测试均可用时进入补充实验。

## 6. 方法实现

## 6.1 Legacy 保留

保留并复现仓库已有：

- Bench-CoE；
- Improve5 legacy；
- Improve6/RepairChain legacy；
- 先前最佳配置。

旧结果必须通过原日志/缓存复核，不能手抄。

## 6.2 Improve5：局部行为可靠性

优先复用现有定义。若缺失，建立清晰版本化实现：

1. 在源域样本上构建每题的专家输出/行为表示；
2. 对目标题只使用无标签可观察行为寻找源域邻域；
3. 在邻域中估计每个专家的局部正确率；
4. 使用源域内部交叉验证选择必要参数；
5. 支持：

\[
k\in\{3,5,10,20,\lfloor\sqrt{N_s}\rfloor,\text{adaptive}\}.
\]

所有 \(k\) 必须裁剪到合法范围。`adaptive` 的规则在 experiment protocol 中预定义，不能根据目标结果修改。

至少记录：

- Brier Score；
- ECE；
- 所选专家跨源域子集的一致率；
- 邻域集合 Jaccard 稳定性；
- 专家选择熵；
- 负迁移目标比例；
- 邻域有效密度；
- 局部可靠性相对全局可靠性的增益。

## 6.3 修正后的 Improve6：高置信有向失败—修复图

删除原有 \(C^2\) 的概率链解释。实现一跳有向边：

\[
C_{ij}=P(Y_j=1\mid Y_i=0).
\]

对每个源域专家 \(i\)：

\[
n_i^-=\sum_x\mathbb I(Y_i=0).
\]

计数：

\[
n_{ij}^{01}=\sum_x\mathbb I(Y_i=0,Y_j=1),
\qquad
n_{ij}^{00}=\sum_x\mathbb I(Y_i=0,Y_j=0).
\]

Beta-Binomial 后验：

\[
C_{ij}\mid\mathcal D_s
\sim
\operatorname{Beta}
(\alpha_0+n_{ij}^{01},\beta_0+n_{ij}^{00}).
\]

用预注册的后验下分位数作为保守边权：

\[
W_{ij}=Q_{\delta}(C_{ij}\mid\mathcal D_s).
\]

只有满足以下条件才保留边：

\[
n_i^-\ge n_{\min},
\qquad
W_{ij}>\tau,
\qquad
W_{ij}-A_j^{\mathrm{global}}>\gamma.
\]

\(\alpha_0,\beta_0,\delta,n_{\min},\tau,\gamma\) 只能在源域内部验证上选择。

默认只使用一跳边。若实现多跳，必须直接估计条件一致的失败链，例如：

\[
P(Y_k=1\mid Y_i=0,Y_j=0),
\]

并要求足够联合失败样本；不得使用 \(C_{ij}C_{jk}\) 冒充连续修复概率。

记录：

- 每条边的后验均值、下置信界和区间宽度；
- 高置信边数量；
- 图稀疏度；
- Top-\(K\) 边 Jaccard 稳定性；
- 基准专家错误被至少一个专家覆盖的比例；
- 源域边权与目标真实条件修复率的 Spearman 相关；
- 图稳定性与目标增益的关系；
- 专家池变化前后的选路一致率。

方法正式名称若未锁定，代码使用中性版本名，例如 `repair_graph_v2`，不要在代码和结果中声称未查重的新论文名。

## 7. 必做基线

按层实现并版本化：

### 单模型

- source-best single；
- source-median single；
- largest single；
- cost-matched larger single。

所有“best”只能由源域选择。

### 简单聚合

- majority vote；
- source-accuracy weighted vote；
- confidence vote（仅当跨模型置信度可比且校准）；
- random expert；
- random subset with matched calls。

### 动态选择

- KNN local routing；
- OLA；
- LCA；
- MCB 思想的输出空间版本；
- META-DES 思想的动态选择版本；
- 源域输出向量上的逻辑回归；
- 小型 MLP；
- 正确性矩阵低秩分解/矩阵补全。

### 图与协作对照

- 仅全局准确率构图；
- 对称图；
- 简单一跳图消息传播；
- shuffled repair graph；
- 如仓库已有 Smoothie/Eagle 类实现，复现其准确版本；否则不得只用名称包装简化实现。

### 上界

- instance oracle；
- top-\(k\) oracle；
- cost-matched oracle。

Oracle 只用于分析，不参与方法选择。

所有基线使用完全相同的模型输出缓存、样本和成本记账。

## 8. 核心 \(N_s\times M\) 实验

### 8.1 源样本规模

Core：

\[
N_s\in\{64,128,256,512,1024\}.
\]

Full：

\[
N_s\in\{64,128,256,512,1024,2048\}.
\]

同一 seed 内使用嵌套子集：

\[
\mathcal D_{64}\subset\mathcal D_{128}\subset\cdots.
\]

按任务类别分层抽样；不能每个 \(N_s\) 独立抽样导致曲线不可解释。

### 8.2 专家数

\[
M\in\{3,5,8,11,14\},
\]

Full 可增加：

\[
M\in\{18,20\}.
\]

只有真实可用专家足够时才运行。每个 \(M\) 分别构造：

- strong-first；
- diversity-first；
- same-lineage-first；
- random；
- small-to-large；
- large-to-small；
- low-complementarity-first；
- high-complementarity-first。

其中强弱、多样性和互补性排序只能由源域计算，并在 protocol 中冻结。

### 8.3 输出

对 Improve5 和 Improve6 分别生成：

- Accuracy/Macro-F1 热力图；
- 负迁移率热力图；
- ECE/Brier 热力图；
- 邻域或边稳定性热力图；
- oracle gap closure 热力图；
- Accuracy–Cost 热力图。

## 9. 动态专家实验

### 9.1 专家逐步加入

按 8.2 的顺序逐个增加专家。每次记录：

\[
\Delta\mathrm{Acc},
\quad
\Delta\mathrm{Oracle},
\quad
\Delta\mathrm{Cost},
\quad
\Delta\mathrm{NegativeTransfer}.
\]

定义：

\[
\mathrm{Utilization}
=
\frac{\Delta A}{\Delta O+\epsilon},
\]

以及：

\[
\mathrm{OGC}
=
\frac{A_{\mathrm{method}}-A_{\mathrm{best\ single}}}
{A_{\mathrm{oracle}}-A_{\mathrm{best\ single}}}.
\]

分解新增模型收益来自 raw oracle 还是可利用互补性。

### 9.2 专家退出与失效

模拟：

- 随机删除一个专家；
- 删除源域最强专家；
- 删除修复图关键桥接专家；
- 同时删除同谱系 2–3 个专家；
- 10%、20%、30% 独立超时；
- 同谱系相关超时。

记录性能下降、选路变化、图重构时间和恢复后需要的源样本数。不能用目标标签重建图。

### 9.3 新专家冷启动

先用 \(M\) 个专家建立系统，再加入一个未出现的新专家，仅给：

\[
b\in\{0,8,16,32,64,\mathrm{full}\}
\]

道源域校准题。

评估：

- Improve5 行为空间扰动；
- 邻域 Jaccard；
- 新专家选择率；
- Improve6 新边数量、区间宽度和稳定性；
- 冷启动准确率/成本曲线；
- \(b=0\) 时必须采用预注册 fallback，不能借用目标标签。

## 10. 必做负对照

### Duplicate Expert

复制一个真实专家的全部缓存输出、成本和错误状态，作为虚拟副本。方法不应因副本明显获益。

### Random Guesser

按每题合法选项均匀随机生成，随机种子固定。报告 raw oracle 的自然变化，但实际方法不应稳定提升。

### Correlated Family

逐步加入同家族不同尺寸/检查点，与相同数量的异构模型比较。

### Shuffled Repair Graph

至少实现：

- 随机置换专家身份；
- 保持边权分布但交换边端点；
- 保持入度/出度近似的 degree-preserving shuffle。

### Source-label Noise

\[
\eta\in\{0,5\%,10\%,20\%,30\%\}.
\]

只污染源域训练标签，不污染源域验证真值和目标标签。每个 seed 独立注入并记录被翻转 ID。

### Random Expert Subset

在相同调用数和近似成本下随机选专家，作为成本匹配对照。

## 11. 源域数量与多样性

在固定总源样本预算下比较：

- 1 个源域全部样本；
- 2 个源域各一半；
- 4 个源域各四分之一；
- 根据源域内部估计的关系相似度加权。

需要回答：

\[
\text{更多标签}
\quad\text{vs}\quad
\text{覆盖更多失败关系}.
\]

文本和视觉分别运行，禁止跨模态混用不可比较的输出表示而不说明。

每种组合报告：

- 源域类别覆盖；
- 专家失败覆盖；
- 有效 \(n_i^-\)；
- 邻域/图稳定性；
- 目标准确率；
- 负迁移率；
- 成本。

## 12. 成本实验

至少报告：

- 每题平均模型调用数；
- 平均输入/输出 token；
- 总 token；
- GPU-hours；
- 串行时延；
- 理想并行时延；
- 实测批处理时延；
- 峰值显存；
- 若可获得价格，单题估计成本；价格来源和日期必须记录；
- 每提升 1 个百分点的成本；
- Accuracy–Calls；
- Accuracy–Tokens；
- Accuracy–Latency；
- Accuracy–GPU-hours Pareto 前沿。

比较：

- 最强单模型；
- 单个更大模型；
- majority；
- 完整专家池；
- top-\(k\) 专家；
- 两阶段级联；
- Improve5；
- Improve6；
- 成本匹配随机子池。

“完整调用全部专家后再选一个答案”与“生成前只路由少数专家”必须分开统计，不能混称低成本路由。

## 13. 指标

主指标：

- Accuracy；
- Macro-F1（类别可定义且不被任务规模误导时）；
- dataset-level average rank；
- 负迁移目标比例；
- worst-case target improvement；
- median target improvement。

可靠性：

- Brier Score；
- ECE；
- NLL（概率可用时）；
- 专家选择熵；
- 选择一致率。

互补性：

- raw oracle；
- top-\(k\) oracle；
- Utilization；
- OGC；
- pairwise error overlap；
- failure coverage。

Improve5：

- 邻域 Jaccard；
- 邻域密度；
- 局部可靠性方差；
- 不同源域子集的专家选择一致率。

Improve6：

- \(n_i^-\)；
- Beta 后验区间宽度；
- 高置信边数；
- 图稀疏度；
- Top-\(K\) 边 Jaccard；
- 源/目标修复率 Spearman；
- 关键桥接专家敏感性。

## 14. 统计协议

Core 使用 10 个预注册 seed；Full 的关键结论使用 20 个。

至少实现：

- paired bootstrap 95% CI，按相同题目成对重采样；
- 方法与主要基线的成对检验；
- 多数据集平均排名；
- 多重比较校正；
- effect size；
- seed 间方差；
- dataset-level 与 pooled 两种结果，但主结论优先 dataset-level；
- 不将样本多的数据集自动赋予更高论文权重。

对于小差异，只有置信区间和效应量支持时才称为提升。

保存每次 bootstrap/检验的 seed、样本索引摘要和原始结果。

## 15. 失败分析与适用边界

自动生成按以下维度的分析：

- 数学、知识、逻辑、代码、视觉感知、空间关系、图表；
- 文本 vs 视觉；
- 同家族 vs 跨家族；
- 小模型 vs 大模型；
- 强专家低错误样本；
- 所有专家共同失败；
- 高/低源—目标关系稳定性；
- 不同 \(n_i^-\)；
- 不同专家池规模；
- 不同 source-label noise。

尝试从结果中识别工作区间，例如：

\[
n_i^-\ge n_{\min},
\quad
M\le M_{\max},
\quad
\text{edge stability}\ge\tau_s.
\]

但阈值必须由源域内部验证或预注册分析得出；不得为了美化目标结果后设。

当证据不足时，结论写成“未确定”或“负结果”，不能强行提出普遍规律。

## 16. 分阶段执行与科学门禁

### Phase 0：审计与 protocol lock

- 验证资源锁；
- 复核现有代码和缓存；
- 写 experiment protocol；
- 不读取目标标签做探索。

### Phase 1：实现与单元测试

实现方法、基线、缓存、指标、统计、图表脚本。

至少测试：

- 目标标签不能进入 fit；
- 嵌套源子集；
- Beta 后验计数；
- \(C^2\) 不存在于新版路径；
- shuffled graph 保持预期统计；
- Duplicate Expert；
- Random Guesser；
- 成本记账；
- 缓存幂等和哈希；
- 中断恢复；
- 小型人工矩阵上的方法正确性。

### Phase 2：Smoke E2E

跑完整小流程。任何 schema、标签泄漏、解析或缓存错误都必须先修复。

### Phase 3：输出缓存

- 使用本地模型和锁定数据生成缺失缓存；
- 已验证缓存不重复生成；
- 每个数据集/模型单独 checkpoint；
- 失败可恢复。

### Phase 4：源域开发

- 只在源域内部选择所有参数；
- 冻结最终配置；
- 生成最终配置哈希。

### Phase 5：Core 主实验

- \(N_s\times M\)；
- 必做基线；
- 必做负对照；
- 主要动态加入/退出；
- 成本分析。

### Phase 6：一次性目标评估

- 目标标签只在冻结配置后打开；
- 输出完整结果，不得挑选正向数据集；
- 将目标评估标记为不可回滚的 protocol event。

### Phase 7：Full 与外部验证

- 只有 Core 工程门禁通过时运行；
- 若科学结果为负，仍可运行预注册的 Full，但不得临时加只为追求正结果的配置；
- LiveBench 必须记录版本日期和哈希；
- HLE/CharXiv 只作为 secondary。

### Phase 8：统计、图表和报告

从机器可读结果自动生成，不手抄数字。

## 17. 工程门禁

以下任一失败则禁止生成论文主结论：

- 目标标签泄漏；
- protocol hash 不一致；
- 缓存 schema 不一致；
- 模型 revision/prompt hash 混用；
- 正确性解析器未锁定；
- 数据角色冲突；
- 运行缺失被当作错误以外的有利处理；
- 结果表与原始 JSON 不一致；
- seed 或样本子集不可复现；
- shuffled/duplicate 负对照实现错误。

## 18. 科学判定

不要预设 GO。至少给出：

- `GO`：多数据集、多个 seed、成本匹配基线下稳定改善，负迁移可控，结构稳定性支持机制；
- `CONDITIONAL`：仅在明确工作区间有效；
- `NO-GO/NEGATIVE`：不能超过强简单基线、成本过高、结构不迁移或负迁移严重。

无论结论如何都保留全部结果。不得因为 NO-GO 自动重调目标相关参数。

## 19. 输出结构

适配仓库现有结构；若不存在，可使用：

```text
configs/experiments/
  dynamic_experts_smoke.yaml
  dynamic_experts_core.yaml
  dynamic_experts_full.yaml
  experiment_protocol.yaml
benchcoe/
  methods/
    improve5_local_reliability.py
    repair_graph_v2.py
  baselines/
  evaluation/
  statistics/
  cost/
tools/
  run_inference_cache.py
  run_dynamic_experiments.py
  validate_results.py
  make_paper_tables.py
  make_paper_figures.py
tests/
results/dynamic_experts/
  protocol/
  manifests/
  caches/
  raw/
  aggregated/
  statistics/
  tables/
  figures/
  reports/
```

必须生成：

- `run_manifest.json`；
- `completion_matrix.csv`；
- `failures.jsonl`；
- `metrics_long.csv`；
- `metrics_summary.csv`；
- `statistical_tests.csv`；
- `cost_metrics.csv`；
- `edge_posteriors.parquet`；
- `neighbor_stability.parquet`；
- `scientific_gate.json`；
- `final_experiment_report.md`。

## 20. 必需图表

至少自动生成：

1. \(N_s\times M\) Accuracy 热力图；
2. \(N_s\times M\) 负迁移率热力图；
3. Improve5 邻域稳定性曲线；
4. Improve6 边区间宽度与 \(n_i^-\) 曲线；
5. 源域边权与目标修复率散点图；
6. strong/diversity/same-lineage/random 加入轨迹；
7. Duplicate/Random Guesser 负对照；
8. 专家退出与超时韧性；
9. 冷启动 \(b\) 曲线；
10. Accuracy–Cost Pareto；
11. raw oracle、实际提升和 Utilization 分解；
12. dataset-level forest plot，含 95% CI。

所有图：

- 从结果文件生成；
- 使用统一学术配色；
- 保存 PDF/SVG 和 PNG；
- 字体、坐标、图例可读；
- 不截断误导性坐标；
- 图注包含样本数和 seed。

## 21. 结果校验

实现一个独立验证脚本，从原始预测和 protocol 重新计算关键表格。

检查：

- canonical runs 是否全部完成；
- 每个 sample × expert 是否唯一；
- 缓存哈希；
- 方法是否只 fit 源域；
- 目标结果是否在冻结后生成；
- 表格数字与原始数据一致；
- CI 和 p-value 可复现；
- 成本总和与逐题记录一致；
- 失败/超时没有被丢弃；
- 专家池顺序与 protocol 一致。

结果不完整时，主表相应单元显示 `MISSING`，不得用 0、均值或旧实验填充。

## 22. 最终报告

`final_experiment_report.md` 必须事实性包含：

1. 仓库 commit、分支、环境；
2. protocol 和 asset lock 哈希；
3. 实际运行的数据集、模型、样本和 seed；
4. 完成矩阵；
5. Improve5 和 Improve6 主结果；
6. 与强简单基线和成本匹配基线比较；
7. \(N_s\times M\) 结论；
8. scale-only vs lineage-diverse；
9. 动态加入/退出/冷启动；
10. 负对照；
11. 边与邻域稳定性；
12. 统计显著性和效应量；
13. 成本；
14. 失败模式；
15. 可用工作区间；
16. `GO`、`CONDITIONAL` 或 `NO-GO/NEGATIVE`；
17. 未完成项和原因；
18. 可复现命令。

报告必须区分：

- 实际观察；
- 统计支持的结论；
- 推测；
- 未验证假设。

## 23. 最终回复格式

向用户汇报：

1. 实际完成的代码和实验阶段；
2. 分支和 commit 状态；
3. canonical runs 完成数/计划数；
4. 工程门禁；
5. 科学判定；
6. 最重要的正/负结果；
7. 关键表图和报告路径；
8. 未完成或阻塞项；
9. 一条最安全的下一步。

不要只提供计划。请实际实现、测试、按可用资源运行、保存可恢复 checkpoint，并基于真实结果报告。若完整实验耗时很长，优先保证 Phase 0–3 和 Core 可恢复执行，不得虚构后续结果。
