# ORBIT-CoE：面向语言与视觉语言 OOD 的下一代 Bench-CoE 开创性方案

日期：2026-07-17  
依据：`chatgpt_pro_breakthrough_innovation_prompt_all_datasets(2).md` 及此前 Improve1–6、RIFT-CoE、MIRAGE-CoE 结论  
状态：原创研究设计与可执行实验方案；本文不虚构尚未运行的结果

## 总体判断

现有路线已经走过了四个层次：

```text
学科映射
  -> 题面能力表示
  -> 多专家输出失败生态
  -> 可路由性审计与保守切换
```

下一轮不应继续增加另一个 failure signature，也不应继续把全专家最终答案当成唯一协作对象。真正缺失的是三类信息：

1. **无目标标签时，怎样估计哪个专家适合作为目标域基准，而不把同家族模型的一致误判成可靠性？**
2. **怎样获得目标形态下、但不使用目标答案的“有真值能力测量仪器”？**
3. **怎样让专家传递可验证的局部证据，而不是只交换最终答案或长篇解释？**

本文提出 **ORBIT-CoE（OOD Reliability by Behavioral Instruments and Transportable Evidence）**。它不是替代 RIFT，而是为 RIFT 提供更可靠的 base、候选和证据：

- **O — Oriented reliability**：源域定向、谱系感知的无标签目标可靠性估计；
- **R — Residual evidence**：剥离位置、长度、选项表面吸引力后的残差投票证据；
- **B — Behavioral instruments**：具有程序化真值、目标形态匹配、能最大区分专家的能力测量题；
- **I — Identifiability mask**：在全部可容许真值世界中判断当前样本是否真正可路由；
- **T — Transportable evidence**：可跨专家盲传、闭环验证的最小证据与感知—推理接力图。

核心主张从“选择最合适模型”提升为：

> 利用 benchmark leaderboard 构造专家池后，先用无标签行为矩和自动真值仪器重建目标域专家可靠性，再只对可识别样本传递可复核证据并执行安全切换。

---

# A. 已有方法的根本瓶颈

## A.1 原始 Bench-CoE 为什么失败

### 1. 路由标签不是目标任务中的稳定因果变量

原始流程学习“Math/Physics/Chemistry -> expert”，但 GPQA、MathVista、MMMU-Pro 的决定性因素通常是：

- 科学事实深度；
- 多步逻辑与数值推导；
- OCR、图表、几何、空间关系；
- 图文指代与跨模态对齐；
- 选项辨析和输出协议。

学科只是表面变量，同一学科内不同 benchmark 的能力结构可能完全不同。

### 2. 源最强专家不等于目标最强专家

原始方法把 source subject ranking 当作不变量。实际上模型家族、prompt 适配、图像分辨率、选项数和任务难度都会造成 ranking reversal。

### 3. 原始目标是预测 expert ID，而不是预测相对修复价值

在 OOD 中，绝对正确率难迁移；更可迁移的对象通常是“候选是否能修复固定基准、是否会破坏基准正确答案”。Improve4–6 的成功已经支持这一点。

### 4. 多专家一致性混入家族复制与选项偏置

三个同架构、同训练数据、相近 prompt 的模型一致，不等于三份独立证据；它可能只是一个错误被复制三次。与此同时，模型对首项、末项、较长选项或词面重叠选项存在不同偏好。

## A.2 Improve5/6 为什么在 BBH 和 MMLU-Pro 成功

### BBH

- beyond-coverage oracle 为正；
- 任务的失败模式具有可重复结构，如量词、排序、逻辑组合、符号操作；
- 某些专家确实稳定修复另一些专家；
- 输出分歧能暴露推理失败，而不仅是随机覆盖；
- 样本量足以学习 correction graph。

因此 FATE/ECR/RepairChain 能把“可靠少数派”与“共错团体”区分开来，FATE 达到约 77.98%，较 best single 68.05% 提升约 +9.92%，且 paired CI 明显为正。

### MMLU-Pro

- source validation 与 test 的任务生成机制较接近；
- 专家修复关系的跨域漂移小于跨 benchmark 设置；
- coverage null 46.57%，raw oracle 89.22%，存在明显 beyond-cover 空间；
- ECR 达到 60.31%，较严格池 best single 57.61% 提升 +2.69%，CI 为正。

这两个成功案例共同说明：**失败生态只有在 repair relation 可迁移且输出中含有可识别信号时才有效。**

## A.3 GPQA 为什么只有小幅提升

1. Raw oracle 86.16% 低于 coverage null 91.60%，高 oracle 主要来自多模型覆盖不同选项。
2. 强弱专家都可能缺少同一个高难科学事实，错误不是从语言表面可证伪的。
3. 弱专家数量多时会形成“流畅但无知识依据”的多数。
4. MMLU/GAOKAO 的科学题不能充分代表 GPQA 的专家级知识深度。
5. RepairChain +0.92% 的 paired 95% CI 为 [-0.59%, +2.16%]，不能排除抽样波动。

GPQA 的瓶颈不是再多一个 output feature，而是缺少 **目标形态下可自动评分的科学能力仪器** 和 **答案背后的可迁移证据**。

## A.4 MMStar 为什么基本不可提升

- text-only best single 24.47% 接近四选一随机水平；
- coverage null 98.50%，raw oracle 80.20%；
- 14 个模型几乎覆盖全部选项，但题面没有足够视觉证据；
- 在这种条件下，答案簇大小、少数派和纠错图都可能只是选择偏置。

所以 MMStar text-only 的首要目标应是识别 `unroutable`，而不是强行选出某个“专家”。只有证明某个子集在不看标签时具有真实可识别信号，才允许路由。

## A.5 CMMMU 与 MMMU-Pro 为什么难

### CMMMU

- 去掉 Qwen3-VL-4B 后 best single 38.56%，DARE 39.44% 但 CI 跨 0；
- coverage null 约 95%，raw oracle 约 79%；
- 多个 VLM 可能读取了相同错误区域，答案一致不等于视觉证据一致；
- 中文 OCR、复杂版面与学科知识混在一起，最终答案无法定位失败阶段。

### MMMU-Pro

- 10-options 使普通一致性天然稀疏；
- 感知、OCR、外部知识、推理、选项映射同时作用；
- 同一个 VLM 可能感知强但推理弱，另一个相反；
- 选择单个“全能专家”浪费了阶段互补性。

## A.6 GAOKAO-MM 低分如何污染视觉迁移

当前 best single 9.91%、subject best 11.15%、oracle 17.96%，远低于正常 VLM 能力预期。若这是 prompt、拼图、截断和抽取错误造成的，则：

1. source correctness matrix 学到的是“协议适配能力”，不是视觉语言能力；
2. repair graph 把格式修复误当成认知修复；
3. source global best 与 target base 都会选错；
4. source LOBO 无法为 CMMMU/MathVista/MMMU-Pro 提供可信阈值；
5. 任何复杂视觉路由都会被错误源监督限制。

因此视觉主实验前必须先修复源评测契约。

---

# B. 八个原创候选方法

## B0. PARD：Protocol-Aware Reliability Decomposition

### 1. 方法名称

**PARD（协议感知可靠性分解）**：先把 GAOKAO-MM 的能力错误、输出协议错误和解析错误分开。

### 2. 核心直觉

一个模型被判错可能有三种原因：

\[
\text{Failure}=\text{Cognitive Error}
\lor\text{Generation Contract Error}
\lor\text{Parser Error}.
\]

只有第一项能用于学习专家能力。

### 3. 与已有 routing/ensemble 的区别

PARD 不是 router，而是 leaderboard-derived routing 的测量层。原 Bench-CoE 默认排行榜分数可直接比较；PARD 把排行榜本身视为带测量误差的观测。

### 4. 使用信息与污染说明

- 使用 GAOKAO-MM 源标签、已有 raw outputs、图片、选项和模型家族；
- 可重新调用模型，但只调用有标签源数据；
- 不读取 CMMMU、MathVista、MMMU-Pro 标签。

### 5. 训练/校准

1. 把 646 个源样本固定拆成 parser-dev / prompt-dev / held-out-source-test；
2. 并行运行多种答案抽取器：严格标记、最后答案、选项文本匹配、JSON parser；
3. 用 source-dev 选择抽取规则，不能在 held-out source 上再改；
4. 为 Qwen、InternVL、MiniCPM、LFM、SmolVLM 等家族固定生成模板；
5. 统一图像输入、分辨率、帧顺序、max tokens 和停止符；
6. 对 half-credit 科目单独验证计分语义。

### 6. 推理流程

源评测输出统一为：

```json
{
  "status": "valid|truncated|format_error|no_answer",
  "answer": ["A"],
  "evidence_span": "...",
  "parser_conflict": false
}
```

target 仅复用冻结后的协议和 parser，不利用 target correctness。

### 7. 样本例子

模型输出：“综合可知应选择第三项，即 C。”旧规则因为缺少 `【答案】 C <eoa>` 判错；PARD 的 option-text/last-answer parser 提取 C，并在 source-dev 上证明该规则不增加误抽取后，才允许修复。

### 8. 可能提升的数据集

直接改善 GAOKAO-MM source leaderboard；间接改善 CMMMU、MathVista、MMMU-Pro 的 source base、repair graph 和能力仪器定向。

### 9. 失败原因

若协议修复后 best single 仍低于约 15%、oracle 仍低于约 25%，说明 GAOKAO-MM 专家池或输入构造本身不足，不能继续把它当唯一视觉源域。

### 10. 快速验证

无需重新调用时，先对缓存 raw outputs 运行多 parser disagreement audit，计算“旧判错但新 parser 可恢复”的比例。

### 11. 输出与消融

- `protocol_audit.csv`
- `parser_confusion_by_family.csv`
- `truncation_and_format_rates.csv`
- `repaired_source_leaderboard.csv`
- 消融：旧 parser / 多 parser / family parser / 统一 prompt / 完整 PARD。

## B1. LEAF-CoE：Lineage-Equivariant Agreement Factorization

### 1. 方法名称

**LEAF-CoE（谱系等变一致性分解）**。

### 2. 核心直觉

目标无标签输出中可以估计专家可靠性，但不能把 14 个模型视为条件独立标注者。先把共享架构、训练谱系和源错误模式相近的专家压缩成 lineage factors，再用目标二阶/三阶一致性矩估计每个谱系的目标混淆矩阵；源标签只用于确定潜变量的方向和类别排列。

### 3. 与普通 EM、majority、DARE 的区别

- 普通 majority 把每个模型当等权独立证据；
- target-unlabeled EM 容易被多数同源模型锁定错误解；
- DARE 主要迁移源稳定性；
- LEAF 用 **三阶矩可识别性 + 谱系商空间 + source orientation + OOD shrinkage** 估计目标域 base。

无标签谱方法估计分类器准确率已有经典基础，但通常针对二分类和条件独立分类器；依赖分类器的扩展也早已存在。参见 [Jaffe et al., 2015](https://proceedings.mlr.press/v38/jaffe15.html) 与 [Jaffe et al., 2016](https://proceedings.mlr.press/v51/jaffe16.html)。LEAF 的可主张新增点是 Bench-CoE 的多选项、模型谱系、源定向、目标 OOD base selection 与 RIFT repair 接口。

### 4. 使用信息

- 源 correctness、源/目标答案矩阵、模型家族/参数量、题面和选项数；
- 不使用目标标签；
- 完全可用现有缓存运行。

### 5. 训练/校准

1. 在源域按 error mutual information、架构家族和输出相似度聚类 lineage；
2. 每个 lineage 学习内部专家偏移；
3. 在目标答案 one-hot 张量上计算跨 lineage 二阶和三阶中心矩；
4. 分解得到 latent label prior 与 lineage confusion tensors；
5. 用源 confusion 确定 latent component 对应 A/B/C/D/…；
6. 用 source LOBO 选择 shrinkage 强度和最少 lineage 数；
7. 对 target 样本 bootstrap 得到每个专家目标可靠性下界。

### 6. 推理

目标 base 不再取 `source_global_best`，而取：

\[
b_T=\arg\max_e \operatorname{LCB}igl(\widehat{Acc}_T(e)\bigr).
\]

然后 FATE/ECR/ORBIT 候选均相对该 base 计算修复价值。

### 7. 样本例子

Qwen 系三个模型都选 C，InternVL 与 MiniCPM 分别选 B。普通一致性认为 C 有三票。LEAF 发现三个 Qwen 属于同一 lineage，只贡献约一份相关证据；两个独立 lineage 对 B 的支持使 B 的 latent posterior 更高，并可能把 InternVL 选为目标 base。

### 8. 数据集作用

- GPQA：抑制多个弱同源模型形成的流畅错误共识；
- CMMMU/MMMU-Pro：防止同架构视觉模型复制相同 OCR/grounding 错误；
- RIFT：减少总是回退错误 source global best 的问题。

### 9. 失败原因

- 独立 lineage 少于 3 个时三阶矩不可稳定识别；
- 所有模型共同错时无解；
- target 样本太少时 tensor estimate 方差大；
- source orientation 在极端 ranking reversal 下可能错误。

### 10. 快速验证

先在 source leave-one-benchmark-out 中隐藏标签，仅用预测矩阵估计 base，再与真实 held-out best single 比较：top-1 base 命中率、regret、Spearman。

### 11. 输出与消融

- `lineage_clusters.json`
- `target_moment_tensors.npz`
- `target_reliability_lcb.csv`
- `leaf_base_predictions.jsonl`
- 消融：无 lineage / 仅二阶 / 二阶+三阶 / 无 source orientation / 完整 LEAF。

## B2. BRES-CoE：Bias-Residual Evidence Scoring

### 1. 方法名称

**BRES-CoE（偏置残差证据评分）**。

### 2. 核心直觉

“三个专家选 D”不是三份同等证据。若 D 是最长选项、与题干词面最重合、位于该模型偏好的位置，则投 D 很可能来自选项吸引力。应先估计每个专家的错误选择倾向，再计算其投票超过这一倾向的残差信息。

### 3. 与 weighted vote/ECC 的区别

- weighted vote 按全局可靠性加权；
- ECC 按错误共现修复；
- BRES 估计 **专家 × 选项表面属性 × 错误状态** 的选择机制，把“票”转换为相对选项偏置空模型的 likelihood ratio。

### 4. 使用信息

- 源标签、源输出、目标无标签输出、选项位置/长度/词面/数值特征；
- VLM 可加 OCR overlap、图像指代词、单位和坐标词；
- 不使用目标标签；完全缓存可实现。

### 5. 训练/校准

对每个专家或 lineage 学习：

\[
s_e(a,\phi)=
\log\frac{P_S(A_e=a\mid Y=a,\phi)+\epsilon}
{P_S(A_e=a\mid Y\neq a,\phi)+\epsilon},
\]

其中 \(\phi\) 是选项位置、长度、题干重叠、数值大小、否定词等。使用 source LOBO 校准截断和 shrinkage。

### 6. 推理

对候选答案 \(a\)：

\[
Score(a)=\sum_{g\in\text{lineages}}
\operatorname{cap}_{g}\left(
\sum_{e\in g}\mathbf1[A_e=a]s_e(a,\phi)
\right).
\]

只把 `Score(candidate)-Score(base)` 交给 RIFT/ORBIT 安全仲裁。

### 7. 样本例子

四个模型选 D，一个模型选 B。D 是最长选项，并且四个模型在源域错误时有明显“长选项偏好”，所以四票的残差证据很低。选 B 的专家平时偏好 A/C，但在该题逆偏置选择 B，且其 source conditional likelihood ratio 高，B 可能成为可靠少数派。

### 8. 数据集作用

- GPQA：抑制措辞最专业但错误的长选项共识；
- MMStar/CMMMU：区分选项位置覆盖与真实语义证据；
- MMMU-Pro 10-options：普通票数稀疏时 likelihood ratio 比票数更有信息。

### 9. 失败原因

- source/target 的选项编写风格差异过大；
- 偏置特征太强会误删真实知识信号；
- 每个专家源错误样本不足。

### 10. 快速验证

在缓存输出上计算“原始票数边际”与“残差证据边际”，用 source LOBO 比较其对 `candidate repairs base` 的 AUROC/AUPRC。

### 11. 输出与消融

- `option_surface_features.parquet`
- `expert_bias_models.pkl`
- `bias_residual_scores.jsonl`
- 消融：position only / length only / semantic overlap / lineage cap / full BRES。

## B3. QUID-CoE：Quotient Unroutability Identification by Decoy Worlds

### 1. 方法名称

**QUID-CoE（基于诱饵真值世界的商空间不可路由识别）**。

### 2. 核心直觉

Coverage null 只问“专家覆盖了几个选项”，仍没有回答：在保持专家相关性、选项偏置和允许的跨域漂移后，是否存在多个完全不同的目标真值世界都能解释当前输出？若存在，则该样本不可识别。

### 3. 与 RIFT interval 的区别

RIFT 为某个候选相对 base 估计 repair interval；QUID 在 target-wide 层面构造所有与无标签输出统计兼容的 latent truth worlds，先判断“真值方向是否可识别”。它是 RIFT 之前的 identifiability mask，而非另一个 margin。

### 4. 使用信息

- 源 confusion/repair 区间；
- 目标答案矩阵、BRES option bias、LEAF lineage moments；
- 不使用目标标签；完全缓存可实现。

### 5. 训练/校准

构造可容许集合 \(\Theta_T\)：

- target label prior 在预注册区间内；
- lineage confusion 与 source/twin confusion 的距离不超过 LOBO 漂移半径；
- 生成的输出二阶/三阶矩落入 target bootstrap CI；
- option choice propensity 与 BRES 估计相容。

对每题、每候选计算：

\[
L_i(a)=\inf_{\theta\in\Theta_T}P_\theta(Y_i=a\mid A_{i,1:E}),
\quad
U_i(a)=\sup_{\theta\in\Theta_T}P_\theta(Y_i=a\mid A_{i,1:E}).
\]

### 6. 推理

- 若一个候选的 \(L_i(a)\) 高于所有其他候选的 \(U_i\) 加阈值，标记 `identified`；
- 若多个真值世界给出不同最优答案，标记 `unroutable`；
- `unroutable` 样本固定使用 LEAF base，不允许 FATE/ECR 切换。

### 7. 样本例子

14 个模型覆盖 A/B/C/D。世界 1 假设 Qwen lineage 可靠，C 为真；世界 2 假设 InternVL lineage 在该 target 可靠，B 为真。两个世界都能重现相同答案拓扑和允许的源漂移，则该题不能从当前输出识别，QUID 拒绝路由。

### 8. 数据集作用

最适合 MMStar text-only、CMMMU 和 GPQA：避免把覆盖多选项误判成可修复机会。

### 9. 失败原因

可容许集合过宽会几乎全部拒绝；过窄会产生伪识别。宽度只能用 source LOBO 固定，不能看目标准确率调。

### 10. 快速验证

先用离散 pattern cluster 而非逐题连续优化：对相同 lineage vote pattern 的样本共享一个 LP，可低成本运行。

### 11. 输出与消融

- `admissible_world_constraints.json`
- `candidate_probability_bounds.jsonl`
- `routability_mask.jsonl`
- 指标：identified coverage、source LOBO risk-coverage、最终 target coverage-risk；
- 消融：coverage-only / +lineage moments / +bias / +drift / full QUID。

## B4. ECI-CoE：Executable Capability Instruments

### 1. 方法名称

**ECI-CoE（可执行能力仪器）**。

### 2. 核心直觉

目标标签不可用，但可以生成“形态像目标、答案由程序确定”的新题，用作测量专家能力的实验仪器。它既不是 target probe，也不是让 LLM 自己生成答案的普通 synthetic routing data。

### 3. 与已有 generated-data routing 的区别

ACL 2026 的 Routing with Generated Data 使用生成器根据任务描述产生 query/answer，并强调生成器自身正确性与模型区分度，CASCAL 仍依赖共识和聚类。参见 [Routing with Generated Data](https://aclanthology.org/2026.acl-long.1498/)。

ECI 的不同点：

1. gold 由程序/符号求解器/渲染引擎确定，不信任生成模型自答；
2. probe 分布由目标无标签结构匹配；
3. probe 通过最优实验设计主动最大化专家 pairwise ranking 的 Fisher information；
4. 目标是定向 LEAF、选择 base 与 stage role，而非训练普通 query-only router。

### 4. 使用信息

- 目标无标签题面、图片统计、选项结构；
- 模型先验和源结果；
- 新生成 probe 的自动真值；
- 不使用任何目标答案。

### 5. 生成/校准

文本仪器：

- 程序化逻辑、约束满足、计数、单位换算；
- 化学计量、物理方程、数值量纲；
- 具有可控 distractor hardness 的科学选择题。

视觉仪器：

- OCR 字体/遮挡/旋转；
- 柱状图、折线图、散点图、表格查询；
- 几何、角度、相对位置；
- 多 panel 对齐、图例映射、图文指代；
- 渲染器同时保存场景图、OCR 真值和答案。

选择模板参数 \(\psi\)：

\[
\psi^*=\arg\max_\psi
\mathbb E[\operatorname{InfoGain}(\text{expert ranking}\mid x_\psi)]
-\lambda d(\phi(x_\psi),\phi_T).
\]

### 6. 推理

ECI 不直接处理目标题。它产生目标形态能力向量：

\[
r_e=[OCR,Chart,Geometry,Logic,Science,Grounding,Mapping,\ldots],
\]

用于 LEAF orientation、base 选择、RELAY stage assignment 和 RIFT prior。

### 7. 样本例子

目标 MMMU-Pro 中大量题具有“小字号双轴图 + 十选项”。ECI 自动生成同类双轴图，程序知道正确读数，逐步增加字号、轴密度和 distractor 相似度。若 MiniCPM OCR 稳定但科学推理弱，InternVL 相反，就能识别后续接力组合。

### 8. 数据集作用

- GPQA：补足高难科学推理的 target-shaped 能力测量，但只覆盖可程序化部分；
- CMMMU/MMMU-Pro/MathVista：直接测量 OCR、图表、几何和 grounding；
- 修复 GAOKAO-MM 源信号不足。

### 9. 失败原因

- synthetic-real gap；
- 自动题过于规则，模型可利用模板；
- GPQA 的深层事实知识难以程序化生成；
- probe 数不足以区分专家。

### 10. 快速验证

先生成 300–500 个低成本 probe，仅选当前缓存模型中的 4–6 个代表专家运行。若 ECI 排名在 source LOBO 上与真实 held-out 排名 Spearman < 0.3，停止扩大。

### 11. 输出与消融

- `eci_templates/`
- `eci_manifest.jsonl`
- `eci_ground_truth.jsonl`
- `eci_expert_outputs/`
- `eci_capability_vectors.csv`
- 消融：随机生成 / target-matched / executable-only / info-gain design / full ECI。

## B5. CAST-CoE：Choice-Axiom Stress Testing

### 1. 方法名称

**CAST-CoE（选择公理压力测试）**。

### 2. 核心直觉

正确答案的语义不应随选项字母和位置改变。一个专家若在语义等价 choice sets 中频繁改变答案，说明其当前回答主要由位置、标签或 distractor interaction 驱动，而非稳定知识。

### 3. 与普通 option shuffle/self-consistency 的区别

LLM 对选项顺序敏感已有充分研究，甚至可出现很大性能差异；参见 [Large Language Models Sensitivity to the Order of Options](https://arxiv.org/abs/2308.11483)。LLM 排序还会违反 transitivity、reversibility 和 independence of irrelevant alternatives；参见 [Measuring the Inconsistency of LLMs in Preferential Ranking](https://arxiv.org/abs/2410.08851)。

CAST 不把 shuffle 当数据增强，也不多数表决多个版本，而是构造每个专家的 **choice-axiom violation vector**，在源域学习哪类违反意味着该专家当前答案不可靠，再用于 target routing。

### 4. 使用信息与污染说明

- 目标题面、图片、选项；
- 重新调用模型的语义等价版本；
- 不读取正确答案；
- 变换规则和评分阈值只在 source/ECI 上选择。

### 5. 校准

压力测试包括：

1. 位置置换等变性；
2. A/B/C/D 改为 ①/②/③/④；
3. 同义改写所有选项并映射回原语义；
4. 对专家原 top-2 做 pairwise preference，检测循环；
5. 对视觉题保持图像不变，只改变 option presentation。

源域学习：

\[
P(C_e=1\mid v_e,\text{task atom}),
\]

其中 \(v_e\) 包含 position equivariance、pairwise cycle、label dependence 和 semantic stability。

### 6. 推理

只对 QUID 标记为潜在可路由、但 BRES/LEAF 仍冲突的样本调用 2–3 个变体。得到专家稳定性后更新候选 evidence score。

### 7. 样本例子

专家 A 原选 C；选项重排后仍选择“第三个位置”，映射回语义变成 A；专家 B 无论位置如何都选择同一语义选项 D。即使 A/B 原答案票数相同，CAST 会显著降低 A 的证据权重。

### 8. 数据集作用

GPQA、MMStar、CMMMU、MMMU-Pro 都是选择题，尤其 coverage null 高时可剥离 position-driven coverage。

### 9. 失败原因

稳定错误专家仍会通过；同义改写可能引入语义漂移；额外调用成本高。必须以 source 校准后的“稳定性对正确性的增量信息”决定是否保留。

### 10. 快速验证

先对源域 200 个分歧样本、3 个代表专家运行位置置换；若在控制 source confidence 后 violation 对错误没有增量 AUROC，则停止 target 调用。

### 11. 输出与消融

- `choice_transform_manifest.jsonl`
- `choice_axiom_violations.parquet`
- `cast_reliability_updates.jsonl`
- 消融：单 shuffle / 多 shuffle vote / violation vector / task-conditioned CAST。

## B6. RELAY-CoE：Typed Capability Relay Graph

### 1. 方法名称

**RELAY-CoE（类型化能力接力图）**。

### 2. 核心直觉

MMMU-Pro/MathVista 中，最强感知模型不一定是最强推理模型。不要只在完整模型之间选一个最终执行者，而把专家拆成受约束的角色：

```text
visual sensor -> typed evidence packet -> reasoner -> option mapper
```

### 3. 与 MoA、普通多智能体和现有模块化 VQA 的区别

VISTA 已通过信息瓶颈把 VLM sensor 与文本 reasoner 分开；EAGLE 强调多 VLM 共识必须与视觉区域证据对齐。参见 [VISTA](https://arxiv.org/abs/2512.22183) 与 [EAGLE](https://arxiv.org/abs/2605.30698)。

RELAY 的新增点不是“感知—推理分离”本身，而是：

1. 从 benchmark leaderboard 和 ECI 学习 **有向专家—角色边**；
2. 每题动态选择不同 sensor/reasoner 组合；
3. packet 使用固定类型和缺失值，不传长 CoT；
4. 边权是 `sensor packet -> reasoner` 的 source repair/harm，而非单模型准确率；
5. 最终仍接入 QUID/RIFT，只在组合收益下界为正时启用。

### 4. 使用信息

- 源标签、PARD 修复后的源输出、ECI 自动真值；
- 目标图片/题面和新调用产生的 typed packets；
- 不使用目标标签。

### 5. 训练

视觉 packet schema：

```json
{
  "ocr": [{"text":"...", "bbox":[...]}],
  "axes": {"x":"...", "y":"...", "units":"..."},
  "objects": [],
  "relations": [],
  "quantities": [],
  "unknown_fields": []
}
```

文本 GPQA packet schema：

```json
{
  "given_facts": [],
  "applicable_laws": [],
  "quantities": [],
  "uncertain_claims": []
}
```

在 source/ECI 上枚举少量 sensor→reasoner 组合，学习：

\[
w_{s\to r,t}=P(r\text{ correct from }s\text{ packet}\mid task\ atom=t)-P(r\text{ harmed}).
\]

### 6. 推理

1. 题面/图像检测 task atoms；
2. 选 ECI 上该 atom 最可靠的 1–2 个 sensor；
3. sensor 看图但不看最终候选答案，输出 packet；
4. 若 packet 冲突过大则回退单模型；
5. 选该 packet type 上最强 reasoner；
6. option mapper 只负责语义到选项映射。

### 7. 样本例子

MathVista 柱状图题中，MiniCPM 正确抽取柱高与坐标，InternVL 擅长比例推理。MiniCPM 输出结构化数值，InternVL 只接收题面、选项和数值 packet 完成计算；不让 InternVL重新“看图猜数值”。

### 8. 数据集作用

- MathVista：图表/几何感知与数学推理解耦；
- MMMU-Pro：OCR、图表、科学推理角色分工；
- CMMMU：中文 OCR sensor 与学科 reasoner 组合；
- GPQA：知识陈述者与逻辑推导者接力，但事实型题风险更高。

### 9. 失败原因

- packet 丢失全局视觉信息；
- sensor 产生结构化幻觉；
- 组合数爆炸；
- source/ECI 组合边不迁移；
- 某些题感知与推理不可分。

### 10. 快速验证

先在 ECI 图表/OCR/几何 probe 上只测试 3 sensors × 3 reasoners，比较 `best end-to-end` 与 `best fixed pair`，无需立即跑目标。

### 11. 输出与消融

- `typed_packets.jsonl`
- `relay_edge_weights.csv`
- `relay_paths.jsonl`
- 消融：single VLM / fixed sensor-reasoner / dynamic pair / no packet conflict gate / full RELAY。

## B7. WITNESS-CoE：Blind Witness-Cycle Routing

### 1. 方法名称

**WITNESS-CoE（盲证据闭环路由）**。

### 2. 核心直觉

正确答案应当能够携带一个最小、可跨模型解释的区分证据。让提出答案的专家生成不含选项字母的 witness，再让不知道原候选答案的另一专家把 witness 映射回选项；只有证据闭环回到同一语义答案时才增加可信度。

### 3. 与 verifier/debate/self-critique 的区别

- verifier 直接看到候选答案，容易迎合；
- debate 交换长文本，无法知道共识来自证据还是说服；
- proof-carrying protocols 已用于数值声明的机械验证，例如 [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902)；
- WITNESS 的新点是 **隐藏候选身份、跨模型证据可运输性、语义闭环和 benchmark-derived pair calibration**。

### 4. 使用信息与污染说明

- 目标题面、图片、选项和模型新生成的 witness；
- 不读取目标答案；
- witness 模板、pair 和阈值只在 source/ECI 上校准。

### 5. 训练

源域对每个答案专家 \(e\) 和 mapper \(m\) 统计：

\[
P(\text{candidate correct}\mid
\text{cycle closes},\text{witness type},e,m).
\]

Witness 类型包括：

- 数值等式/不等式；
- 逻辑反例；
- 被读取的表格单元格/图像区域；
- 区分两个科学选项的关键事实；
- 单位与量纲约束。

### 6. 推理

1. 从答案簇中选 base 和一个 challenger；
2. 两者分别生成“不含选项字母与完整选项复述”的最小 witness；
3. 随机打乱选项后，blind mapper 根据 witness 选择语义答案；
4. 若 witness 映射回原候选且结构检查通过，形成 closed cycle；
5. 用 source-calibrated cycle likelihood 更新 RIFT proposal LCB。

### 7. 样本例子

GPQA 中候选 B 与 D 分歧。支持 B 的 witness 是一个可检验的守恒关系，但 blind mapper 映射到 D，说明 witness 实际不支持 B；支持 D 的 witness 给出关键方向性事实，两个独立 mapper 都映射到 D，于是 D 获得可运输证据。

### 8. 数据集作用

- GPQA：把流畅解释转换为能否区分最接近选项的 witness；
- MathVista/MMMU-Pro：数值、OCR 与图表 witness 更容易机械检查；
- CMMMU：要求证据对应具体中文文本/图像区域，而非只看答案共识。

### 9. 失败原因

- witness 暗含选项措辞，mapper 可反推出候选；
- 多模型共享同一错误知识；
- 事实型 witness 无外部机械验证；
- 调用成本较高。

### 10. 快速验证

先在 source 错误分歧样本上缓存 100–200 个 witness，比较：普通自评、非盲 verifier、blind cycle 对 candidate correctness 的 precision@coverage。

### 11. 输出与消融

- `witness_packets.jsonl`
- `blind_mapping_results.jsonl`
- `witness_cycle_scores.jsonl`
- 消融：self-verification / visible-answer verifier / blind mapper / shuffled-option blind mapper / executable witness only。

---

# C. 可以立刻实现的离线缓存方法

以下方法不重新调用任何专家，也不使用目标标签。

## C.1 LEAF-CoE 伪代码

```python
def fit_leaf(source_answers, source_correct, target_answers,
             model_meta, source_benchmark_ids, n_boot=500):
    # 1. 源域构造谱系：架构先验 + 错误互信息 + 输出相似度
    dep = pairwise_error_mutual_information(source_correct)
    sim = pairwise_answer_agreement(source_answers)
    lineages = constrained_cluster(dep, sim, model_meta)

    # 2. 每个 lineage 在目标上形成多类 vote distribution
    Z = lineage_vote_tensor(target_answers, lineages)  # [N, G, K]

    # 3. 目标无标签二阶/三阶矩
    M2 = centered_second_moment(Z)
    M3 = centered_third_moment(Z)

    # 4. 张量分解得到 latent label components 与 lineage confusion
    latent_prior, lineage_conf = tensor_latent_class_fit(M2, M3)

    # 5. 仅用源标签确定 component 的 A/B/C/... 方向
    orientation = choose_orientation_by_source_lobo(
        lineage_conf, source_answers, source_correct,
        source_benchmark_ids
    )
    lineage_conf = apply_orientation(lineage_conf, orientation)

    # 6. lineage reliability + source expert-within-lineage offset
    expert_rel = expand_to_experts(
        lineage_conf, lineages,
        source_within_lineage_offsets(source_correct, lineages)
    )

    # 7. 目标样本 bootstrap，得到目标 accuracy 下界
    lcb = bootstrap_target_reliability_lcb(
        target_answers, lineages, orientation, n_boot=n_boot
    )
    base = argmax(lcb)

    return {
        "lineages": lineages,
        "target_reliability": expert_rel,
        "target_lcb": lcb,
        "base_expert": base,
    }
```

### 立即验证指标

- source LOBO base-selection regret；
- held-out source best-single 命中率；
- estimated vs true expert ranking Spearman；
- lineage 数、有效秩、bootstrap interval width；
- 与 source global、DARE、unlabeled EM 的对比。

## C.2 BRES-CoE 伪代码

```python
def fit_bres(source_items, source_answers, source_labels,
             target_items, target_answers, lineages):
    # 1. 选项表面属性；不需要 target labels
    phi_s = extract_option_features(source_items)
    phi_t = extract_option_features(target_items)

    bias_models = {}
    evidence_models = {}

    for expert in experts(source_answers):
        # 专家在错误时偏好什么位置/长度/词面？
        wrong_mask = source_answers[:, expert] != source_labels
        bias_models[expert] = fit_multinomial_propensity(
            phi_s[wrong_mask], source_answers[wrong_mask, expert]
        )

        # 某次投票超越错误偏置空模型后，含多少正确性信息？
        evidence_models[expert] = fit_crossfitted_likelihood_ratio(
            phi_s,
            votes=source_answers[:, expert],
            gold=source_labels,
            bias_model=bias_models[expert]
        )

    scores = []
    for i in range(len(target_items)):
        candidate_score = {}
        for a in available_options(target_items[i]):
            group_scores = []
            for lineage in lineages:
                raw = 0.0
                for expert in lineage:
                    if target_answers[i, expert] == a:
                        raw += evidence_models[expert].score(a, phi_t[i])
                group_scores.append(cap_lineage_evidence(raw))
            candidate_score[a] = sum(group_scores)
        scores.append(candidate_score)

    return scores
```

### 立即验证指标

- source LOBO repair AUROC/AUPRC；
- vote margin vs residual margin；
- 少数派被正确提升的比例；
- 错误多数派被抑制的比例；
- 对 position、length、lexical-overlap 单独消融。

## C.3 QUID-CoE 简化伪代码

```python
def quid_mask(target_patterns, leaf_intervals, bres_scores,
              source_drift_bounds, label_prior_bounds):
    constraints = build_admissible_world_polytope(
        observed_patterns=target_patterns,
        lineage_moment_intervals=leaf_intervals,
        option_bias_scores=bres_scores,
        drift_bounds=source_drift_bounds,
        label_prior_bounds=label_prior_bounds,
    )

    result = {}
    for pattern_id, pattern in unique_patterns(target_patterns):
        bounds = {}
        for answer in pattern.available_answers:
            bounds[answer] = solve_min_max_posterior_lp(
                constraints, pattern, answer
            )
        identified = one_candidate_separates_all_bounds(bounds)
        result[pattern_id] = {"bounds": bounds,
                              "identified": identified}
    return result
```

优先使用 pattern-level LP，避免逐样本大规模优化。

---

# D. 需要额外调用但不污染目标标签的方法

## D.1 ECI：调用专家回答自动真值能力仪器

### 调用内容

- 所有或代表性专家回答程序化生成的文本/视觉 microbench；
- 每题的真值来自求解器、场景图或渲染参数；
- 生成参数匹配目标无标签题长、选项数、图像布局、OCR 密度等。

### 为什么不污染

这些题不是目标测试样本，也不读取目标答案。目标数据只提供无标签分布统计，类似使用 target covariates 设计测量仪器。

### 首轮规模

- 文本：300–500 题；
- 视觉：OCR/Chart/Geometry 各 100–200 题；
- 先测试 4–6 个代表专家；
- 只有 source LOBO 排名相关性达标才扩展全池。

## D.2 CAST：调用语义等价 choice-set 变体

### 调用内容

对高价值目标分歧题生成 2–3 个不使用 gold 的呈现变体：选项位置置换、标签符号替换、对称同义改写、top-2 preference query。

### 为什么不污染

不读取标准答案；只测专家行为是否满足预先定义的语义等变/选择一致性。所有变换、权重和阈值在 source/ECI 上冻结。

### 注意

不能根据 target 上哪个变体“最终答对”选择版本；只能把 violation vector 输入冻结的可靠性模型。

## D.3 RELAY：调用感知专家产生 typed packet，再调用推理专家

### 调用内容

- sensor 只看图像和原子感知请求，输出 OCR/坐标/关系/数值 packet；
- reasoner 看题面、选项和 packet，不重新访问图像；
- pair 由 source/ECI 预注册。

### 为什么不污染

整个过程只访问目标输入和模型输出，不访问目标正确性；pair selection 来自 source/ECI 真值。

## D.4 WITNESS：调用 blind witness producer 与 mapper

### 调用内容

- base/challenger 生成不含选项字母的最小区分证据；
- mapper 不知道候选身份，在重排选项下映射 witness；
- closure score 使用 source 冻结模型。

### 为什么不污染

闭环只检验不同模型对证据语义是否一致，不把目标正确答案作为 verifier 输入。

---

# E. 推荐的下一轮实验计划

## E0：冻结实验协议

在读取任何新 target score 前固定：

- 专家池与排除列表；
- source splits；
- parser/prompt；
- LEAF/BRES/QUID 超参；
- ECI 生成器版本；
- CAST 变换数；
- RELAY/WITNESS 调用预算；
- 主方法唯一配置；
- bootstrap seeds 和 CI 规则。

## E1：GAOKAO-MM PARD 修复

### 成本

最低；先用已有 raw outputs。

### 预期结果

若低分主要来自格式与截断，valid-answer rate、best single 和 instance oracle 应明显上升。

### 成功标准

- held-out source parser accuracy 显著高于旧 parser；
- format/truncation error 至少下降 50%；
- prompt/parser 调整在 held-out source 上仍有效。

### 停止条件

修复后 best single <15% 且 oracle <25%：停止用 GAOKAO-MM 单独训练视觉 repair graph，转向 ECI + 其他有标签视觉源数据。

## E2：完全缓存的 LEAF + BRES + QUID

### 成本

不调用模型。

### 实验顺序

1. source LOBO 验证 LEAF base selector；
2. source LOBO 验证 BRES repair precision；
3. source LOBO 调 QUID admissible-world width；
4. 冻结后一次性应用目标。

### 预期结果

- LEAF 比 source global/EM 更少选到同源错误多数；
- BRES 改善 GPQA、CMMMU 的少数派 precision；
- QUID 在 MMStar 上大规模拒绝路由，在 BBH/MMLU-Pro 上保留更多 identified coverage。

### 成功标准

- source LOBO base regret 低于 source global 与 DARE；
- BRES 相对 raw vote 的 repair AUPRC 提升；
- QUID risk-coverage 在每个 held-out source 上支配 coverage-null gate；
- 目标最终 accuracy 与 best single 比较，并给 paired bootstrap 95% CI。

### 停止条件

若 LEAF 有效 lineage <3，或 bootstrap reliability interval 极宽，则不使用其点估计，只把它作为相关性审计。

## E3：小规模 ECI

### 成本

中等；300–500 题 × 4–6 专家。

### 预期结果

视觉 ECI 应比 GAOKAO-MM 原始榜单更清楚地区分 OCR、Chart、Geometry、Grounding；文本 ECI 可能区分逻辑/数值能力，但未必覆盖 GPQA 深层事实知识。

### 成功标准

- ECI ranking 与 source held-out real ranking 的 Spearman ≥0.3；
- 相对随机 synthetic probes，target-matched ECI 的 pairwise ranking accuracy 显著更高；
- 至少存在若干 probe family 使专家表现差距 >10 个百分点。

### 停止条件

若 target-matched ECI 不比随机 probe 更能预测 source held-out ranking，停止扩大。

## E4：CAST 分歧样本试验

### 成本

只对 source 200 个分歧题运行，之后最多对目标 10%–20% 高价值样本运行。

### 成功标准

- source 中 violation vector 在控制 expert/task 后对错误有显著增量预测力；
- 相同调用预算下优于普通多次 shuffle majority；
- 目标主方法固定，不能按 target score 选择变换。

### 停止条件

若稳定性与正确性无关或 stable-wrong 占主导，取消目标 CAST。

## E5：RELAY 视觉接力

### 成本

较高；先 ECI，再 source，最后 target。

### 首轮矩阵

- 3 个 sensor × 3 个 reasoner；
- OCR、Chart、Geometry 三类；
- 每类 100–200 个自动真值题。

### 成功标准

- 最佳固定 pair 超过对应 best end-to-end expert；
- 动态 pair 在 source/ECI held-out 上进一步提高；
- 完整目标中，相对严格池 best single 的 paired CI 下界 >0；
- 同时报告每题平均调用数。

### 停止条件

若 typed packet 导致 best reasoner 下降超过 2 个点，或 packet conflict 无法预测失败，停止接力。

## E6：WITNESS 只用于 GPQA/高价值视觉题

### 成本

最高；只处理 base/challenger 强分歧且 QUID 可识别性中等的样本。

### 成功标准

- source/ECI 中 blind cycle 的 precision@20% coverage 显著高于 visible verifier；
- GPQA 相对 RepairChain 33.10% 的净增益 paired CI 下界 >0；
- 若只改善选择性预测，则报告 coverage-risk，不包装成整体 accuracy 突破。

### 停止条件

若 witness closure 在 correct/wrong 候选间差异 <5 个百分点，或 mapper 可通过 option wording 猜出候选，停止。

## E7：目标级成功判据

| 数据集 | 严格比较基准 | 成功标准 |
| --- | ---: | --- |
| MMLU-Pro test | 57.61% strict best single；ECR 60.31% | 不低于 60.31%，并降低调用成本或提高 CI 下界 |
| BBH | 68.05% best single；FATE 77.98% | 不牺牲现有显著增益；ORBIT 主要验证 base/成本而非强行刷新最高点 |
| GPQA | 32.17% strict best single；RepairChain 33.10% | 相对 best single 和 RepairChain 的 paired CI 下界均 >0 |
| MMStar text-only | 24.47% | 若 forced-choice，整体 CI 下界 >0；否则主要报告正确识别不可路由的 risk-coverage |
| CMMMU excl. Qwen3-VL-4B | 38.56% | gain 的 paired CI 下界 >0，不再接受仅点估计 +0.5–0.9 |
| MathVista excl. Qwen3-VL-4B | 60.80%；现有 63.70% | RELAY/ECI 至少不低于 63.70%，并报告成本 |
| MMMU-Pro excl. Qwen3-VL-4B | 30.46% | paired CI 下界 >0；10-option setting 单独报告 |

所有 target 标签只能在方法冻结后读取一次。若需要继续开发，必须使用 source LOBO、ECI 或新的独立 validation，不得回看同一 target score 调参。

---

# F. 论文级创新表述

## F.1 中文 Introduction/Method 表述

现有大模型路由通常把问题表述为从查询到模型标识的预测，而 Bench-CoE 的分布外场景更困难：专家池由源 benchmark 排行榜构建，源域最优关系在目标 benchmark 上可能发生逆转；同时，多专家 instance oracle 还会被选项覆盖、模型谱系相关和选择偏置显著抬高。因此，目标无标签输出中“至少有专家给出正确选项”并不意味着路由器能够识别该专家。

我们提出 ORBIT-CoE，一种面向 benchmark-derived expert pools 的无目标标签 OOD 协作框架。ORBIT 首先在模型谱系商空间中分解目标答案的二阶与三阶一致性矩，并利用源 benchmark 对潜在混淆成分定向，从而估计目标域专家可靠性并选择安全基准。随后，ORBIT 通过偏置残差证据剥离位置、长度和词面吸引力造成的伪投票，并在与目标无标签统计兼容的多个潜在真值世界中识别不可路由样本。为补充目标域缺失的能力监督，我们进一步构造由程序或渲染引擎提供真值、由目标无标签结构驱动、以最大化专家可区分性为目标的可执行能力仪器。对于可路由的高价值样本，ORBIT 不进行普通多数投票或自由多智能体讨论，而是学习类型化的感知—推理接力图，并要求候选答案携带可跨专家盲映射的最小证据。所有基准选择、阈值、接力边和证据闭环精度均仅由源 benchmark 或自动真值仪器校准；目标标签仅用于最终一次性评分。

## F.2 英文贡献表述草案

> We study OOD routing for benchmark-derived expert pools, where source leaderboard rankings may reverse on the target domain and a high per-instance oracle can be dominated by option coverage, lineage-correlated errors, and answer-choice propensities. We introduce ORBIT-CoE, a target-label-free framework that (i) orients lineage-quotiented second- and third-order agreement factors using source benchmarks to estimate target expert reliability, (ii) converts votes into bias-residual evidence and identifies samples whose labels are not distinguishable across admissible latent truth worlds, (iii) constructs target-shaped executable capability instruments with programmatic ground truth and optimal expert-discrimination design, and (iv) composes experts through typed perception–reasoning relay edges and blind transportable-witness cycles. Unlike conventional query-to-model routing or answer-level ensembling, ORBIT explicitly models whether expert advantage is identifiable under the available observations and acquires additional evidence only through source-calibrated, label-free interventions.

## F.3 可主张的贡献点

1. 将 source leaderboard 的 OOD 迁移建模为带测量误差、谱系依赖和潜在方向不确定性的可靠性识别问题；
2. 提出多选项、谱系商空间、源定向的目标无标签可靠性估计，用于安全 base selection；
3. 提出偏置残差证据，剥离选项位置/长度/词面吸引力造成的伪共识；
4. 提出 admissible decoy worlds 下的逐样本不可路由识别，扩展 option-coverage null；
5. 提出 target-shaped executable capability instruments，以程序化真值和最优实验设计替代目标标签 probe；
6. 提出 benchmark-derived typed capability relay graph，把视觉感知、科学推理和选项映射按专家优势重新组合；
7. 提出 blind witness cycle，用跨专家证据可运输性替代可迎合的 self-verification。

---

# G. 原创性边界与近邻工作核查

截至 2026-07-17，以下单独部件已有相关研究，论文中不能声称它们本身首次出现。

| 方向 | 已有工作 | ORBIT 必须强调的差异 |
| --- | --- | --- |
| 无标签专家准确率估计 | [Jaffe et al., 2015](https://proceedings.mlr.press/v38/jaffe15.html) 使用二/三阶矩；[Jaffe et al., 2016](https://proceedings.mlr.press/v51/jaffe16.html) 处理依赖分类器 | 多选 Bench-CoE、模型谱系商空间、source orientation、OOD base 与 repair 接口 |
| 生成数据路由 | [Routing with Generated Data, ACL 2026](https://aclanthology.org/2026.acl-long.1498/) | 程序/渲染真值、目标结构匹配、最优实验设计、用于定向可靠性与角色图而非普通 query router |
| 选项顺序稳定性 | [Pezeshkpour & Hruschka, 2023](https://arxiv.org/abs/2308.11483) | per-instance choice-axiom violation 作为 OOD repair evidence，不是单模型校准或多版本投票 |
| 偏好公理违反 | [Zhao et al., 2024](https://arxiv.org/abs/2410.08851) | 从排序诊断推进到 source-calibrated target routing 与 unroutability gate |
| 感知—推理分离 | [VISTA](https://arxiv.org/abs/2512.22183) | 动态选择不同专家对，边权来自 leaderboard/ECI repair，typed packet 与安全切换 |
| 视觉证据对齐多智能体 | [EAGLE](https://arxiv.org/abs/2605.30698) | 不只对齐区域；还学习阶段角色、跨专家 packet transport 和盲 witness closure |
| Proof-carrying output | [Proof-Carrying Numbers](https://arxiv.org/abs/2509.06902) | 从数字机械验证扩展为隐藏候选身份的跨专家语义证据闭环 |

较稳妥的原创性表述是：

> To our knowledge, ORBIT-CoE is the first framework to jointly use lineage-oriented unlabeled moment factorization, option-bias residualization, programmatically grounded target-shaped capability instruments, and transportable evidence cycles for safe OOD routing over benchmark-derived LLM/VLM expert pools.

不要声称“第一个使用三阶矩”“第一个生成路由数据”“第一个做选项置换”或“第一个分离视觉感知与推理”。

---

# 最终推荐

下一轮最优执行顺序是：

```text
PARD 修复 GAOKAO-MM 测量
  -> LEAF 选择 target-unlabeled base
  -> BRES 剥离选项偏置
  -> QUID 标记不可路由样本
  -> 小规模 ECI 测量目标形态能力
  -> CAST 只处理少数高价值分歧题
  -> 视觉任务测试 RELAY
  -> GPQA/高价值视觉题测试 WITNESS
```

如果只能先实现两个方法，应选择 **LEAF-CoE + BRES-CoE**：二者完全使用缓存，分别解决 RIFT 的 base 不可靠和弱专家共识问题。如果允许新增一次中等规模实验，应优先实现 **ECI-CoE**，因为它可能同时修复 GPQA 缺少目标型科学能力信号、GAOKAO-MM 源监督失真以及 VLM 能力原子不可测三个问题。
