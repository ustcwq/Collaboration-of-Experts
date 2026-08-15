# ATLAS-CoE：基于源—目标关系几何与多目标联合证据的下一代 Bench-CoE

日期：2026-07-17  
依据：`mmlu_pro_and_mmmu_pro_source_results_no_language_bridge_summary.md`  
前置工作：Improve1–6、RIFT-CoE、MIRAGE-CoE、ORBIT-CoE  
状态：原创方法与实验设计；不把尚未运行的结果写成实验结论

## 核心结论

这份新结果最重要的信息不是“又有几个数据集超过 best single”，而是揭示了一个新的迁移规律：

> **源 benchmark 的价值主要取决于源—目标专家关系结构是否可对齐，而不是源样本数量。**

证据非常鲜明：

- MMLU-Pro validation 只有 70 个样本、每类仅 5 个，却能在 BBH 上产生 +9.80% 的稳定增益；
- MMMU-Pro validation 有 577 个样本，但迁移到 CMMMU、MathVista、MMMU-Pro test-id 的增益只有 +1.11%、+1.90%、+1.47%，且 paired CI 全部跨 0；
- 同一源域的最佳方法随目标剧烈变化：BBH 是 LEAF，GPQA 是 BRES，MMStar 是 disagreement topology；视觉侧 CMMMU 是 DARE，MathVista 是 FATE，MMMU-Pro test-id 却是 dynamic subject discovery。

因此，下一轮不应再假设存在一个统一 router，也不应继续直接迁移 source correction graph。本文提出：

## ATLAS-CoE

**Anchored Transfer-Landscape Alignment and Selection for Collaborative Experts**

ATLAS 将路由分成五个问题：

1. 源 repair edge 是否有足够证据，还是由 1–2 个源样本偶然决定？
2. 源失败状态与目标无标签行为状态是否具有可对齐的关系几何？
3. 多个相近无标签目标能否联合估计共享能力，同时保留目标特异性？
4. 当前“路由信号”是否也会在打乱输入—输出关系的负对照中出现？
5. 多个方法中选出的 winner 是否经过选择后统计校正？

对应六个新模块：

- **LEDGER**：稀疏源域 repair edge 后验账本；
- **A-FGW**：专家锚定的失败状态关系几何对齐；
- **CONSTELLATION**：多无标签目标联合可靠性层析；
- **TRIDENT**：源证据、迁移几何、目标稳定性三钥匙切换证书；
- **NULLSHIELD**：基于负对照世界的伪路由信号过滤；
- **TAPESTRY**：由关系几何决定的 source-only 方法组合；
- 另加 **PROSPECT**：选择后推断与前瞻式新目标评测协议。

它们不重复 RIFT 的 interval arbitration、ORBIT 的 option-bias residualization、MIRAGE 的 metamorphic intervention，而是解决这份新结果暴露出的 **source transfer geometry** 问题。

---

# 1. 对当前结果的重新审计

## 1.1 增益对应多少道题

仅看百分比容易高估小目标上的证据强度。按样本数粗略换算净修复题数：

| Source → Target | Gain | Samples | 约净增加正确题数 | 当前证据判断 |
| --- | ---: | ---: | ---: | --- |
| MMLU-Pro val → MMLU-Pro test | +2.69% | 12,032 | +324 | 有潜力，但缺 paired CI |
| MMLU-Pro val → GAOKAO objective | +1.73% | 2,944 | +51 | 有潜力，但缺 paired CI |
| MMLU-Pro val → BBH | +9.80% | 6,511 | +638 | 强且 CI 全正 |
| MMLU-Pro val → GPQA | +0.65% | 4,768 | +31 | CI 跨 0 |
| MMLU-Pro val → MMStar text-only | +0.27% | 1,500 | +4 | 不能作为稳定提升 |
| MMMU-Pro val → CMMMU | +1.11% | 900 | +10 | CI 跨 0 |
| MMMU-Pro val → MathVista | +1.90% | 1,000 | +19 | 接近但仍未显著 |
| MMMU-Pro val → MMMU-Pro test-id | +1.47% | 1,153 | +17 | CI 跨 0 |

BBH 与其他目标不是同一级别的证据。视觉侧目前最合理的表述是“正向探索性结果”，不能写成已经稳定超过 best single。

## 1.2 MMLU-Pro validation 的 70 样本风险

14 类、每类 5 个样本意味着：

- 某个 expert pair 的修复关系可能只由一题决定；
- category-level accuracy 的一个样本对应 20 个百分点；
- correction graph 的边权点估计极不稳定；
- 复杂方法容易在源域学习到偶然 repair edge。

LEAF 在 BBH 的成功可能说明目标无标签大样本提供了强结构信息，但 source orientation 和 repair edge 仍必须报告有限样本后验，不能只报告点估计。

## 1.3 “每个目标的最佳方法”不是可部署主方法

表格从 Improve1–6、ORBIT-lite 及其多个具体方法中选出目标得分最高者。即使每个方法训练时都没有使用目标标签，**根据目标最终分数挑 winner** 仍属于方法层面的目标适应。

普通 paired bootstrap CI 只回答：

> 如果事先固定这个方法，它与 best single 的差值区间是什么？

它没有回答：

> 搜索许多方法再挑最高者后，这个 winner 的真实增益区间是什么？

近期 LLM 评测研究已专门讨论 adaptive benchmarking 的 winner's curse，并提出冻结 shortlist、分离选择与评估、使用 simultaneous bootstrap 的程序级推断。参见 [Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking](https://arxiv.org/abs/2605.05973)。

因此：

- BBH +9.80% 很大，选择校正后仍可能成立，但需要正式计算；
- MathVista +1.90%、CMMMU +1.11%、MMMU-Pro +1.47% 很可能受 winner selection 影响；
- MMStar +0.27% 只有约 4 题，不能作为主张；
- MMLU-Pro test、GAOKAO 必须补 paired CI。

## 1.4 已有目标已成为开发型 OOD benchmark

Improve1–6、RIFT、ORBIT 已多次查看这些目标的最终结果，并据此继续设计方法。即使代码没有直接用目标标签训练，研究过程已经获得了“哪个方向在 BBH/GPQA/MMStar/CMMMU 上有效”的反馈。

所以最严格的论文口径应是：

- 现有目标：development OOD / retrospective evaluation；
- 最终 ATLAS：在新的、从未查看结果的 locked target 上做一次前瞻式验证；
- 若无法增加新 benchmark，至少预留从未被方法选择使用的 secret split，并由独立脚本一次性解封。

这是避免研究者层面 target leakage 的必要条件。

---

# 2. 创新一：LEDGER——稀疏源域 Repair Edge 后验账本

## 2.1 核心思想

不再把 source correction graph 的边权记成一个频率，而把每条边拆成：

```text
证据数
repair 数
harm 数
category 覆盖
benchmark 覆盖
leave-one-item 影响
后验净收益区间
```

对基准专家 \(b\) 和候选专家 \(e\)，源域净修复为：

\[
\Delta_{s,e,b}=P_s(C_e=1,C_b=0)-P_s(C_e=0,C_b=1).
\]

使用层次 Beta-Binomial / logistic-normal 模型：

\[
\operatorname{logit}P(\text{repair}_{i,e,b})
=\mu_{e,b}+u_{category(i),e,b}+v_{benchmark(i),e,b}.
\]

对 harm 建立对应模型，并保存：

\[
P(\Delta_{e,b}>0\mid\mathcal D_S),
\qquad
LCB_S(e,b).
\]

## 2.2 新增的 Leave-One-Item Fragility

定义边的单样本影响：

\[
I_{e,b}=\max_i
\left|\widehat\Delta_{e,b}-widehat\Delta^{(-i)}_{e,b}\right|.
\]

如果移除一题就改变 repair edge 符号，则该边不能迁移。MMLU-Pro validation 每类 5 题时，这个指标尤其关键。

## 2.3 与已有方法的区别

- ECR/FATE 使用 repair 点估计或状态频率；
- RIFT 给出跨域收益下界，但没有显式记录每条源边由多少独立证据支撑；
- LEDGER 将 **source evidence provenance** 变成路由的一等变量。

## 2.4 推理使用

只有同时满足以下条件的边进入候选池：

\[
P(\Delta_{e,b}>0\mid\mathcal D_S)>1-\alpha,
\quad
LCB_S(e,b)>0,
\quad
I_{e,b}<\tau_I.
\]

不满足时，后续 target signal 再强也不能把它解释为“源上已证实的修复关系”。

## 2.5 快速实现

完全使用当前 source correctness cache；不需要模型调用，也不使用目标标签。

输出：

- `source_edge_ledger.parquet`
- `edge_posterior_samples.npz`
- `edge_leave_one_item_influence.csv`
- `category_coverage_matrix.csv`

## 2.6 预期价值

- 防止 70 样本源域产生虚假强边；
- 保留跨类别、跨 benchmark 稳定 repair edge；
- 为后续几何迁移提供带不确定性的源关系，而非脆弱点图。

---

# 3. 创新二：A-FGW——专家锚定的失败状态关系几何对齐

## 3.1 为什么需要关系几何

结果表说明“题面语义相似”不足以预测迁移：

- MMLU-Pro → BBH 非同 benchmark，却强迁移；
- MMMU-Pro validation → MMMU-Pro test-id 同 benchmark，最佳却是简单 dynamic subject discovery；
- MMMU-Pro → MathVista 的 FATE 有一定作用；
- MMLU-Pro → GPQA/MMStar 几乎不稳定。

真正需要比较的是：

> 源域中“哪些专家共同失败、谁修复谁、哪些状态相邻”的关系结构，是否在目标无标签行为中保留。

## 3.2 两层失败状态图

构造源图：

\[
G_S=(E\cup Z_S,\,A_S),
\]

其中：

- \(E\)：专家节点；
- \(Z_S\)：源 failure states；
- expert–state 边：correct、repair、harm、输出形式；
- state–state 边：错误集合重叠、任务原子相似、转移概率。

构造目标图：

\[
G_T=(E\cup Z_T,\,A_T),
\]

其中 \(Z_T\) 只由目标无标签答案簇、输出特征、BRES residual、稳定性等形成，不使用 correctness。

## 3.3 Anchored Fused Gromov-Wasserstein

专家节点身份已知，必须固定；只对齐 source/target failure states。求耦合矩阵 \(\Gamma\)：

\[
\min_{\Gamma}
\sum_{z,z',t,t'}
\left(D_S(z,z')-D_T(t,t')\right)^2
\Gamma_{zt}\Gamma_{z't'}
+\lambda\sum_{z,t}M_{zt}\Gamma_{zt},
\]

约束 expert anchors 的 incidence distortion 最小。

然后将 source repair posterior 运输到 target state：

\[
\widehat\Delta_T(t,e,b)
=\frac{\sum_z\Gamma_{zt}\,\mathbb E[\Delta_S(z,e,b)]}
{\sum_z\Gamma_{zt}}.
\]

## 3.4 Transferability Certificate

对目标状态 \(t\) 计算局部对齐残差：

\[
\delta_{align}(t)
=\mathbb E_{z\sim\Gamma(\cdot|t)}
\left[\operatorname{distortion}(z,t)\right].
\]

- 低残差：允许迁移 source repair；
- 中残差：只允许高证据 LEDGER edge；
- 高残差：判为 source-unmatched，回退 target-unlabeled base 或 abstain。

## 3.5 与 GraphRouter/GW domain adaptation 的区别

Fused Gromov-Wasserstein 已用于非 IID 图域适配和异构数据对齐，例如 [Gadget](https://arxiv.org/abs/2505.12709) 与 [Joint Metric Space Embedding](https://proceedings.mlr.press/v267/beier25a.html)。

A-FGW 的原创性不能写成“首次使用 GW”，而应限定为：

- 专家身份锚定；
- 对齐的是 benchmark failure-state ecology；
- source 一侧边含 correctness/repair posterior，target 一侧仅含无标签行为；
- 输出不是节点分类，而是 repair relation 的可迁移性证书；
- graph 不作为 GNN router 输入。

## 3.6 结果假设

应首先验证：

\[
d_{A\text{-}FGW}(\text{MMLU},\text{BBH})
<d_{A\text{-}FGW}(\text{MMLU},\text{GPQA/MMStar}),
\]

并且 MMMU-Pro → MathVista 距离低于 → CMMMU/MMMU test-id 的难迁移状态，或至少能区分局部可迁移状态。

若 A-FGW 距离与现有八个 transfer gains 无相关性，则停止该方向，不应用到目标路由。

---

# 4. 创新三：CONSTELLATION——多无标签目标联合可靠性层析

## 4.1 新观察

视觉侧单个目标样本量只有 900、1000、1153，但三者共享同一 VLM pool 和部分能力原子。单独估计 target reliability 方差较大；简单合并又会抹去 CMMMU、MathVista、MMMU-Pro 的差异。

## 4.2 共享—私有可靠性模型

对目标域 \(d\)、专家 \(e\)、latent task state \(k\)：

\[
\operatorname{logit}\pi_{d,e,k}
=\mu_e+u_e^Tv_k+s_{d,e,k}.
\]

- \(\mu_e\)：跨目标共享可靠性；
- \(u_e^Tv_k\)：专家与能力原子的交互；
- \(s_{d,e,k}\)：目标特异偏移；
- A-FGW 距离决定 \(s_{d,e,k}\) 的 shrinkage：关系接近则多共享，关系远则少共享。

目标 latent truth 仍不可见，通过多专家答案张量、lineage moments 和 source/ECI orientation 联合推断。

## 4.3 为什么不是普通 multi-target domain adaptation

多目标无监督域适配已有共享/私有表示与知识蒸馏路线，例如 [Unsupervised Multi-Target Domain Adaptation](https://arxiv.org/abs/1810.11547)。CONSTELLATION 的不同点是：

- 不适配一个统一预测模型；
- 联合估计的是同一专家池在多个 benchmark 上的 latent reliability；
- 每个目标仍保留独立的 confusion/repair residual；
- 共享强度由 failure-geometry alignment 决定；
- 输出服务于 base、repair edge 和拒绝路由。

## 4.4 具体使用

视觉组联合：

```text
CMMMU 900
MathVista 1000
MMMU-Pro test-id 1153
总无标签行为样本 3053
```

语言组可联合 MMLU-Pro test、GAOKAO、BBH、GPQA、MMStar，但 GPQA/MMStar 只有在 A-FGW 对齐通过时才能从其他目标借力，避免负迁移。

## 4.5 重要统计说明

联合建模可以降低 **目标可靠性估计** 的方差，但不能直接缩窄最终 accuracy gain 的 paired bootstrap CI。最终 CI 仍由目标题目上的实际 repair/harm 决定；要获得显著性，方法必须真正增加净修复，而不是借模型把区间“算窄”。

## 4.6 输出与消融

- `multi_target_shared_reliability.csv`
- `target_private_residuals.npz`
- `cross_target_borrowing_weights.csv`
- 消融：separate targets / full pooling / fixed hierarchical / A-FGW adaptive sharing。

成功标准：source meta-target 中 adaptive sharing 的 base regret 和 repair AUPRC 同时优于 separate/full pooling。

---

# 5. 创新四：TRIDENT——三钥匙 Repair Certificate

## 5.1 核心思想

一个切换必须同时回答三个不同问题：

1. **Source evidence**：候选真的在源上修复过基准吗？
2. **Transport alignment**：当前目标状态真的与源 repair state 对齐吗？
3. **Target replicability**：这一目标无标签信号在不同 target 子样本、不同初始化下稳定吗？

## 5.2 三项证书

定义：

\[
Cert(t,e,b)=
LCB_{LEDGER}(e,b)
-L\delta_{align}(t)
-\xi_{split}(t,e,b).
\]

其中：

- \(LCB_{LEDGER}\)：源 repair 后验下界；
- \(\delta_{align}\)：A-FGW 局部关系失真；
- \(\xi_{split}\)：将目标无标签样本随机分成若干块后，候选排序和状态映射的不稳定惩罚；
- \(L\) 只用 source meta-transfer 估计。

切换规则：

\[
\pi(t)=
\begin{cases}
e^*, & \max_e Cert(t,e,b)>\tau,\\
b, & \text{otherwise}.
\end{cases}
\]

## 5.3 与 RIFT 的区别

RIFT 直接估计候选相对 base 的保守收益区间；TRIDENT 将不确定性显式分解为：

- 源证据不足；
- 关系不可迁移；
- 目标统计不可复现。

这三种失败对应不同处理：增加源数据、拒绝跨域、或扩大目标无标签估计，避免所有不确定性都被一个 margin 混合。

## 5.4 预期解释当前结果

- BBH：LEDGER 部分边有证据、alignment 低、split stability 高，因此保留大部分 LEAF/FATE 增益；
- GPQA：alignment 或 target stability 不足，减少弱切换；
- MMStar：多数状态关系不可识别，几乎全部回退；
- MathVista：只在图表/几何对齐状态中允许 FATE；
- MMMU-Pro test-id：若 subject structure 对齐但 failure graph 不对齐，则动态 subject proposal 通过、FATE 被拒绝。

---

# 6. 创新五：NULLSHIELD——负对照世界中的伪路由信号过滤

## 6.1 为什么需要负对照

一个 router 在真实目标输出上给出高置信切换，并不说明它找到了可路由能力。若打乱题目与输出的对应关系后，它仍然高置信切换，说明置信度来自：

- 选项频率；
- 专家全局偏好；
- 模型家族数量；
- 答案簇大小；
- 目标 dataset-level prior；

而不是当前题的可修复证据。

## 6.2 四类无标签负对照

### NC1：Question–Output Decoupling

随机把第 \(i\) 题的专家输出矩阵配给第 \(j\) 题，保持整体答案频率与专家相关性，但破坏逐题语义对应。

### NC2：Image–Question Decoupling

视觉任务中打乱图片与问题的配对；只用于计算 router score，不需要知道 gold，也不把结果作为真实预测。

### NC3：Lineage-Preserving Expert Shuffle

只在同一 lineage 内置换专家身份，检测方法是否依赖没有独立证据的家族复制。

### NC4：Option-Semantic Knockoff

保持每个专家的 A/B/C/…位置分布和一致性图，但随机重映射选项语义，检测路由是否只是位置统计。

## 6.3 Null-Calibrated Routing p-value

对真实样本路由证书 \(S_i\)，通过 \(B\) 个负对照世界得到 \(S_i^{(1:B)}\)：

\[
p_i=\frac{1+\sum_b\mathbf1[S_i^{(b)}\ge S_i]}
{B+1}.
\]

只有：

\[
Cert_i>\tau\quad\text{且}\quad p_i<\alpha
\]

才允许切换。

## 6.4 为什么不污染目标标签

整个 null calibration 只使用目标输入和无标签专家输出；没有读取标准答案或 correctness。阈值控制的是“输入—输出已经被破坏时仍产生高置信切换”的伪发现率。

## 6.5 与 CAST/MIRAGE 的区别

- CAST 重新调用模型，测试单个专家在语义等价 choice sets 下的行为稳定性；
- MIRAGE 使用 metamorphic interventions 定位失效；
- NULLSHIELD 不重新调用模型，而是对 **整个路由统计量** 构造失去逐题语义链接的负对照世界。

## 6.6 快速验证

完全使用当前缓存即可运行。若某方法在 NC1/NC4 上仍保持接近真实目标的 switch rate 和 score，说明该方法不应继续使用。

输出：

- `negative_control_scores.parquet`
- `routing_null_pvalues.jsonl`
- `null_switch_rate_by_method.csv`
- `real_vs_null_score_curves.pdf`

---

# 7. 创新六：TAPESTRY——关系几何驱动的方法组合

## 7.1 新问题

当前最佳方法随 target 变化：

| Target | Winner family |
| --- | --- |
| MMLU-Pro test | ECR |
| GAOKAO | RepairChain |
| BBH | LEAF |
| GPQA | BRES |
| MMStar | disagreement topology |
| CMMMU | DARE |
| MathVista | FATE |
| MMMU-Pro test-id | dynamic subject discovery |

这证明不存在简单固定 winner。

## 7.2 Source-Only Meta-Environments

把源 benchmark、类别、难度层和人为协议扰动构造成 meta-environments。对每个环境：

1. 一部分环境作为 labeled source；
2. 另一环境隐藏标签，模拟 target-unlabeled；
3. 运行所有方法；
4. 记录每种方法的真实 held-out regret；
5. 提取不含标签的 transfer descriptors。

Descriptors 包括：

- A-FGW distance 与局部 distortion；
- target moment effective rank；
- coverage null；
- lineage 数量；
- option count；
- modality/task atoms；
- target split stability；
- source LEDGER edge coverage。

## 7.3 方法权重

训练 source-only regret predictor：

\[
\widehat R_m(x_{transfer})
\approx \text{regret of method }m.
\]

目标方法权重：

\[
w_m\propto
\exp(-\eta\widehat R_m)
\mathbf1[\text{TRIDENT}_m\text{ passes}].
\]

不是直接混合最终答案，而是加权每个方法的 proposal evidence；最终仍需 TRIDENT + NULLSHIELD。

## 7.4 与 LOBO meta-router 的区别

- LOBO 通常从少量源 benchmark 中选一个全局方法；
- TAPESTRY 用关系几何预测每个方法的 regret；
- 输出是经过证书过滤的方法权重，而非目标 winner；
- 允许同一 target 的不同 failure states 使用不同方法，但所有参数仍 source-only。

## 7.5 风险

如果 meta-environments 数量太少，regret predictor 会过拟合。必须通过 benchmark-level outer leave-one-out 验证；不能用 category 随机切分冒充独立 benchmark transfer。

---

# 8. PROSPECT：选择后推断与前瞻式新目标协议

## 8.1 Retrospective Selection-Aware CI

对已经运行的所有方法，构造每题相对 best single 的 paired difference vector：

\[
d_i^{(m)}=C_i^{(m)}-C_i^{(best)}.
\]

在每次 bootstrap 中同时重采样所有方法，并记录：

\[
T^*=\max_m
\frac{\bar d^{*(m)}-\bar d^{(m)}}{se^{*(m)}}.
\]

用 max-statistic 构造 simultaneous CI，覆盖“从候选方法中选最大者”的过程。不能只对 winner 单独 bootstrap。

## 8.2 主表分层

建议论文结果分三层：

1. **Pre-registered main method**：完全 source-only 冻结；
2. **Selection-aware method-family comparison**：simultaneous CI；
3. **Exploratory oracle-over-methods**：明确标为 post-hoc，不作为部署结果。

## 8.3 Fresh Locked Target

最终 ATLAS 必须在一个未用于 Improve1–6/RIFT/ORBIT 反馈的新 target 上一次性评测。建议流程：

```text
冻结代码与配置哈希
-> 生成预测文件
-> 保存不可修改时间戳
-> 解封 target labels
-> 只运行预注册 scorer
-> 输出 accuracy、paired CI、repair/harm、cost
```

如果没有新 benchmark，则从后续新增数据中先锁定 secret split；已经多次查看的现有目标不能重新称为完全 untouched test。

---

# 9. 三个可立即运行的缓存算法

## 9.1 LEDGER 伪代码

```python
def build_edge_ledger(source_correct, categories, benchmarks,
                      experts, posterior_draws=5000):
    ledger = []
    for base in experts:
        for cand in experts:
            if cand == base:
                continue

            repair = (source_correct[:, cand] == 1) & \
                     (source_correct[:, base] == 0)
            harm = (source_correct[:, cand] == 0) & \
                   (source_correct[:, base] == 1)

            model = fit_hierarchical_repair_harm_model(
                repair=repair,
                harm=harm,
                categories=categories,
                benchmarks=benchmarks,
            )
            draws = model.sample_net_gain(posterior_draws)

            influence = max_leave_one_item_change(
                repair, harm, categories, benchmarks
            )

            ledger.append({
                "base": base,
                "candidate": cand,
                "mean_gain": draws.mean(),
                "lcb_gain": quantile(draws, 0.05),
                "prob_positive": (draws > 0).mean(),
                "item_influence": influence,
                "category_coverage": count_supported_categories(repair, harm),
            })
    return ledger
```

## 9.2 A-FGW 伪代码

```python
def anchored_failure_alignment(source_states, target_states,
                               expert_ids, edge_ledger):
    # expert identities are fixed anchors
    Ds = state_relational_distance(source_states, expert_ids,
                                   labeled=True)
    Dt = state_relational_distance(target_states, expert_ids,
                                   labeled=False)

    cross_cost = anchored_incidence_cost(
        source_states, target_states, expert_ids
    )

    gamma = solve_fused_gromov_wasserstein(
        source_distance=Ds,
        target_distance=Dt,
        cross_feature_cost=cross_cost,
        fixed_expert_anchor=True,
    )

    transported_edges = transport_edge_posteriors(
        gamma, edge_ledger, source_states, target_states
    )
    local_distortion = compute_local_fgw_distortion(
        gamma, Ds, Dt
    )
    return gamma, transported_edges, local_distortion
```

## 9.3 NULLSHIELD 伪代码

```python
def nullshield(router_score_fn, target_questions, target_outputs,
               lineages, B=500):
    real_score = router_score_fn(target_questions, target_outputs)
    null_scores = []

    for b in range(B):
        mode = b % 3
        if mode == 0:
            q_null = permute_rows(target_questions)
            o_null = target_outputs
        elif mode == 1:
            q_null = target_questions
            o_null = shuffle_experts_within_lineage(target_outputs,
                                                    lineages)
        else:
            q_null = target_questions
            o_null = remap_option_semantics_preserve_positions(
                target_outputs
            )
        null_scores.append(router_score_fn(q_null, o_null))

    null_scores = stack(null_scores)
    pvalue = (1 + (null_scores >= real_score).sum(axis=0)) / (B + 1)
    return real_score, pvalue
```

三个算法都不需要重新调用专家，也不需要目标标签。

---

# 10. 下一轮推荐实验顺序

## Phase 0：先纠正报告口径

1. 为 MMLU-Pro test、GAOKAO 补 paired bootstrap CI；
2. 对所有目标做 selection-aware simultaneous CI；
3. 报告每个方法的 repair、harm、net repaired items；
4. 把 per-target best 标为 exploratory；
5. 固定一个 source-only main method。

### 停止条件

若视觉侧 simultaneous CI 全部跨 0，则不再声称已经稳定超过 best single，而把它们作为 ATLAS 的动机。

## Phase 1：LEDGER

### 成功标准

- source meta-transfer 中，过滤高 influence edge 后 negative transfer 降低；
- BBH 至少保留现有增益的 80%；
- GPQA/MMStar 的错误切换率下降。

### 停止条件

若大多数 MMLU source edge 都因 70 样本无法获得正 LCB，则不强行降低阈值；改用更大 source portfolio 或只保留 target-unlabeled methods。

## Phase 2：A-FGW Transferability Audit

### 首先只做审计

用现有八个 source→target transfer 计算距离，检验距离与 observed gain、repair precision、negative transfer 的相关性。

### 成功标准

- A-FGW distance 与 transfer gain 的 Spearman 显著为负；
- BBH 被识别为高可迁移；
- GPQA/MMStar 被识别为低可迁移；
- 优于题面 embedding distance、benchmark transfer graph 和普通 graph Frobenius distance。

### 停止条件

若关系距离不能区分 BBH 与 GPQA/MMStar，停止把它用于 routing，只保留分析图。

## Phase 3：TRIDENT + NULLSHIELD

### 成功标准

- source outer-LOBO 每个 benchmark 上净收益非负；
- 负对照 switch rate 控制在预注册水平；
- 对真实 target 的 switch coverage 不全部坍缩；
- 在 BBH 保留大部分强增益，同时减少 GPQA/MMStar/CMMMU 的无证据切换。

## Phase 4：视觉 CONSTELLATION

先联合 CMMMU、MathVista、MMMU-Pro 的无标签输出，但最终 accuracy 仍分别报告。

### 成功标准

- source meta-target 上 target base regret 低于 separate/full pooling；
- 目标 private residual 能识别 MathVista 与其他两者的差异；
- 相对 current best，paired gain CI 下界在新 locked target 上 >0。

## Phase 5：TAPESTRY

仅当拥有足够独立 source meta-environments 后实现。否则宁可固定 TRIDENT，也不要训练一个过拟合的方法选择器。

---

# 11. 针对各数据集的具体策略

## MMLU-Pro test

- 现有 ECR +2.69%，先补 CI；
- LEDGER 检查 70 样本 repair edge 是否由少数题驱动；
- 同 benchmark 不必复杂 A-FGW，优先用 posterior shrinkage。

## GAOKAO objective

- RepairChain +1.73%，约 51 题；
- 检查 category shift 与 source edge coverage；
- 若 CI 为正，可作为第二个语言稳定结果。

## BBH

- 作为关系几何可迁移的正对照；
- 目标不是盲目刷新 77.85%，而是证明 A-FGW/TRIDENT 能在不看标签时识别它是适合 LEAF/FATE 的目标；
- 进行 selection-aware CI 后再作为主结果。

## GPQA

- BRES 已经是当前最合理候选，但 +0.65% CI 跨 0；
- 使用 LEDGER + NULLSHIELD 抑制弱共识；
- 若 A-FGW 判断低对齐，不允许从 MMLU 迁移复杂 repair graph；
- 不以继续增加 switch rate 为目标。

## MMStar text-only

- +0.27% 只有约 4 题；
- 主要作为 NULLSHIELD/QUID 的不可路由负对照；
- 不应继续把 point improvement 写成成功。

## CMMMU

- DARE +1.11%、约 10 题；
- CONSTELLATION 可借 MathVista/MMMU-Pro 的共享 OCR/grounding 信息；
- A-FGW 只运输低失真视觉状态；
- 最终需要新 locked evaluation 才能成为强主张。

## MathVista

- 当前视觉最接近成功：+1.90%，CI 下界 -0.20%；
- FATE 成功说明部分 failure ecology 可迁移；
- A-FGW 应定位图表/几何低失真状态，TRIDENT 只在这些状态切换；
- 不能根据当前 testmini 结果继续调阈值。

## MMMU-Pro test-id

- 同 benchmark 下 dynamic subject discovery 胜过 failure ecology，说明类别结构比 correction graph 更稳定；
- TAPESTRY 应学习“低 benchmark shift 但高 failure-graph distortion”时偏向 taxonomy proposal；
- 10-option 或复杂图文状态需要单独建图，不能与四选一任务直接共享输出几何。

---

# 12. 论文级创新表述

## 中文草案

现有 benchmark-derived expert routing 通常默认源 benchmark 上的专家优劣关系能够迁移到目标域，并以题面相似度或目标输出一致性选择专家。然而，我们的跨源实验显示，迁移效果与源样本规模并不一致：仅包含 70 个样本的 MMLU-Pro validation 能在 BBH 上产生近 10 个百分点的稳定增益，而包含 577 个样本的 MMMU-Pro validation 在多个视觉语言目标上仅产生统计不确定的小幅提升。这表明 OOD 路由的关键变量不是源数据规模或学科相似度，而是源—目标专家失败关系是否保持可对齐的几何结构。

为此，我们提出 ATLAS-CoE。ATLAS 首先通过层次 repair-edge ledger 量化每条源修复关系的后验强度、类别覆盖和单样本脆弱性；随后构造专家身份锚定的源/目标失败状态图，并利用 anchored fused Gromov-Wasserstein coupling 运输具有充分证据的 repair posterior，同时以局部几何失真刻画关系可迁移性。对于多个共享专家池的无标签目标，ATLAS 通过共享—私有可靠性层析选择性借用统计强度，而不强制共享目标特异错误。最终切换必须同时通过源证据、关系对齐和目标分割稳定性三项证书，并在打乱输入—输出语义对应的负对照世界中控制伪路由信号。所有参数与方法权重只通过源域外层元验证确定；目标标签仅在前瞻式锁定评测中一次性解封。

## 英文草案

> Our cross-source experiments reveal that the transferability of a benchmark-derived expert pool is governed less by source sample size than by the relational compatibility of expert failures: a 70-example MMLU-Pro source yields a robust 9.8-point gain on BBH, whereas a 577-example MMMU-Pro source produces only small, statistically uncertain gains across multimodal targets. We introduce ATLAS-CoE, a target-label-free framework that treats source-to-target routing as anchored transfer of expert-failure geometry. ATLAS (i) maintains a hierarchical posterior ledger over source repair edges and their single-example fragility, (ii) aligns source correctness states with target unlabeled behavior states through expert-anchored fused Gromov-Wasserstein coupling, (iii) jointly estimates shared and target-private expert reliability across multiple unlabeled targets, and (iv) permits a switch only when source evidence, relational alignment, and target split stability jointly certify positive repair. A negative-control shield further rejects routing signals that persist after destroying query–output semantic correspondence. We complement the method with selection-aware simultaneous inference and a prospective locked-target protocol, avoiding winner-based reporting over a large routing-method portfolio.

---

# 13. 原创性边界

| 组成 | 已有近邻 | ATLAS 的新边界 |
| --- | --- | --- |
| Gromov-Wasserstein 域对齐 | [Gadget](https://arxiv.org/abs/2505.12709)、[Joint Metric Space Embedding](https://proceedings.mlr.press/v267/beier25a.html) | expert identity anchors、correctness-to-behavior failure-state alignment、repair posterior transport |
| 多目标无监督适配 | [Information-Theoretic MTDA](https://arxiv.org/abs/1810.11547) | 不训练共享分类器；联合估计 benchmark-specific expert reliability，并由关系几何控制共享 |
| 多标注者专业度 | [Modeling Annotator Expertise](https://proceedings.mlr.press/v9/yan10a.html) | source leaderboard repair/harm、单样本脆弱性、OOD transfer certificate |
| Winner's curse 校正 | [SIREN](https://arxiv.org/abs/2605.05973) | 应用于 target-label-free routing 方法组合、paired gain 和 prospective OOD protocol |
| 负对照与 shuffled controls | routing artifact 研究已使用 shuffled-label controls，例如 [Unsolvability Ceiling](https://arxiv.org/abs/2605.07395) | 逐样本 routing certificate 的 target-unlabeled null p-value 与语义解耦对照 |

不能声称“首次使用 FGW”“首次做多目标适配”“首次使用层次标注者模型”或“首次修正 winner's curse”。较合理的主张是：

> To our knowledge, ATLAS-CoE is the first benchmark-derived LLM/VLM routing framework that transports source repair posteriors through expert-anchored failure-state geometry, combines them with multi-target reliability tomography, and calibrates instance-level switching against target-unlabeled semantic-decoupling controls.

---

# 最终建议

当前最优先的不是继续调用模型，而是运行四个完全缓存的检查：

1. **selection-aware simultaneous CI**：确认现有 winner 中哪些真正可靠；
2. **LEDGER**：检查 70 样本源 repair edge 的单样本脆弱性；
3. **A-FGW audit**：验证关系几何能否解释 BBH 强迁移与 GPQA/MMStar 弱迁移；
4. **NULLSHIELD**：检查 LEAF/BRES/FATE 的高置信切换是否在语义打乱后仍存在。

如果 A-FGW 能在不看目标标签的情况下把 BBH 判为高可迁移、把 GPQA/MMStar 判为低可迁移，并且 TRIDENT 在 source outer-LOBO 上保持非负净收益，那么这将比再增加一个 0.5–1.0 点的 heuristic 更接近真正的论文级突破：它解释了 **何时 source leaderboard 可以迁移、为什么可以迁移，以及何时系统必须拒绝迁移**。
