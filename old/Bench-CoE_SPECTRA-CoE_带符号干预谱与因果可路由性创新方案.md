# SPECTRA-CoE：基于带符号干预谱、动态伪基准与因果可路由性的下一代 Bench-CoE

副标题：从“观察专家共识”升级为“主动检验共识是否由题目证据支撑”

状态：原创方法设计与预注册实验方案；所有预测均为待验证假设，不把未运行结果写成实验事实。

---

## 摘要

当前 Bench-CoE 的核心矛盾已经不再是缺少新的投票或 repair heuristic，而是：在目标标签完全不可见时，系统只能观察专家的一次静态回答。静态共识不能区分三种情况：多个独立专家因正确推理而一致、同一 lineage 因共享偏差而一致、所有专家在高难题上稳定地选择同一错误选项。BBH 上 LEAF 的 +9.80% 与 GPQA/MMStar 上的不稳定结果正好展示了这种不可识别性。

本方案提出 **SPECTRA-CoE（Signed Perturbation Evidence for Causal Transfer and Routability Assessment）**。SPECTRA 不把单次回答当成全部证据，而是对每道目标题构造一个小型、无标签的“干预轨道”：

1. 答案应保持不变的 nuisance-preserving 干预；
2. 答案应按已知映射变化的 equivariant 干预；
3. 删除关键证据后应表现出选择性响应的 evidence-ablation 干预；
4. 删除等量无关内容、用于校正一般脆弱性的 sham 干预。

正确且真正使用题目证据的专家，理想上应同时满足“对无关变化稳定、对选项置换等变、对关键证据删除敏感、对等量无关删除不敏感”。这种双向、带符号的响应比普通 self-consistency 提供更强的无标签信息。

SPECTRA 进一步将 LEDGER 从 hard gate 改为由局部干预证据和 A-FGW 失真控制的可衰减先验；用干预轨道上的潜在答案后验产生逐题 target pseudo-base；将可路由性定义为 **目标集—专家池—探针集合三元组** 的属性；并使用序贯探针、负对照 p-value 和预先冻结的拒绝策略控制成本与负迁移。

---

# 1. 从当前实验事实推出的新问题

## 1.1 当前证据

### 语言侧

| Target | Best single | 当前最佳 | Gain | 当前判断 |
| --- | ---: | ---: | ---: | --- |
| MMLU-Pro test | 57.61% | 60.31% | +2.69% | 稳定正收益 |
| GAOKAO objective | 78.67% | 80.40% | +1.73% | 中等正收益 |
| BBH | 68.05% | 77.85% | +9.80% | 最强、selection-aware CI 全正 |
| GPQA diamond | 32.17% | 32.82% | +0.65% | CI 跨 0，ATLAS Tapestry 负迁移 |
| MMStar text-only | 24.47% | 24.73% | +0.27% | 约 4 题，不稳定 |

### 视觉语言侧

| Target | Best single | 当前最佳 | Gain | 当前判断 |
| --- | ---: | ---: | ---: | --- |
| CMMMU val | 38.56% | 39.67% | +1.11% | CI 跨 0 |
| MathVista testmini | 60.80% | 62.70% | +1.90% | 最接近成功，但 CI 跨 0 |
| MMMU-Pro test-id | 31.48% | 32.96% | +1.47% | subject structure 有效，CI 跨 0 |

这些结果支持两个事实：

1. 源样本量不是迁移成败的主变量；
2. 目标无标签答案生态有时是极强证据，有时几乎没有信息。

## 1.2 ATLAS 的关键负结果不是失败，而是诊断

ATLAS strict TRIDENT 几乎全部回退，说明一个源域中成立的 repair edge，即使后验显著，也未必是 BBH 大增益的来源。BBH 的强信号更多来自目标内部的答案结构。

因此：

- LEDGER 应当提供可衰减的先验，不应拥有一票否决权；
- A-FGW 应决定“源先验保留多少”，而不是直接决定“能否切换”；
- pseudo-base 必须逐题由目标证据产生，不能固定为 source-global best；
- 需要比静态 agreement 更强的无标签观测。

## 1.3 更深层的不可识别性

若只观察一次目标输出矩阵

\[
A_{i,e}\in\{1,\ldots,K_i\},
\]

则“多个专家共同答对”与“多个相关专家共同答错”可能产生完全相同的观测。除非引入额外假设、源域锚点或新的可观测视图，否则不存在普适的无标签规则能够判断哪一个世界是真实世界。

这解释了为什么继续叠加 agreement、entropy、repair graph 或 posterior threshold，最终仍会在 GPQA/MMStar 上碰到天花板。

SPECTRA 的核心动作就是主动制造新的可观测视图，而不是只在同一张静态输出矩阵上继续做复杂统计。

---

# 2. 核心创新：带符号干预谱是目标无标签弱预言机

## 2.1 不是普通一致性

只要求模型在多种表达下给出相同答案是不够的。一个模型可以对所有改写稳定地答错。

SPECTRA 使用四类具有不同预期方向的干预：

| 干预类 | 关系 | 正确响应的预期 | 例子 |
| --- | --- | --- | --- |
| \(\mathcal T^{inv}\) | 不变 | 语义答案保持 | 无关句位置、图像 padding、合法同义改写 |
| \(\mathcal T^{eq}\) | 等变 | 答案按已知映射变化 | 选项置换、实体/变量一致重命名 |
| \(\mathcal T^{abl}\) | 关键证据消融 | 原答案支持应下降或响应改变 | OCR 区域遮挡、图例移除、关键条件删除 |
| \(\mathcal T^{sham}\) | 无关对照消融 | 原答案应基本保持 | 等面积背景遮挡、非关键句删除 |

前三类不能简单相加。关键判别是：

> 可靠专家应对 nuisance 稳定，却对真正承载答案的证据具有选择性敏感性。

这形成 **signed response**：对 \(\mathcal T^{inv}\) 和 \(\mathcal T^{eq}\) 的正确映射是一类正证据；关键消融相对于 sham 消融的额外响应是另一类正证据。

## 2.2 专家干预指纹

对目标样本 \(i\)、专家 \(e\)、干预 \(t\)，记录原回答 \(a_{ie}\) 和干预回答 \(a^{(t)}_{ie}\)。若干预具有已知答案映射 \(\phi_t\)，定义：

\[
q^{map}_{iet}=\mathbf 1\left[\phi_t^{-1}(a^{(t)}_{ie})=a_{ie}\right].
\]

对证据消融，定义响应强度：

\[
r^{abl}_{iet}=d\left(p_{ie},p^{(t)}_{ie}\right),
\]

其中 \(p\) 可以是选项概率；若只有离散输出，则使用跨多个消融变体的答案翻转率。

关键证据选择性为差分：

\[
g_{ie}=\mathbb E_{t\in\mathcal T^{abl}}r^{abl}_{iet}
-\mathbb E_{t\in\mathcal T^{sham}}r^{abl}_{iet}.
\]

最终干预指纹为：

\[
h_{ie}=
[q^{inv}_{ie},q^{eq}_{ie},g_{ie},v_{ie},m_{ie}],
\]

其中 \(v_{ie}\) 是不同变体间方差，\(m_{ie}\) 是原答案与最强竞争答案的干预后 margin。

## 2.3 为什么比 LEAF/BRES 多一层信息

LEAF 观察“哪些 lineage 支持同一答案”；BRES 观察“哪些输出模式具有残差证据”。SPECTRA 进一步问：

- 同一个 lineage 在选项置换后是否仍支持相同语义答案？
- 支持某答案的专家是否对关键证据删除比对无关删除更敏感？
- 一个看似强共识是否在轻微、合法的等变变换后崩塌？

因此它不是新的投票权重，而是为每个投票新增一个可检验的“证据来源是否真实连接到题目”的观测。

---

# 3. SPECTRA-CoE 总体框架

## 3.1 输入

### Source-only

- 源样本 \(S=\{(x_s,y_s)\}\)；
- 每个专家在源样本及合法源干预上的输出；
- expert lineage；
- 源 correctness、repair/harm；
- 干预生成器及其源域有效性审计结果。

### Target-unlabeled

- 目标输入 \(x_i\)，不读取目标 gold；
- 专家原始输出；
- 按序贯策略调用的少量干预输出；
- 图像/OCR/图表/几何等无标签结构元信息；
- 目标整体和局部输出生态。

### Scoring-only

- 目标 gold 只在算法、阈值、方法家族和报告规则全部冻结后用于一次性 scoring；
- paired bootstrap、selection-aware CI 只在最终阶段运行。

## 3.2 输出

SPECTRA 对每个样本输出：

1. 动态 pseudo-base 答案及其专家集合；
2. 最终选择的专家或 strict fallback；
3. item routability；
4. source prior、target intervention evidence、null evidence 三部分分解；
5. 使用了多少额外探针调用；
6. 拒绝切换原因。

对每个目标集输出：

- pool-conditioned routability phase；
- 预计可安全切换比例；
- 有效独立 lineage 数；
- 干预关系可恢复度；
- 负对照存活率；
- 目标无标签分割稳定性。

## 3.3 六个模块

1. **ORBITAL-PROBE**：构造带符号干预轨道；
2. **SOFT-LEDGER**：将 source repair posterior 转为局部可衰减先验；
3. **QUASAR**：联合推断潜在答案、专家可靠性和 lineage 相关性；
4. **PSEUDO-BASE**：逐题产生目标动态基准，而非使用 source-global base；
5. **ROUTE-PHASE**：判断目标—专家池—探针三元组的可路由性；
6. **EVIDENCE-BUDGET**：序贯调用、负对照校准和拒绝切换。

---

# 4. ORBITAL-PROBE：答案关系明确的目标干预轨道

## 4.1 优先使用“关系可证明”的干预

干预生成器按可靠性分层：

### Tier 0：程序化精确关系

- 多项选择题选项顺序置换；
- 选项字母重新编码；
- BBH 中可程序化的实体/变量双射重命名；
- 表格行列的等价重排；
- 图像 resize、padding、无损格式变化。

Tier 0 的答案映射可以由程序精确给出，应优先覆盖所有目标。

### Tier 1：源标签验证的近似保持关系

- 题干同义改写；
- 无关上下文位置改变；
- 中文数字格式、标点与术语规范化；
- 图表配色变化；
- 非语义性的亮度、对比度变化。

生成器必须先在源集上验证：若变换后 gold 关系破坏率超过预注册阈值，则整类干预被禁用，而不是在目标上看结果后挑选。

### Tier 2：诊断性证据消融

- OCR bounding box 遮挡；
- 图例或坐标轴局部遮挡；
- 几何标记/辅助线遮挡；
- 语言题关键条件删除；
- 与关键区域等面积、同纹理的 sham mask。

Tier 2 不提供目标答案，只检验模型响应是否与证据位置存在选择性联系。

## 4.2 干预有效性账本

每个变换族 \(t\) 在源集上维护 Beta 后验：

\[
\rho_t\sim\mathrm{Beta}
(\alpha_0+n^{valid}_t,\beta_0+n^{invalid}_t).
\]

目标证据权重使用 \(\mathbb E[\rho_t]\) 与后验下界，而不是给所有生成式改写同等权重。

若某干预无法在源上验证，又不存在程序化保证，只能作为探索性诊断，不能进入主路由分数。

## 4.3 防止“稳定错误”

单独使用 \(\mathcal T^{inv}\) 会奖励稳定错误。SPECTRA 要求至少满足以下二者之一：

- exact equivariance：选项置换后能映射回同一语义答案；
- evidence specificity：关键证据消融的响应显著大于 sham 消融。

因此“所有变体永远输出 C”会在选项置换中暴露位置偏差；“无论图像是否被遮挡都输出同一答案”会在关键消融中暴露视觉忽略。

---

# 5. SOFT-LEDGER：从硬门控改为局部温度先验

## 5.1 源 repair posterior

对能力状态 \(k\)、incumbent \(e\)、challenger \(f\)，源域中维护：

\[
\pi^S_{e\rightarrow f,k}
=P(f\ correct,e\ wrong\mid k,S)
-\lambda P(f\ wrong,e\ correct\mid k,S).
\]

使用层次 Beta-Binomial 或 Dirichlet-Multinomial 收缩，并保留 LEDGER 的 leave-one-item fragility。

## 5.2 局部温度

对目标题 \(i\)，定义源先验保留率：

\[
\tau_i=\sigma(
\alpha_0
-\alpha_1 d^{AFGW}_i
+\alpha_2 q^{orbit}_i
+\alpha_3 c^{state}_i
-\alpha_4 u^{source}_k).
\]

其中：

- \(d^{AFGW}_i\)：局部关系几何失真；
- \(q^{orbit}_i\)：干预关系可恢复度；
- \(c^{state}_i\)：源目标能力状态覆盖；
- \(u^{source}_k\)：源 repair edge 不确定性/脆弱性。

软化后的先验：

\[
\operatorname{logit}\tilde\pi_{i,e\to f}
=\tau_i\operatorname{logit}\pi^S_{e\to f,k}.
\]

当 \(\tau_i\to0\) 时，先验趋于无信息，而不是自动回退；目标干预证据仍可以独立支持 BBH 式切换。当 \(\tau_i\to1\) 时，MMLU-Pro test/GAOKAO 可保留 source repair graph 的优势。

## 5.3 与 ATLAS 的根本差异

ATLAS strict TRIDENT 问：“源 repair edge 是否足够强，足以允许切换？”

SPECTRA 问：“在这一道目标题上，源 repair prior 应当在最终后验中保留多少温度？”

前者是 hard permission，后者是 uncertainty-aware information fusion。

---

# 6. QUASAR：带干预轨道与 lineage 依赖的潜变量推断

## 6.1 潜变量

对目标样本 \(i\)：

- \(z_i\)：潜在语义答案；
- \(k_i\)：潜在能力原子；
- \(r_i\)：可路由状态；
- \(\theta_{e,k}\)：专家在能力原子 \(k\) 上的可靠性；
- \(b_{\ell,k}\)：lineage \(\ell\) 的共享偏差；
- \(h_{ie}\)：专家干预指纹。

观测似然不只包含原始答案，还包含整个干预轨道：

\[
p(A_i,A_i^{orb}\mid z_i,k_i,\theta,b,h).
\]

## 6.2 lineage 相关性

经典无监督 ensemble 方法通常需要或近似依赖条件独立性。LLM 专家显然可能因共同底座、蒸馏或训练数据而高度相关。

SPECTRA 使用两层结构：

\[
\eta_{ie}=\theta_{e,k_i}+b_{\ell(e),k_i}+\epsilon_{ie},
\]

并以目标干预轨道估计同 lineage 中“共同稳定、共同失灵”的程度。若五个 Qwen 衍生模型在所有 option permutation 中同步翻转，它们的有效证据不会被计为五票。

## 6.3 source anchoring

纯无标签潜变量模型存在 label switching 和一致错误问题。SPECTRA 用三种锚点降低不可识别性：

1. 源 gold 固定答案语义方向；
2. exact equivariant mapping 固定选项置换后的答案映射；
3. SOFT-LEDGER 为专家—能力关系提供不确定先验。

这不能在任意相关专家池上保证恢复真值，但比单次 output embedding 或普通 Dawid–Skene 式模型拥有更多可检验约束。

## 6.4 目标函数

可使用变分推断最大化：

\[
\mathcal L=
\mathbb E_q[\log p(A,A^{orb}\mid z,k,\theta,b)]
-\mathrm{KL}(q(\theta,k,b)\Vert p_{soft-ledger})
-\lambda_{null}\mathcal L_{null}
-\lambda_{corr}\mathcal R_{lineage}.
\]

所有超参数只通过 source leave-one-category/group-out 确定；目标上只执行预先冻结的推断规则。

---

# 7. 动态 target pseudo-base

## 7.1 pseudo-base 是答案簇，不先绑定单一模型

对原始目标题的候选答案 \(c\)，计算：

\[
Q_i(c)=P(z_i=c\mid A_i,A_i^{orb},S).
\]

定义：

\[
c_i^{PB}=\arg\max_c Q_i(c).
\]

支持该答案的专家集合为：

\[
E_i^{PB}=\{e:a_{ie}=c_i^{PB}\}.
\]

最终在该集合中选择具有最高干预后可靠性、最低 lineage 冗余和最低成本的专家。这样 pseudo-base 每题变化，不再等同于 source-global best。

## 7.2 challenger 与 fallback

- **pseudo-base**：目标干预后后验最强的答案簇；
- **challenger**：后验第二强但具有更强 source repair prior 或更强 evidence specificity 的答案簇；
- **strict fallback**：预先冻结的 best single；
- **no-decision**：证据不足，直接 fallback，不把随机小 margin 当成可路由性。

## 7.3 防止 pseudo-base 自举崩溃

pseudo-base 不能直接作为伪标签反复训练。SPECTRA 采用三条限制：

1. 一次性后验推断，不在同一目标上循环强化自己的答案；
2. 目标 fold 间交叉拟合，某题的超参数不由该题自身决定；
3. negative-control survival 高的答案簇不能成为 pseudo-base。

---

# 8. ROUTE-PHASE：可路由性不是数据集的固定属性

## 8.1 新定义

定义：

\[
\mathcal R(D,E,T),
\]

其中：

- \(D\)：目标数据分布；
- \(E\)：当前专家池及 lineage 结构；
- \(T\)：可用干预探针集合。

同一个 GPQA 数据集，若专家全部同源且准确率接近随机，可能不可路由；加入一个真正不同的科学推理专家或一组高有效性的等变探针后，可能变为局部可路由。因此不能写“GPQA 不可路由”，只能写“当前专家池与探针预算下，GPQA 的可路由机会很低”。

## 8.2 五维相图

目标级可路由性由五个无标签维度构成：

| 维度 | 含义 | 高值作用 |
| --- | --- | --- |
| \(G\) Grounded orbit gap | 真干预与 sham/decoupled null 的差距 | 共识更可能由语义证据驱动 |
| \(D_{eff}\) Effective diversity | 有效独立 lineage 数 | 降低共享错误 |
| \(A\) Alignment | source state/repair prior 的局部适配度 | 源知识可借用 |
| \(C\) Concentration | pseudo-base 后验 margin 与 split 稳定性 | 可形成明确决策 |
| \(N\) Null survival | 语义链接破坏后路由分数仍存活的比例 | 高值是危险信号 |

预注册的 routability score：

\[
R_D=w_GG+w_D\log(1+D_{eff})+w_AA+w_CC-w_NN,
\]

权重仅通过源 pseudo-target 环境学习。

## 8.3 三种部署相位

### Green：ensemble-routable

- 高 \(G,D_{eff},C\)，低 \(N\)；
- 允许 LEAF-like 大规模目标驱动切换；
- 预期 BBH 属于此相位。

### Amber：selectively routable

- 部分能力原子有高干预证据；
- 只在逐题证书通过时切换；
- 预期 MMLU-Pro test、GAOKAO、MathVista、CMMMU 属于此相位。

### Red：fallback-dominant

- posterior 接近随机、lineage 高相关或 null survival 高；
- 只保留极少数 exact-equivariance + strong prior 题；
- 预期当前专家池下 GPQA、MMStar 多数区域属于此相位。

这些是预实验假设，必须在不看目标标签的情况下先给出相位预测，再解封评分。

---

# 9. 视觉语言突破：用干预响应雅可比发现能力原子

## 9.1 不再只给图片贴“图表/OCR/几何”标签

静态 image classifier 只能说明图中有什么，不能说明某个专家是否真正使用了相关视觉证据。SPECTRA 使用专家响应对图像干预的变化，构造 **behavioral response Jacobian**：

\[
J_{i,e,m}=d(p_{ie},p^{(m)}_{ie}),
\]

其中 \(m\) 是 OCR-mask、legend-mask、axis-mask、geometry-mark-mask、background-sham、resize、recolor 等操作。

跨专家聚合得到：

\[
J_i=[J_{i,:,1},\ldots,J_{i,:,M}].
\]

对 \(J_i\) 聚类或使用稀疏因子模型，发现目标中的视觉能力原子。能力原子不是仅由图片内容定义，而是由“专家池在何种证据破坏下发生怎样的响应”定义。

## 9.2 可能出现的原子

- **OCR-dependent**：遮挡文字区域显著改变回答，背景 sham 不改变；
- **chart-legend coupled**：图例/颜色映射干预敏感，坐标轴轻微变化稳定；
- **geometry-relation**：对旋转/反射等变，对关键角度标记消融敏感；
- **language-dominant**：图像干预几乎无效，题干改写更敏感；
- **visual-ignored**：遮挡大部分视觉证据仍保持原答案，可能依赖语言先验；
- **globally brittle**：任何同面积改动都导致翻转，不应被解释为 grounding。

## 9.3 三个视觉目标的专门处理

### MathVista

- 优先 chart/geometry/OCR 三类程序化或半程序化干预；
- 用 source MMMU-Pro 标签验证哪些 Jacobian 因子与专家 correctness 相关；
- 只在 active atom 有源覆盖且 grounded orbit gap 为正时使用 FATE/DARE prior。

### CMMMU

将三个方向分开：

1. 中文语言扰动：中文数字格式、同义改写、选项字母置换；
2. OCR 扰动：中文文本框遮挡、分辨率下降、背景 sham；
3. 视觉知识扰动：非文字对象区域与文字区域分别消融。

由三类响应区分“中文语言失败”“中文 OCR 失败”“一般视觉失败”，避免把它们压成一个 failure ecology。

### MMMU-Pro test-id

- 第一阶段使用 subject/taxonomy posterior；
- 第二阶段在 subject 内使用 response Jacobian 选择专家；
- 10-option 任务使用归一化 entropy、option permutation 与 effective choice count，不能直接复用四选一阈值。

## 9.4 关键风险

证据消融后的答案改变既可能表示真正 grounding，也可能表示模型脆弱。因此必须使用 matched sham 的 difference-in-differences，而不能把“改变越多”直接当成“理解越好”。

---

# 10. EVIDENCE-BUDGET：序贯探针与无标签拒绝策略

## 10.1 为什么不能对所有模型、所有题生成大量变体

八个目标总样本量大，若每个专家对每题运行多个变体，调用成本可能超过路由收益。SPECTRA 采用 coarse-to-fine 预算。

## 10.2 序贯流程

### Stage 0：零额外调用

用现有缓存输出计算：

- lineage-capped agreement；
- answer entropy；
- source soft prior；
- A-FGW local distortion；
- 明显 fallback 题。

若所有 proposal 都选择 best single，立即停止。

### Stage 1：一个 exact probe

对存在 actionable disagreement 的题，只运行一次 option permutation。若候选共识不能保持语义等变，回退。

### Stage 2：第二个独立 exact/near-exact probe

仅对第一阶段仍不确定的题运行第二次 permutation、变量重命名或图像无损变换。

### Stage 3：能力特异探针

仅对视觉 amber items 运行一个关键区域 mask 与一个 matched sham mask。

### Stage 4：停止

当 pseudo-base posterior 超过预注册阈值、或证据确定不足时停止。证据不足不是失败，而是 fallback 的正常终点。

## 10.3 负对照校准

对真实路由统计量 \(s_i\)，构造不使用 gold 的 null worlds：

- 将干预输出与同选项数的其他题错配；
- 使用错误的 option inverse mapping；
- 在 lineage 内复制/置换专家身份；
- 将关键 mask 与 matched sham 标签交换。

定义随机化 p-value：

\[
p_i=\frac{1+\sum_{b=1}^{B}\mathbf 1[s_{ib}^{null}\ge s_i]}{B+1}.
\]

它检验的是“路由统计量是否依赖真实的题目—输出语义链接”，不是直接检验“候选答案一定正确”。论文必须明确这一边界。

对多个 proposed switches 可使用预注册的 BY/e-BH 型规则控制错误证据发现率；不能把这种控制误写成 accuracy 或 harm 的严格保证。

---

# 11. 完整算法

## 11.1 训练/校准阶段

```text
Input:
    labeled source S
    source expert outputs A_S
    lineage map L
    frozen intervention library T

1. For each intervention family t:
       apply t to source items
       validate answer relation using source gold
       fit validity posterior rho_t
       disable t if its pre-registered validity lower bound fails

2. Fit hierarchical SOFT-LEDGER:
       repair posterior pi_S[e -> f, state]
       harm posterior
       leave-one-item fragility

3. Build source intervention fingerprints h_S
4. Learn source capability atoms and expert priors
5. Run leave-one-category/group-out pseudo-target episodes:
       freeze thresholds for soft-prior temperature
       freeze routability phase boundaries
       freeze sequential probe stopping rule
       freeze fallback policy

Output:
    frozen source priors, intervention validity ledger,
    inference parameters, refusal thresholds
```

## 11.2 目标阶段

```text
Input:
    unlabeled target inputs U
    original target expert outputs A_U
    all frozen source artifacts

1. Compute zero-call target ecology and local A-FGW distortion
2. Identify actionable-disagreement items
3. Sequentially acquire exact and capability-specific probe outputs
4. Build signed intervention fingerprints h_U
5. Infer latent answer posterior Q_i(c), capability atom k_i,
   lineage dependence and effective diversity
6. Temper source repair priors locally with tau_i
7. Construct dynamic pseudo-base c_i^PB
8. Compute item and dataset routability phase
9. Calibrate routing evidence against unlabeled null worlds
10. If evidence and phase pass frozen policy:
        choose an expert supporting pseudo-base/challenger
    else:
        strict fallback to predeclared best single
11. Save all decisions and hashes before opening target labels
12. Use target labels once for final scoring and paired inference
```

## 11.3 简化伪代码

```python
for item in target_unlabeled:
    prior = soft_ledger(source_ledger, local_alignment(item))
    base_stats = cached_output_ecology(item, lineage)

    if no_actionable_disagreement(base_stats):
        output[item] = best_single[item]
        continue

    orbit = []
    for probe in sequential_probe_policy(item, base_stats):
        orbit.append(run_experts(probe(item)))
        posterior = quasar(item, orbit, prior)
        if posterior.certified or posterior.futile:
            break

    pseudo_base = argmax(posterior.answer_prob)
    p_null = semantic_linkage_randomization_test(item, orbit)
    phase = route_phase(posterior, orbit, lineage, p_null)

    if frozen_switch_rule(phase, posterior, p_null):
        output[item] = best_expert_supporting(pseudo_base)
    else:
        output[item] = best_single[item]
```

---

# 12. 如何解释当前八个目标

## 12.1 BBH

预期：Green。

待验证解释：BBH 的大样本量、多个可程序化逻辑结构和 lineage 去相关共识，使 exact equivariance 与 split stability 同时较高。SPECTRA 应保留 LEAF 的大部分收益，同时剔除在 option permutation 或变量重命名下崩塌的伪共识。

关键目标不是刷新 77.85%，而是**在不看 BBH 标签前预测它属于 Green，并预测 GPQA/MMStar 不属于 Green**。

## 12.2 MMLU-Pro test

预期：Amber/Green 边界。

同 benchmark 结构使 SOFT-LEDGER 温度较高；exact option permutation 主要过滤位置偏差。SPECTRA 应接近 ECR 60.31%，而不是强行取代 ECR。

## 12.3 GAOKAO

预期：Amber。

RepairChain prior 仍有价值，但中文改写的 validity 必须先在 source 或独立开发集验证。option permutation 是主 probe，生成式中文 paraphrase 只能作为次级证据。

## 12.4 GPQA

预期：Red 为主、少数 Amber islands。

当多个专家都接近随机时，高 agreement 也可能稳定错误。SPECTRA 应通过低 effective diversity、低 posterior margin 或高 null survival 自动减少切换。只在 exact equivariance、独立 lineage 支持和 BRES residual evidence 同时出现时保留少数路由机会。

## 12.5 MMStar text-only

预期：Red。

视觉 benchmark 的 text-only 切片与 MMLU source 的能力几何可能错位。若 option permutation 无法产生明显 grounded orbit gap，系统应回退 24.47% best single，不再追逐约 4 题的 point gain。

## 12.6 MathVista

预期：Amber，最值得新增视觉探针。

FATE +1.90% 表明存在可迁移 failure ecology。response Jacobian 可进一步限定在 OCR、chart 或 geometry 的局部高证据状态中切换，减少全局套用 FATE 的 harm。

## 12.7 CMMMU

预期：Amber/Red 混合。

必须将中文语言、OCR、一般视觉知识拆开。若 OCR-mask 与 sham-mask 的差值只在部分题显著，路由应局限于这些题，而不是给整个 CMMMU 一个统一 DARE 权重。

## 12.8 MMMU-Pro test-id

预期：subject-first Amber。

先用 dynamic subject discovery 产生能力先验，再用干预指纹做局部专家选择。若第二阶段没有额外 evidence，保持 subject router 或 best single，不强制 failure graph。

---

# 13. 预注册实验设计

## 13.1 Phase A：只验证“干预证据是否有信息”

### Source LOCO/LOGO

- MMLU-Pro validation：leave-one-category-out；
- MMMU-Pro validation-id：leave-one-subject/skill-group-out；
- 比较 source correctness 与：普通 agreement、option-equivariance、双向 signed score、完整 orbit posterior 的关系；
- 所有阈值在此冻结。

### 成功标准

1. signed score 相比普通 consistency 更能预测 source repair/harm；
2. lineage duplication 不会显著增加后验置信；
3. sham mask 能显著降低将一般脆弱性误判为 grounding 的比例；
4. 变换 validity 在 source outer folds 中稳定。

### 停止条件

- exact/validated orbit score 在 source outer folds 中与 correctness 或 repair 无稳定关系；
- 结果仅由某一个模型或某一类题驱动；
- paraphrase 等近似干预的语义破坏率超过预注册阈值。

## 13.2 Phase B：目标无标签 routability 预判

在完全不读取目标标签时，输出八个目标的：

- Green/Amber/Red phase；
- predicted safe switch rate；
- expected useful proposal family；
- refusal rate；
- uncertainty interval；
- 配置文件 hash 与时间戳。

核心检验：是否能在解封标签前将 BBH 与 GPQA/MMStar 分开。

## 13.3 Phase C：低成本 pilot

使用固定随机种子从每个目标抽取无标签 diagnostic subset，例如 256 题；标签保持封存。

- 所有题：1 次 option permutation；
- 仍有 actionable disagreement 的题：再加 1 次 permutation；
- 视觉 amber items：1 个关键 mask + 1 个 sham mask；
- 仅选择 3–5 个 lineage 代表做 probe，原始投票仍可使用完整专家池。

根据无标签 orbit likelihood 和 null gap 检查方法是否运行正常；不得根据 pilot accuracy 选参数。

## 13.4 Phase D：前瞻式评分

冻结后运行：

- best single；
- 当前 target-specific winner；
- LEAF/BRES/ECR/FATE/subject discovery；
- ATLAS main；
- SPECTRA full；
- SPECTRA strict fallback；
- 所有预注册消融。

最终一次性解封 gold，报告 paired gain、selection-aware simultaneous CI、switch repair/harm、coverage-risk、额外调用成本。

---

# 14. 必做消融

| 编号 | 变体 | 回答的问题 |
| --- | --- | --- |
| A0 | best single | 无路由基准 |
| A1 | LEAF/BRES/ECR/FATE best predeclared | 当前方法上界对照 |
| A2 | static QUASAR only | 潜变量模型本身是否有效 |
| A3 | + option equivariance | 最便宜 exact probe 的价值 |
| A4 | + invariance only | 是否只是普通 consistency |
| A5 | + evidence ablation, no sham | 会否奖励一般脆弱性 |
| A6 | + matched sham | 是否得到真正 specificity |
| A7 | hard LEDGER | 复现 ATLAS 过保守问题 |
| A8 | SOFT-LEDGER | 软先验是否保留 BBH 信号 |
| A9 | fixed source base | 固定基准的损失 |
| A10 | dynamic pseudo-base | 逐题目标基准的价值 |
| A11 | ignore lineage | 相关专家重复计票的伤害 |
| A12 | no null calibration | 伪语义路由信号的影响 |
| A13 | no refusal | GPQA/MMStar 负迁移是否增加 |
| A14 | visual static tags | 静态 OCR/chart 标签基线 |
| A15 | response Jacobian atoms | 行为型视觉能力原子的价值 |
| A16 | full probes | 效果上界 |
| A17 | sequential budget | 成本—收益折中 |

---

# 15. 预期指标与不应越界的表述

## 15.1 主要指标

- accuracy 与 best single paired gain；
- selection-aware simultaneous CI；
- switch rate；
- repair / harm / net repair；
- risk–coverage curve；
- Green/Amber/Red phase prediction accuracy；
- extra calls、tokens、latency；
- null survival rate；
- intervention validity；
- visual atom split stability。

## 15.2 预注册目标，不是结果承诺

- BBH：保留至少 80% 的 LEAF 增益，即相对 best single 约 +7.84 个百分点；
- MMLU-Pro test / GAOKAO：保留当前稳定增益的大部分；
- GPQA：避免 ATLAS Tapestry 式显著负迁移，默认以非劣于 best single 为目标；
- MMStar：高比例自动回退；
- MathVista：先证明视觉 atom 能提升 repair/harm ratio，再追求总体 accuracy；
- CMMMU：证明能区分语言/OCR/视觉状态；
- MMMU-Pro test-id：证明 subject-first 策略优于统一 failure router。

这些目标必须写入预注册；若未达到，应如实报告，不能事后更换主指标。

## 15.3 不能声称的保证

- 干预一致不等于答案正确；
- negative-control p-value 不等于 correctness p-value；
- evidence-ablation sensitivity 不自动等于 causal understanding；
- 无标签 refusal 不能严格保证相对 best single 非劣；
- 在高度相关且全部弱于随机的专家池中，任何无标签 router 都可能不可识别。

---

# 16. 快速、可缓存的实现路线

## 16.1 最小版本 SPECTRA-Lite

只实现：

1. option permutation 两次；
2. lineage-capped semantic answer vote；
3. SOFT-LEDGER temperature；
4. dynamic pseudo-base；
5. question–orbit decoupling null；
6. Red-phase fallback。

先在 BBH、GPQA、MMStar 三个最有诊断价值的语言目标运行。

若 SPECTRA-Lite 不能在不看标签时给出 BBH 高 routability、GPQA/MMStar 低 routability，则不进入昂贵视觉版本。

## 16.2 视觉版本

优先 MathVista，再 CMMMU，最后 MMMU-Pro test-id：

1. OCR detector 获取文本框；
2. chart/geometry 轻量 detector 提供候选区域；
3. 关键区域与 matched background mask；
4. response Jacobian；
5. source-anchored sparse factor model；
6. 只在稳定 atom 内执行 FATE/DARE proposal。

## 16.3 调用预算建议

- 不对已无分歧或所有 proposal 都同意 best single 的题做 probe；
- 每个 lineage 先选一个代表模型；
- exact option probe 优先于生成式 paraphrase；
- 只有 Stage 1 无法决定的题进入视觉/语义重探针；
- 报告每获得一个净正确答案所需的额外模型调用数。

---

# 17. 与现有工作的原创边界

## 17.1 近邻工作

- [Smoothie](https://arxiv.org/abs/2412.04692) 使用无标签 LLM 输出和潜变量图模型估计逐样本模型质量；
- [Estimating the accuracies of multiple classifiers without labeled data](https://proceedings.mlr.press/v38/jaffe15.html) 与 [dependent classifiers](https://proceedings.mlr.press/v51/jaffe16.html) 研究无标签 ensemble 可靠性及相关分类器；
- [LLM option-order sensitivity](https://arxiv.org/abs/2308.11483) 说明多项选择模型可能对选项顺序高度敏感；
- [MetaRA](https://arxiv.org/abs/2605.19307) 使用 metamorphic relations 评估多模态 VQA 鲁棒性；
- [Classification with Rejection](https://proceedings.mlr.press/v139/charoenphakdee21a.html) 提供一般性的拒绝分类理论；
- [TTAB](https://proceedings.mlr.press/v202/zhao23d.html) 指出无标签 test-time adaptation 的模型选择和评测容易失真。

## 17.2 SPECTRA 的合理新边界

不能声称：

- 首次使用 metamorphic testing；
- 首次使用 option permutation；
- 首次无标签估计专家可靠性；
- 首次使用拒绝策略；
- 首次做 VLM 干预评测。

较合理的论文主张是：

> SPECTRA-CoE 将源 benchmark 的不确定 repair posterior、目标逐题的带符号干预轨道、lineage-dependent 潜在答案推断和 pool-conditioned routability phase 统一到一个 target-label-free expert routing 框架中。与只使用静态输出相似度或普通一致性的无标签路由不同，SPECTRA 同时要求对答案保持变换稳定、对已知映射变换等变，并通过关键证据与 matched-sham 消融的差分响应检验 grounding；源 repair graph 只作为由局部失真控制的软先验，最终基准由目标干预后验逐题生成。

更短的英文表述：

> We introduce SPECTRA-CoE, a target-label-free benchmark-derived expert router that treats signed metamorphic response orbits as weak supervision for dynamic pseudo-base construction. SPECTRA combines source-calibrated repair priors with lineage-aware latent answer inference, but tempers source evidence by local transfer distortion instead of using it as a hard gate. Routability is modeled as a property of the target–expert-pool–probe triad, and switching is permitted only when semantic-equivariance and evidence-specificity signals survive matched unlabeled null worlds.

## 17.3 真正需要文献检索确认的句子

投稿前仍需系统检索 2026 年最新工作，尤其是：

- signed metamorphic relations 是否已被用于逐题 LLM routing；
- matched evidence/sham ablation 是否已用于无标签 VLM router；
- target–expert-pool–probe triad routability 是否已有同等定义；
- source repair posterior 的 local-temperature transfer 是否已有近似方法。

因此当前最多使用 “to our knowledge”，不能使用无条件的 “the first”。

---

# 18. 论文级核心故事

## 中文版本

现有 benchmark-derived expert routing 通常从一次性的目标专家输出中提取共识、分歧或残差证据。然而，静态输出无法区分由正确推理产生的共识与由共享 lineage 偏差产生的伪共识。这一不可识别性在我们的跨目标实验中表现得尤为明显：lineage-aware posterior voting 在 BBH 上获得 9.80 个百分点的稳定增益，但类似的无标签输出信号在 GPQA 和 MMStar 上几乎无效，甚至导致负迁移。

我们提出 SPECTRA-CoE，将每个目标样本扩展为具有已知答案关系的干预轨道。SPECTRA 同时检验专家对答案保持变换的稳定性、对选项置换的语义等变性，以及对关键证据消融相对于 matched sham 消融的选择性响应，从而获得不依赖目标标签的带符号证据。源 benchmark 上的 repair graph 不再充当硬门控，而被局部关系失真与干预可恢复度连续衰减；目标 pseudo-base 则由 lineage-dependent 潜在答案后验逐题产生。由此，可路由性被重新定义为目标分布、专家池和可用探针的联合属性，系统能够在 BBH 式高证据区域保留目标驱动协作，在 GPQA/MMStar 式低证据区域主动拒绝迁移，并通过视觉响应雅可比发现 OCR、图表和几何等行为能力原子。

## 英文版本

> Benchmark-derived expert routers typically infer consensus, disagreement, or residual evidence from a single matrix of target predictions. Such static observations cannot identify whether agreement arises from correct reasoning or correlated lineage bias. This ambiguity is exposed by our results: lineage-aware posterior voting yields a robust 9.8-point gain on BBH, while closely related unlabeled-output signals provide little or negative value on GPQA and MMStar. We propose SPECTRA-CoE, which expands each unlabeled target query into a signed intervention orbit with known invariance, equivariance, evidence-ablation, and matched-sham relations. These relations supply weak supervision for lineage-aware latent answer inference and dynamic pseudo-base construction. Source repair posteriors are locally tempered by transfer distortion and orbit recoverability rather than used as hard gates. SPECTRA further defines routability as a property of the target–expert-pool–probe triad and discovers multimodal capability atoms through behavioral response Jacobians. This design aims to preserve target-driven gains in routable regimes while refusing unsupported transfer under low-evidence, highly correlated expert ecosystems.

---

# 19. 最优先的四个实验

1. **三目标鉴别实验**：只用 BBH、GPQA、MMStar 的无标签输出与两次 option permutation，检查 SPECTRA-Lite 能否在解封标签前给出 Green/Red 区分。
2. **动态 pseudo-base 实验**：比较 source-global base、static target majority、lineage LEAF base、SPECTRA orbit pseudo-base。
3. **软先验实验**：比较 hard LEDGER、无 LEDGER、固定温度 LEDGER、local-temperature SOFT-LEDGER，重点观察 BBH 保留率与 GPQA harm。
4. **MathVista 视觉差分实验**：对 OCR/chart/geometry 候选区域做关键 mask 与等面积 sham mask，验证 response Jacobian 是否能预测 FATE switch 的 repair/harm。

如果第一个实验失败，即无标签 routability 相图无法在 BBH 与 GPQA/MMStar 之间形成稳定差异，则应停止 SPECTRA 主线。若成功，即使 accuracy 暂未超过现有 winner，也已经得到一个更重要的论文贡献：在不看目标标签前判断 **何时专家共识具有可路由含义、何时系统应拒绝相信共识**。

---

# 结论

本次创新不再把问题表述为“寻找下一种更强的 router”，而是将其改写为：

> 在目标标签不可见时，能否通过答案关系明确的主动干预，为专家共识构造一个弱预言机，并据此识别可路由区域、动态建立逐题基准、软化源迁移先验以及拒绝不可识别的切换？

SPECTRA-CoE 的潜在突破点不是必然获得更高的单表 accuracy，而是首次在当前 Bench-CoE 实验链条中把 **静态共识的不可识别性、主动无标签干预、动态 pseudo-base、pool-conditioned routability 和视觉行为能力原子** 组织成一个可证伪的统一框架。它直接回应了 ATLAS hard gate 过保守、BBH 依赖目标结构、GPQA/MMStar 需要拒绝、视觉任务缺少图像状态的全部关键负结果。
