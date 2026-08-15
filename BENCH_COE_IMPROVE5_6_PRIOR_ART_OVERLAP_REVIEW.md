# Improve5 与 Improve6 已有论文思路重叠调研

> 调研对象：Bench-CoE 后续改进方法 5（DARE Reliability）与改进方法 6（RepairChain）  
> 原始研究问题：在扩充模型和数据集后，Bench-CoE 泛化性不足且多数情况下未超过最佳单模型；实验验证发现只有 Improve5 和 Improve6 对泛化性有提升，需要审查其与已发表论文的思路重叠程度。  
> 调研时间口径：截至 2026 年 8 月 8 日  
> 依据材料：
>
> - `bench_coe_fcs.pdf`
> - `ALL_EXPERIMENTS_SUMMARY(1).md`
> - `IMPROVE5_6_REAL_EXAMPLE(1).md`

---

# 一、调研结论

截至 **2026年8月8日**，对经典动态分类器选择、动态集成选择、多模型回答选择、LLM/VLM 路由、不确定性集成、专家互补性和图路由文献进行交叉检索后，结论如下：

| 方法 | 与既有论文的重叠程度 | 能否作为论文主创新 | 最稳妥定位 |
|---|---:|---|---|
| **Improve5 / DARE Reliability** | **很高** | **不建议** | 将其作为经典 Output-Profile Dynamic Selection 在 LLM/VLM 场景中的适配基线或增强基线 |
| **Improve6 / RepairChain** | **构件层面较高，整体组合层面中等** | **有条件可以** | 将“失败条件专家纠错图及有限跳传播”作为主创新，但必须收窄表述并补齐强基线和消融 |
| Improve5/6 相对原 Bench-CoE | **方法类别已改变** | 必须重新定位 | 从单次推理路由变成多专家先生成、再进行后验回答选择 |

最关键的判断是：

> **Improve5 的核心计算流程与已有的 Multiple Classifier Behaviour、K-Nearest Output Profiles 和动态局部能力估计高度同构。Improve6 的基础信号也均有先例，但尚未检索到“失败条件有向纠错矩阵＋查询相关失败质量＋一跳/两跳传播＋候选回答选择”完全同构的已发表方法。**

因此：

> **Improve5 不适合作为新的算法主贡献；Improve6 比 Improve5 更有可能形成可发表的组合级方法创新。**

---

# 二、首先必须纠正方法类别：Improve5/6 已经不是原来的单专家路由

原 Bench-CoE 的实验协议明确规定：

- 路由器在推理前选择一个专家；
- 每个 Query 只激活一个专家；
- 最终比较的是该专家输出的准确率。

来源：`bench_coe_fcs.pdf` 第 5 页，解析文本约第 565–574 行。

但 Improve5 和 Improve6 都需要当前 Query 上多个乃至全部专家的已生成回答：

- Improve5 的答案支持率 \(A_m(q)\) 要统计所有 \(M\) 个专家是否给出相同答案；
- Improve5 的局部历史状态检索使用“当前所有专家的答案分布”，然后检索约 32 个相似历史输出状态；
- Improve6 的一跳和两跳传播发生在所有候选回答已经生成之后，并不是真正依次调用三个模型。

来源：

- `IMPROVE5_6_REAL_EXAMPLE(1).md` 第 82–123 行；
- `IMPROVE5_6_REAL_EXAMPLE(1).md` 第 608–622 行；
- `IMPROVE5_6_REAL_EXAMPLE(1).md` 第 466–472 行。

所以完整流程已经变成：

\[
\text{所有专家生成回答}
\rightarrow
\text{构造回答状态和不确定性}
\rightarrow
\text{后验评分}
\rightarrow
\text{选择一个已有回答}
\]

学术上，这更接近：

- **post-hoc response selection**；
- **multi-inference ensemble selection**；
- **dynamic classifier/model selection from completed outputs**；

而不是原 Bench-CoE 的：

- **pre-inference single-model routing**。

MORE 在 2023 年已经明确区分了这两种设置：

- 完整 MORE 让每个专家先产生回答；
- 再利用回答置信度和专家间一致性选择答案；
- question-only router 则是不生成所有答案、直接选择一个专家。

参考：

- [MORE: Multi-Model Answer Selection, Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.552.pdf)

Smoothie 同样先获得不同 LLM 的输出，再计算样本相关质量分数并选择最高分输出。

参考：

- [Smoothie, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6b57a990462df5afa58d64ce2709db9-Paper-Conference.pdf)

这意味着论文中不能继续笼统声称 Improve5/6：

- 每题只调用一个专家；
- 保持原 Bench-CoE 的低推理成本；
- 与 RouteLLM、GraphRouter 等单调用路由器处于相同推理预算；
- 只需更新 subject-expert mapping 即可替换专家。

原始 subject 路由在专家池变化时可以只更新映射而不重训分类器。

来源：

- `ALL_EXPERIMENTS_SUMMARY(1).md` 第 5–8 行；
- `bench_coe_fcs.pdf` 第 5 页，解析文本约第 496–511 行。

但 Improve5/6 依赖：

- \(N_s\times M\) 正确性矩阵；
- \(M\times M\) 纠错矩阵；
- 当前专家池的输出状态空间；
- 各专家全局准确率和局部邻域。

新增或替换专家后，至少需要重新获得该专家在源域上的预测，并更新：

- 输出状态空间；
- 全局准确率；
- 局部邻域；
- 纠错图。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 55–72 行、第 295–325 行。

---

# 三、Improve5：核心思路与经典动态分类器选择高度重叠

## 3.1 Improve5 的本质

Improve5 最终使用：

\[
S_m^{\mathrm{DARE}}(q)
=
0.44L_m(q)
+0.28R_m^{\mathrm{robust}}
+0.24B_m(q)
+0.04G_m ,
\]

其中：

- \(L_m(q)\)：相似历史输出状态中的局部成功率；
- \(R_m^{\mathrm{robust}}\)：不同源域划分上的均值减标准差惩罚；
- \(B_m(q)\)：答案支持率经过不确定性修正后的行为稳定性；
- \(G_m\)：源域全局准确率。

这些权重均为预设常数，不是训练得到的。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 207–291 行。

真正决定 Improve5 方法身份的核心是：

\[
\boxed{
\text{当前多专家输出状态}
\rightarrow
\text{检索相似历史输出状态}
\rightarrow
\text{估计各专家局部正确率}
\rightarrow
\arg\max_m
}
\]

这一结构并不是新的。

---

## 3.2 与 Multiple Classifier Behaviour 的近乎直接重叠

Giacinto 和 Roli 在 2001 年提出的：

> **Dynamic Classifier Selection Based on Multiple Classifier Behaviour，MCB-DCS**

已经使用以下流程：

1. 对一个未知样本运行多个分类器；
2. 将所有分类器的决策组成 multiple-classifier-behaviour 向量；
3. 从训练或验证集中寻找邻近样本；
4. 保留与当前样本具有相似分类器行为的邻居；
5. 在这些邻居上估计每个分类器的局部准确率；
6. 选择局部准确率最高的分类器。

参考：

- [Dynamic Classifier Selection Based on Multiple Classifier Behaviour](https://www.sciencedirect.com/science/article/abs/pii/S0031320300001503)

与 Improve5 对照：

| MCB-DCS | Improve5 |
|---|---|
| 多分类器决策向量 | 多专家回答状态/答案分布 |
| 相似 classifier behaviour | 相似历史 output state |
| 邻域上的 classifier local accuracy | \(L_m(q)\) |
| 选择局部能力最高的分类器 | 选择综合分数最高的专家 |

因此，Improve5 的主体并不只是“理念类似”，而是具有明显的**算法结构同构性**。

区别主要只是：

- 传统分类器输出离散类别，Improve5 输入是 LLM/VLM 答案；
- Improve5 使用了相似度加权的局部正确率；
- Improve5 额外混入了跨划分可靠性、回答支持率和全局准确率。

这些差异足以构成应用扩展，但不足以把核心方法描述成全新的动态专家选择原理。

---

## 3.3 与 K-Nearest Output Profiles 的重叠更强

后续动态集成选择文献提出了：

> **K-Nearest Output Profiles，KNOP**

该类方法不是在原始特征空间中寻找邻居，而是在分类器的：

- 决策空间；
- 输出画像空间；

中寻找相似历史样本，再根据各分类器正确处理这些邻居的次数估计其能力。

参考：

- [K-Nearest Output Profiles / Dynamic Ensemble Selection](https://www.sciencedirect.com/science/article/abs/pii/S0031320312001124)

Improve5 文档明确写道：

> 系统根据当前所有专家的答案分布，在 MMMU-Pro 中检索约 32 个输出状态相似的问题，再计算各专家的局部成功率。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 608–622 行。

这与 KNOP 的核心结构可以写成：

\[
\text{Output Profile}
\xrightarrow{\mathrm{KNN}}
\text{Local Competence}
\xrightarrow{\mathrm{Selection}}
\text{Expert Output}.
\]

Improve5 只是将传统分类器的类别预测向量换成了多 LLM/VLM 的回答状态，并在局部能力上增加额外统计项。

因此：

> **Improve5 最接近的已有方法不是通常的 LLM query router，而是经典的 output-profile dynamic classifier/ensemble selection。**

这是当前初稿相关工作中最需要补充的一条文献线。

---

## 3.4 答案支持率和不确定性同样已有充分先例

Improve5 使用：

\[
B_m(q)=A_m(q)\left(1-0.35U_m(q)\right).
\]

其中：

- \(A_m(q)\) 是专家间答案支持比例；
- \(U_m(q)\) 是输出不确定性。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 235–256 行。

这两个信号也不是新的：

### MORE

MORE 让所有专家生成答案，答案选择器同时使用：

- 答案频率；
- 专家间 agreement；
- 模型 confidence；
- 回答特征。

参考：

- [MORE, Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.552.pdf)

### Smoothie

Smoothie 根据多个 LLM 输出之间的关系建立潜变量图模型，计算样本相关质量分数，并提供：

- GLOBAL；
- 基于最近邻的 LOCAL；

两个版本。

参考：

- [Smoothie, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6b57a990462df5afa58d64ce2709db9-Paper-Conference.pdf)

### Uncertainty-Aware Answer Selection

2025 年 Findings of EMNLP 的 Uncertainty-Aware Answer Selection 已经专门研究：

- 利用校准后的模型不确定性；
- 在多个 LLM 已生成的回答中选择最可靠回答。

参考：

- [Uncertainty-Aware Answer Selection, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1367.pdf)

因此不能把：

> “答案支持率＋不确定性修正”

作为 Improve5 的主要新颖点。

此外，当前文档对 \(U_m(q)\) 只说明其被缩放、截断到约 \(0\sim1\)，没有给出完整的原始不确定性定义。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 126–139 行。

正式论文中必须明确它究竟是：

- token entropy；
- sequence log-likelihood；
- verbal uncertainty；
- 多次采样的语义熵；
- 格式异常分数；
- 若干启发式指标的组合。

否则既无法复现，也无法判断与已有 uncertainty selector 的具体区别。

---

## 3.5 跨划分可靠性是可能存在差异的部分，但创新强度不足

Improve5 定义：

\[
R_m^{\mathrm{robust}}
=
\max(\mu_m-0.55\sigma_m,0).
\]

在此次检索中，没有找到与该**精确公式和精确系数**完全一致的动态 LLM 回答选择论文。

但从算法性质看，它是一个手工设置的均值—波动惩罚：

- 均值高则加分；
- 跨划分标准差高则减分；
- \(0.55\) 没有对应置信水平或样本规模；
- 源域不能划分时又退化为全局准确率。

它可以作为一个合理的 robustification term，但很难单独支撑方法原创性。审稿人通常不会把一个未经学习、未经理论推导的常数加权项视为核心方法突破。

---

## 3.6 Improve5 的最终判断

### 重叠等级

| 组成部分 | 重叠程度 |
|---|---:|
| 输出状态 KNN 检索 | 很高 |
| 邻域局部专家准确率 | 很高 |
| 根据局部能力选择专家 | 很高 |
| 回答一致性/支持率 | 很高 |
| 不确定性修正 | 高 |
| 局部与全局能力混合 | 高 |
| \(\mu-0.55\sigma\) 精确项 | 较低 |
| 完整固定权重向量 | 未发现精确一致，但学术新颖性弱 |

**综合 prior-art 风险：很高。**

### 不建议的表述

> We propose a novel output-state-aware dynamic expert routing method.

> We are the first to retrieve similar historical expert-output patterns for expert selection.

> DARE introduces a new principle for local reliability-aware routing.

这些表述容易被 MCB、KNOP、MORE 和 Smoothie 直接反驳。

### 较安全的表述

> We adapt output-profile dynamic classifier selection to heterogeneous LLM/VLM response selection and augment local competence estimation with cross-partition reliability and response-level stability.

中文可写为：

> 我们将基于输出画像的动态分类器选择迁移到异构 LLM/VLM 专家池，并结合跨划分稳定性和回答级不确定性，对局部专家能力进行鲁棒调整。

但在这种定位下，Improve5 更适合作为：

- 一个强基线；
- 一个解释性方法；
- Improve6 的前置版本或消融版本；

而不适合成为论文唯一或第一主贡献。

---

# 四、Improve6：基础构件已有先例，但完整“修复传播”组合仍有一定新颖空间

## 4.1 Improve6 的核心结构

Improve6 首先定义：

\[
C_{i,j}
=
\frac{
\sum_{r:Y_{r,i}=0}Y_{r,j}+4G_j
}{
N_i^{\mathrm{fail}}+4
},
\]

表示当专家 \(i\) 在源域失败时，专家 \(j\) 成功的平滑条件比例。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 295–336 行。

然后根据当前答案支持率和不确定性构造失败信号：

\[
F_i(q)
=
\operatorname{clip}
\bigl(
1-A_i(q)+0.40U_i(q),0,1.6
\bigr),
\]

并归一化得到 \(\omega_i(q)\)。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 338–393 行。

传播过程为：

\[
\mathbf H^{(1)}(q)=\boldsymbol\omega(q)C,
\]

\[
\mathbf H^{(2)}(q)
=
\widehat{\mathbf H}^{(1)}(q)C.
\]

最终分数是：

\[
S_m^{\mathrm{RC}}(q)
=
0.30L_m(q)
+0.25H_m^{(1)}(q)
+0.18H_m^{(2)}(q)
+0.16A_m(q)
+0.11G_m .
\]

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 396–509 行。

它的潜在独特点不是某个单独指标，而是：

\[
\boxed{
\text{查询相关失败质量}
+
\text{有向失败条件互补图}
+
\text{有限跳传播}
+
\text{已有回答排序}
}
\]

---

## 4.2 条件纠错矩阵本身已有明确先例

2006 年 JMLR 的：

> *Ensemble Pruning via Semi-definite Programming*

已经构造分类器错误矩阵 \(P\) 和共同错误矩阵：

\[
G=P^\top P,
\]

并明确指出：

\[
\frac{G_{ij}}{G_{ii}}
=
P(j\text{错误}\mid i\text{错误}).
\]

该矩阵被用于刻画：

- 分类器的个体强度；
- 成对错误重叠。

参考：

- [Ensemble Pruning via Semi-definite Programming, JMLR](https://www.biz.uiowa.edu/faculty/nstreet/research/sdpprune_jmlr_final.pdf)

在二元正确/错误条件下，不考虑平滑时：

\[
P(j\text{正确}\mid i\text{错误})
=
1-P(j\text{错误}\mid i\text{错误}).
\]

所以 Improve6 的 \(C_{ij}\) 本质上是经典共同错误或 double-fault 信息的“救援能力”版本：

- 旧文献关注 \(i\) 错时 \(j\) 是否也错；
- Improve6 关注 \(i\) 错时 \(j\) 是否能答对。

因此不能主张：

> 首次利用专家 \(i\) 失败时专家 \(j\) 的条件成功率描述专家互补性。

这一基础思想已有明确 prior art。

---

## 4.3 平滑公式可以解释为标准的 Beta-Bernoulli 收缩估计

令：

\[
s_{ij}
=
\sum_{r:Y_{r,i}=0}Y_{r,j},
\qquad
n_i=N_i^{\mathrm{fail}}.
\]

Improve6 的公式为：

\[
C_{ij}
=
\frac{s_{ij}+4G_j}{n_i+4}.
\]

它可以精确解释为给条件成功率 \(p_{ij}\) 设置先验：

\[
p_{ij}
\sim
\operatorname{Beta}
\left(
4G_j,\,
4(1-G_j)
\right),
\]

则后验均值就是：

\[
\mathbb E[p_{ij}\mid s_{ij},n_i]
=
\frac{s_{ij}+4G_j}{n_i+4}.
\]

也就是说，该式是：

- 先验均值为专家 \(j\) 全局准确率 \(G_j\)；
- 先验强度为 4 个伪样本；
- 对条件纠错率做经验贝叶斯式收缩。

这是合理设计，但“加 4 并回缩到 \(G_j\)”本身并不是新的数学机制。真正可能形成创新的是它随后如何被查询相关失败质量激活和传播。

---

## 4.4 建模专家关系、能力依赖和图路由也已有相关工作

已有动态集成选择工作已经显式建模分类器能力之间的依赖。

### Probabilistic Classifier Chains for Dynamic Ensemble Selection

2017 年的 Probabilistic Classifier Chains 动态集成选择方法：

- 将每个分类器是否适合当前样本视为相关的多标签变量；
- 显式建模 competence-label dependencies。

参考：

- [Probabilistic Classifier Chains for Dynamic Ensemble Selection](https://ecmlpkdd2017.ijs.si/papers/paperID595.pdf)

### GNN-DES

GNN-DES：

- 将局部样本信息和分类器关系编码为图；
- 用图神经网络完成动态集成选择。

参考：

- [GNN-DES](https://dl.acm.org/doi/10.1007/978-3-031-42795-4_6)

### GraphRouter

ICLR 2025 的 GraphRouter：

- 构建 task、query 和 LLM 异构图；
- 通过边预测选择合适 LLM。

参考：

- [GraphRouter](https://arxiv.org/pdf/2410.03834)

### RouteMoA

ACL 2026 的 RouteMoA：

- 在后续层利用真实模型输出；
- 使用自评和交叉评价；
- 更新模型性能分数；
- 使用“来自模型输出的后验知识”修正最初的模型选择。

参考：

- [RouteMoA, ACL 2026](https://aclanthology.org/2026.acl-long.558.pdf)

因此下面这些宽泛表述都不安全：

> 首个基于图的 LLM 专家路由方法。

> 首个显式建模专家间关系的方法。

> 首个使用模型输出后验信息修正专家选择的方法。

这些方面都已有已发表论文。

---

## 4.5 尚未发现完全同构的部分

在此次检索到的论文中，尚未发现一个已发表方法同时满足以下全部条件：

1. 用源域正确性矩阵构建有向边：

   \[
   C_{ij}\approx P(j\text{成功}\mid i\text{失败});
   \]

2. 根据当前 Query 的专家答案支持率和不确定性，构造所有专家上的失败质量分布：

   \[
   \boldsymbol\omega(q);
   \]

3. 计算：

   \[
   \boldsymbol\omega(q)C
   \]

   作为直接修复证据；

4. 再进行一次归一化和第二次矩阵传播；

5. 将一跳、两跳、局部历史成功率、回答一致性和全局准确率联合用于选择一个**已生成的回答**。

最接近的文献分别只覆盖部分结构：

| 文献线 | 覆盖部分 | 未覆盖部分 |
|---|---|---|
| common-error / double-fault | 失败条件成对关系 | 没有 Query 相关失败质量和两跳回答选择 |
| MCB / KNOP | 输出画像和局部能力 | 没有有向条件纠错传播 |
| PCC-DES / GNN-DES | 分类器依赖或图关系 | 边语义不是 \(P(j\text{成功}\mid i\text{失败})\) |
| GraphRouter | 图建模 LLM 选择 | 是 query-task-LLM 图，不是失败修复图，且主要是预生成路由 |
| MORE / Smoothie | 使用多个已生成回答 | 没有源域失败条件纠错图和两跳传播 |
| RouteMoA | 用真实输出更新后续选择 | 是多轮 MoA，不是固定纠错矩阵上的有限跳传播 |

因此，Improve6 可以形成的合理原创性不是：

> “我们首次使用图选择专家。”

而是更窄的：

> “我们提出一种查询条件化的有向专家修复图。其边编码源域中平滑后的失败条件互补性，并通过有限深度传播，将当前专家分歧转化为候选回答的直接和间接修复证据。”

这个表述在目前检索结果下是可辩护的。

但必须写成：

> **To the best of our knowledge, we did not find prior work that combines these components in the same post-hoc response-selection formulation.**

而不是绝对声称：

> **This is the first expert repair graph.**

“未检索到完全同构方法”是文献检索结论，不是不存在相关未索引论文的数学证明。

---

# 五、Improve6 目前存在的几个关键技术问题

这些问题并不直接否定新颖性，但很可能成为审稿人的主要攻击点。

## 5.1 \(C\) 不是转移概率矩阵

每个 \(C_{ij}\) 可以落在 \(0\sim1\)，但一行中的多个专家可以同时回答正确，因此通常有：

\[
\sum_j C_{ij}>1.
\]

所以：

- \(C\) 不是 row-stochastic matrix；
- \(\omega C\) 也不是状态转移后的概率分布。

当前文档本身也明确说明：

- 一跳和两跳分数不是严格概率；
- DARE 和 RepairChain 的综合分数也不是校准正确率。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 574–589 行。

因此论文中不应把两跳解释成严格的：

\[
P(k\text{成功}\mid i\text{失败})
\]

或 Markov random walk。

更准确的术语是：

- **bounded-depth linear evidence propagation**；
- **directed rescue-score propagation**；
- **finite-hop repair evidence aggregation**。

---

## 5.2 “两跳纠错链”并不是真实的专家纠错链

文档明确指出：

- 所有回答在传播前已经生成；
- 两跳只是在统计矩阵上进行二次传播。

来源：`IMPROVE5_6_REAL_EXAMPLE(1).md` 第 466–472 行。

所以：

\[
A\rightarrow B\rightarrow C
\]

并不意味着：

1. A 先回答；
2. 检测 A 错误；
3. B 查看 A 后修复；
4. C 再查看 B 后修复。

它实际表示：

\[
\sum_i \widehat H_i^{(1)} C_{ij}
\]

的二阶关系聚合。

名称 **RepairChain** 容易让审稿人误认为存在：

- 顺序生成；
- 交互修改；
- 真正 cascade。

更准确的名称可以是：

- **Failure-Conditioned Repair Graph，FCRG**；
- **Directed Expert Rescue Graph，DERG**；
- **Repair Evidence Propagation，REP**；
- **Failure-Conditioned Complementarity Propagation，FCCP**。

---

## 5.3 需要明确是否屏蔽自环

按照当前公式，如果 \(i=j\)，在 \(i\) 失败的样本上 \(Y_{r,i}=0\)，因而：

\[
C_{ii}
=
\frac{4G_i}{N_i^{\mathrm{fail}}+4}>0.
\]

也就是说，除非代码显式执行：

\[
C_{ii}=0,
\]

否则专家会通过平滑先验获得“自我修复边”。

这不一定错误，但必须明确：

- 是否保留自环；
- 保留自环的统计解释；
- 去掉自环后结果是否下降；
- 两跳收益是否只是自环和强专家节点反复累积造成的。

---

## 5.4 两跳项可能主要反映“强专家中心性”

若某个专家 \(j\) 对很多失败专家都有较高 \(C_{ij}\)，它会拥有较高的加权入度：

\[
d_j^{\mathrm{in}}
=
\sum_i C_{ij}.
\]

那么 \(H^{(1)}_j\) 和 \(H^{(2)}_j\) 可能只是偏向：

- 全局较强专家；
- 对大多数专家失败集都有一定正确率的通用专家；
- 图中的高入度中心节点。

这并不必然说明二跳路径捕获到了新的组合性纠错结构。

因此至少要比较：

\[
C,\quad
D_{\mathrm{out}}^{-1}C,\quad
C D_{\mathrm{in}}^{-1},\quad
\operatorname{softmax}_{\mathrm{row}}(C)
\]

以及：

- 原始图；
- 随机打乱边权图；
- 保持节点度分布的置换图；
- 对称化图；
- 去掉对角线图。

只有真实有向图的两跳传播显著胜过这些 null models，才能说明收益确实来自“修复路径”。

---

## 5.5 固定权重存在事后调参风险

Improve5 和 Improve6 分别使用非常具体的权重：

\[
(0.44,0.28,0.24,0.04)
\]

和：

\[
(0.30,0.25,0.18,0.16,0.11).
\]

文档说明这些不是学习参数。

来源：

- `IMPROVE5_6_REAL_EXAMPLE(1).md` 第 285–291 行；
- `IMPROVE5_6_REAL_EXAMPLE(1).md` 第 503–509 行。

用户描述中已经说明：

- 先提出了多个改进；
- 再根据实验结果发现只有 Improve5 和 Improve6 有效。

这里存在明显的**方法筛选偏差**风险：

- 若多个方法和多组权重都在同一批 OOD 数据集上反复测试；
- 最后只保留表现好的 5 和 6；
- 再把同一批 OOD 结果作为泛化证据；

那么这些 OOD 数据已经部分成为方法选择集，而不再是严格独立测试集。

必须增加：

1. 方法和权重冻结后的全新 OOD 数据集；
2. 或 nested validation；
3. 或 leave-one-dataset-out 权重选择；
4. 最终完全 untouched 的测试集。

否则即使平均准确率提升，审稿人也可能认为是对 OOD benchmark 的间接过拟合。

---

# 六、当前材料还不足以验证 Improve5/6 的“泛化性提升”

上传的原始实验汇总充分表明，扩大模型和数据集后，原 Bench-CoE 在多数语言和多模态 OOD 设置下低于最佳单模型，只有少量设置取得很小增益。

来源：`ALL_EXPERIMENTS_SUMMARY(1).md` 第 258–312 行。

汇总结论也指出：

- 高路由类别准确率不等于最终系统增益；
- 专家互补性是主要瓶颈；
- 映射质量是主要瓶颈；
- query router 发生专家塌缩；
- 替换专家后性能明显下降。

来源：`ALL_EXPERIMENTS_SUMMARY(1).md` 第 368–376 行。

但目前上传的 Improve5/6 材料是一个：

> **MathVista testmini 第 85 题的真实案例**

该案例展示了：

- 最佳单模型 InternVL 回答错误；
- Improve5 选择 LFM 并修复；
- Improve6 选择 Qwen3-VL 并修复。

来源：

- `IMPROVE5_6_REAL_EXAMPLE(1).md` 第 1–27 行；
- `IMPROVE5_6_REAL_EXAMPLE(1).md` 第 820–837 行。

这个案例适合用于论文图示和解释，但不能证明总体泛化性。

目前上传材料中没有看到以下完整结果：

- Improve5/6 在每个 ID 和 OOD 数据集上的总准确率；
- 相对最佳单模型的逐数据集增益；
- 相对 majority vote、MCB、KNOP、Smoothie 等基线的增益；
- 参数敏感性；
- 多随机种子或置信区间；
- 不同专家池替换后的稳定性；
- 推理成本和延迟。

因此本次能够给出的结论是：

> **机制层面的 prior-art 审计已经可以完成；但“Improve5/6 确实稳定改善泛化”的经验结论，尚不能仅从当前上传材料独立核验。**

---

# 七、正式投稿前必须补充的基线

## 7.1 Improve5 必须面对的经典基线

至少应实现或适配：

| 基线 | 作用 |
|---|---|
| Global Best | 始终选择源域全局最强专家 |
| Majority Vote / Answer Support | 只按当前答案支持率选择 |
| OLA / LCA | 基于 Query 特征邻域的局部准确率选择 |
| MCB-DCS | KNN 后根据相似多分类器行为计算局部能力 |
| KNOP | 直接在输出画像空间检索邻居 |
| KNORA-U / KNORA-E | 根据邻域 oracle 表现动态选集成 |
| META-DES | 学习多个能力指标判断专家是否适合 |

若 Improve5 不能显著超过 MCB/KNOP，则应明确将其称为：

> **robust output-profile dynamic selection baseline**

而不是独立原创算法。

---

## 7.2 需要加入的 LLM/VLM 回答选择基线

至少包括：

| 基线 | 对应 Improve5/6 的部分 |
|---|---|
| MORE-style selector | 专家回答、置信度、一致性 |
| Smoothie-GLOBAL | 全局输出质量 |
| Smoothie-LOCAL | 输出相关的局部样本质量 |
| Uncertainty-only selector | 仅利用输出置信度/不确定性 |
| Agreement × Global Accuracy | 当前答案支持率和历史性能 |
| Local KNN only | 只使用 \(L_m(q)\) |
| Global + Local rank | 检验混合全局与局部能力是否已足够 |
| Learned logistic/MLP selector | 检验固定权重是否优于简单学习器 |

MORE 已经直接使用：

- 所有专家回答；
- confidence；
- agreement；

训练候选答案选择器。

参考：

- [MORE, Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.552.pdf)

Smoothie 则利用：

- 输出嵌入；
- 邻域；
- 样本相关质量评分；

进行选择。

参考：

- [Smoothie, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6b57a990462df5afa58d64ce2709db9-Paper-Conference.pdf)

这两项尤其不能缺失。

---

## 7.3 Improve6 的决定性消融

建议至少报告：

| 版本 | 要回答的问题 |
|---|---|
| \(G\) only | 全局最佳专家能达到多少 |
| \(A\) only | 答案多数是否已足够 |
| \(L\) only | Improve5 核心局部能力贡献 |
| \(C\) column mean only | 图的静态中心性是否已足够 |
| \(H^{(1)}\) only | 一跳条件修复是否有效 |
| \(H^{(2)}\) only | 二跳是否独立有用 |
| \(H^{(1)}+H^{(2)}\) | 二跳是否提供增量 |
| No \(A/U\) | Query 相关失败信号是否必要 |
| No \(L\) | 图方法是否独立于 output-profile KNN |
| No \(G\) | 是否只是偏向全局强专家 |
| \(C_{ii}=0\) | 自环的影响 |
| Row-normalized \(C\) | 是否需要概率化传播 |
| Symmetric \(C\) | 有向性是否真正必要 |
| Random/permuted \(C\) | 图结构是否包含真实信息 |
| 1–5 hops | 两跳是否优于一跳且不会过平滑 |
| Learned weights | 固定权重是否真的优于验证集学习 |

真正能支持 Improve6 新颖性的结果应当是：

\[
\text{Full Repair Graph}
>
\text{Local KNN}
>
\text{Global Best},
\]

同时满足：

\[
\text{真实有向 }C
>
\text{随机图/对称图/静态中心性},
\]

并且：

\[
H^{(1)}+H^{(2)}
>
H^{(1)}
\]

在多个 OOD 数据集上稳定成立。

否则两跳项很可能只是一个没有独立贡献的分数平滑器。

---

# 八、可以主张和不应主张的创新点

| 可能的论文表述 | 判断 |
|---|---|
| 首次使用相似历史样本估计专家局部能力 | **不能主张** |
| 首次基于多专家输出状态进行 KNN 专家选择 | **不能主张** |
| 首次结合局部与全局专家表现 | **不能主张** |
| 首次利用专家答案一致性选择最终回答 | **不能主张** |
| 首次使用不确定性进行多 LLM 回答选择 | **不能主张** |
| 首次构建图结构进行 LLM 路由 | **不能主张** |
| 首次建模专家成对错误或失败条件互补性 | **不能主张** |
| 将 Output-Profile Dynamic Selection 系统化迁移到跨 benchmark LLM/VLM 专家池 | **可以作为应用/适配贡献** |
| 将跨划分稳定性加入 output-profile 局部能力估计 | **可以作为较弱的组件贡献** |
| 构造有向的 \(P(j\text{成功}\mid i\text{失败})\) 专家修复图 | **可以主张具体构造，但不能声称条件互补思想本身全新** |
| 由当前回答分歧形成失败质量，并在修复图上执行有限跳传播 | **目前最有希望的创新点** |
| 将直接和间接修复证据用于跨 benchmark 的后验候选回答选择 | **可以作为完整任务表述中的组合创新** |

---

# 九、命名也需要调整

## 9.1 Improve5 的 “DARE”

已有一篇 ICLR 2026 OpenReview 投稿使用了：

> **DARE: Difficulty-Aware Dynamic Routing for Mixture of Experts**

该稿件后来显示为 withdrawn submission，但在同一 expert-routing 领域中已经造成明显名称和缩写冲突。

参考：

- [OpenReview DARE 检索结果](https://openreview.net/search?content=authors&group=all&sort=cdate%3Adesc&source=forum&term=~Chaohu_Liu1)

所以不建议继续使用 DARE。可改为：

- **RADS：Reliability-Adjusted Dynamic Selection**；
- **OPRS：Output-Profile Reliability Selection**；
- **Robust-OPDS：Robust Output-Profile Dynamic Selection**。

## 9.2 Improve6 的 “RepairChain”

由于当前不存在：

- 顺序模型调用；
- 真实链式修复；

“Chain” 容易误导。

更建议：

- **FCRG：Failure-Conditioned Repair Graph**；
- **DERG：Directed Expert Rescue Graph**；
- **FCRP：Failure-Conditioned Repair Propagation**。

其中 **FCRG** 最准确地反映方法本质。

---

# 十、最终投稿策略建议

最稳妥的论文结构不是把 Improve5 和 Improve6 都描述成两个全新路由器，而是：

## 10.1 将 Improve5 降级为强基线

将其明确定位为：

> 面向异构 LLM/VLM 专家池的、经过跨划分可靠性和不确定性增强的 output-profile dynamic selection。

主动承认其继承 MCB/KNOP 思想，反而可以避免被审稿人指出遗漏经典文献。

## 10.2 将 Improve6 作为主方法

将核心贡献收敛为：

> **Failure-Conditioned Directed Repair Graph for Post-hoc Expert Response Selection**

主张的重点是：

- 边是平滑的失败条件救援率，而不是一般相关性；
- 当前 Query 的回答分歧产生节点失败质量；
- 有限跳传播区分直接和间接修复证据；
- 源域纠错结构能够迁移到未见目标 benchmark；
- 不需要在目标域使用标签。

## 10.3 不再把完整 Improve5/6 称为低成本单专家路由

建议把系统分成两个工作模式：

\[
\text{Fast Bench-CoE}
\]

和：

\[
\text{Full Repair-Graph Selection}.
\]

更进一步，可以设计：

\[
\text{单专家快速路径}
\xrightarrow{\text{低置信度触发}}
\text{少量专家救援路径}.
\]

这会同时保留：

- 原 Bench-CoE 的单专家成本优势；
- Improve6 的后验纠错能力；
- 更清楚的 accuracy–cost trade-off；
- 与纯多专家全调用方法的差异。

---

# 十一、最终学术判断

## Improve5

> 核心思路已经被经典动态分类器选择和 output-profile 方法覆盖；精确加权公式可能未出现，但不足以构成强原创性。应作为适配方法、基线或 Improve6 的组成部分。

## Improve6

> 条件错误互补、agreement、uncertainty、局部能力和图建模都存在 prior art；但截至 2026年8月8日，尚未发现与“查询失败质量驱动的有向救援图一跳/两跳传播式后验回答选择”完全同构的已发表方法。该组合具有中等、可辩护的创新潜力，但必须通过 MCB/KNOP/MORE/Smoothie、图随机化、方向性、跳数和固定权重等实验排除已有机制与简单启发式解释。

因此，当前最合理的论文决策是：

> **保留 Improve5 作为经典动态选择基线，把 Improve6 重构为论文唯一主方法，并将论文从“单次专家路由”重新定位为“基于 benchmark 失败互补结构的跨域专家回答修复选择”。**

---

# 十二、主要参考文献与链接

1. Giacinto, G. & Roli, F. Dynamic Classifier Selection Based on Multiple Classifier Behaviour.  
   https://www.sciencedirect.com/science/article/abs/pii/S0031320300001503

2. K-Nearest Output Profiles / Dynamic Ensemble Selection.  
   https://www.sciencedirect.com/science/article/abs/pii/S0031320312001124

3. MORE: Multi-Model Answer Selection. Findings of EMNLP 2023.  
   https://aclanthology.org/2023.findings-emnlp.552.pdf

4. Smoothie: Label-Free Language Model Routing and Selection. NeurIPS 2024.  
   https://proceedings.neurips.cc/paper_files/paper/2024/file/e6b57a990462df5afa58d64ce2709db9-Paper-Conference.pdf

5. Uncertainty-Aware Answer Selection. Findings of EMNLP 2025.  
   https://aclanthology.org/2025.findings-emnlp.1367.pdf

6. Ensemble Pruning via Semi-definite Programming. JMLR.  
   https://www.biz.uiowa.edu/faculty/nstreet/research/sdpprune_jmlr_final.pdf

7. Probabilistic Classifier Chains for Dynamic Ensemble Selection.  
   https://ecmlpkdd2017.ijs.si/papers/paperID595.pdf

8. GNN-DES.  
   https://dl.acm.org/doi/10.1007/978-3-031-42795-4_6

9. GraphRouter.  
   https://arxiv.org/pdf/2410.03834

10. RouteMoA. ACL 2026.  
    https://aclanthology.org/2026.acl-long.558.pdf

11. DARE: Difficulty-Aware Dynamic Routing for Mixture of Experts，OpenReview 检索。  
    https://openreview.net/search?content=authors&group=all&sort=cdate%3Adesc&source=forum&term=~Chaohu_Liu1
