我认真分析了 Improve1–5 的完整实验链路后，认为现在研究状态已经发生了一个**非常重要的转折**。

之前的问题：

> “如何设计一个更好的 router？”

已经被证明不是正确的问题。

因为 Improve1–3 已经说明：

- query feature 不够；
- subject/capability 不够；
- source-domain transfer 不够。

真正突破来自 Improve4/5：

- Improve4：**利用目标无标签 expert outputs 有效**
- Improve5：**建模 failure ecology 有效**

尤其：

- FATE failure ecology：
  - BBH：77.85%
  - 最强单模型：68.05%
  - **+9.80%**
- ECR correction graph：
  - BBH：76.18%
  - GPQA：32.97%
- semantic contradiction：
  - GPQA：33.01%

说明：

> 专家之间的错误结构、修复关系、输出生态，比输入问题本身包含更多 OOD 可迁移信息。



因此下一阶段不能继续做：

```
question → router → expert
```

而应该进入：

```
expert ecosystem → latent state → adaptive collaboration
```

下面提出基于 Improve5 结果进一步推导的原创方向。

------

# 一、最高优先级方向

# 1. Meta-FATE:

# Learning When to Trust Which Failure Ecology

## 核心发现

Improve5 最大的问题：

不同目标集最佳策略不同：

| Dataset | 最优                          |
| ------- | ----------------------------- |
| BBH     | FATE                          |
| GPQA    | ECR / DARE                    |
| MMStar  | benchmark transfer / semantic |

说明：

不是某一个 router 永远最好。

真正的问题：

> 如何在没有目标标签时，判断当前 benchmark/sample 应该相信哪一种 expert failure model？

------

## 新方法

学习：

[
P(strategy|target\ distribution)
]

策略集合：

[
S=
{
FATE,
ECR,
DARE,
ECC,
Semantic
}
]

------

## 输入（无标签）

目标：

只使用：

### 输入分布

- embedding distribution
- question length
- category names
- modality

### 输出生态

- answer entropy
- disagreement entropy
- semantic conflict
- output length variance

形成：

[
z_{target}
]

------

## 输出：

选择：

# [ s^*

argmax_s
Utility(s,z)
]

------

## 为什么原创？

已有：

AutoML selection

但是：

这里不是选模型。

是：

**选择 expert collaboration mechanism。**

也就是：

meta-routing。

------

## 和已有区别

不是：

```
query → expert
```

而是：

```
benchmark/sample ecology
        ↓
routing algorithm
        ↓
expert
```

这是更高层 routing。

------

## 实现成本

非常低。

因为：

Improve5 已经有所有策略。

只需要：

source benchmark LOBO:

例如：

训练：

MMLU

验证：

BBH-like

学习：

什么时候：

FATE > ECC

------

## 预期

很可能：

BBH:

保持 FATE 77.85

GPQA:

超过 33.01

MMStar:

避免失败。

------

# 2. Neuro-ECC:

# Neural Error-Correcting Expert Codes

这是我认为最有论文价值的方向。

------

## Improve5 的 ECC

现在：

人工设计：

expert answer code。

例如：

```
A B C D
1 1 0 1
```

然后 decode。

问题：

code 结构固定。

------

## 新思想

学习：

专家纠错码。

类似：

神经通信。

------

## 定义

专家输出：

[
O=
[o_1,o_2,...o_M]
]

学习：

latent code:

[
z=f(O)
]

然后：

decoder:

[
g(z)\rightarrow expert
]

------

## 关键创新：

不是：

哪个 expert 正确。

而是：

学习：

专家错误模式空间。

------

## 训练：

source benchmark：

得到：

```
output pattern

+
correctness
```

训练：

autoencoder:

encoder:

expert behavior

decoder:

oracle expert

------

## 测试：

target:

无标签：

expert outputs

encoder:

得到 latent state

decoder:

选择专家。

------

## 为什么比 FATE 新？

FATE：

cluster + lookup

Neuro-ECC：

learned nonlinear error code。

------

## 风险

需要更多 source benchmark。

但是你已有：

GAOKAO

MMLU

BBH

CMMMU

------

# 3. RepairChain-CoE:

# Expert Repair Chain Discovery

## 核心发现

ECR 已经证明：

A 错 → B 修复。

但是现在只是：

pairwise graph。

------

## 升级：

学习：

repair chain。

例如：

```
Qwen
 |
失败
 |
Mathstral
 |
失败
 |
DeepSeek
 |
成功
```

------

## 建图：

节点：

expert

边：

[
A\rightarrow B
]

不仅：

correct

加入：

failure type。

------

## 推理：

目标：

识别失败状态：

然后执行：

repair chain。

------

## 为什么原创？

已有：

cascade routing。

但是 cascade：

固定。

这里：

adaptive repair path。

------

# 4. OASIS-CoE:

# Oracle Approximation via Structural Invariant Selection

这个方向解决：

为什么 oracle 高，但是 router 低。

------

## 核心思想

Oracle:

每题知道答案。

现实：

不知道。

但是：

oracle 选择依赖：

隐藏结构。

------

学习：

oracle decision boundary。

不是学习：

correctness。

而学习：

oracle 与 best single 的差异区域。

------

## 定义：

源域：

# [ \Delta(q,m)

Y_m-Y_{best}
]

找到：

positive repair regions。

------

目标：

无标签：

预测：

query 是否属于 repair region。

------

## 区别

不是：

预测 expert。

预测：

“是否值得离开强专家”。

------

## 这个特别适合解决：

你的 conservative routing。

------

# 5. Multi-View Failure Signature

Improve5 当前 failure signature 仍然简单：

- answer cluster
- output length
- uncertainty

进一步增强：

构造：

专家失败多视角表示。

------

## View 1: Answer topology

已有。

------

## View 2: Reasoning trajectory

例如：

分析：

```
because
therefore
however
I am not sure
```

构造：

reasoning uncertainty。

------

## View 3: Semantic contradiction

已有。

------

## View 4: Agreement evolution

不是最终答案。

而是：

生成过程。

如果保存：

token log:

可以分析：

专家什么时候分叉。

------

## View 5: Abstention behavior

例如：

```
cannot determine
likely
maybe
```

------

形成：

# [ F(q)

[F_a,F_r,F_s,F_u,F_c]
]

然后：

failure state。

------

# 6. Sparse-ECC++:

# Adaptive Expert Subset Discovery

Improve5：

Sparse ECC：

k=8 有效。

但 k=4/6失败。

说明：

简单贪心不够。

------

## 新思想

选择专家不是：

最大覆盖。

而是：

纠错能力最大。

------

定义：

专家子集：

[
S
]

优化：

## [ \max Coverage(S) + Correction(S)

Cost(S)
]

------

## 新贡献

从：

expert selection

升级：

error-correcting subset design。

------

## 实验

非常容易：

已有 ECC。

------

# 7. FATE Distillation++:

# Distill Failure State, Not Expert ID

Improve5 teacher-free ECC失败：

64.2%。

原因：

蒸馏目标错了。

------

当前：

teacher:

```
choose expert A
```

太离散。

------

改：

蒸馏：

中间状态：

[
failure\ state
]

学生预测：

```
failure ecology
↓
expert
```

------

结构：

Teacher:

full output router

Student:

question-only

loss:

[
L=
L_{state}
+
L_{expert}
+
L_{risk}
]

------

## 预计

比直接 expert-id distillation 高。

------

# 8. MMStar 专项：

# Visual Failure Ecology

这是必须单独处理的。

因为：

MMStar：

所有文本方法失败。

说明：

不是 router 问题。

而是：

failure signature 缺少视觉。

------

## 新方法：

V-FATE。

输入：

专家：

image-text output

增加：

视觉错误模式。

例如：

- OCR disagreement
- object grounding disagreement
- spatial relation disagreement
- visual evidence missing

------

构造：

visual failure state。

------

## 为什么可能突破 MMStar

MMStar：

视觉理解为主。

文本生态不足。

------

# 二、我认为最有论文价值的组合

如果目标 FCS，我不会继续增加十个小方法。

建议形成：

# **FATE++: Failure Ecology Guided Adaptive Expert Collaboration**

三个模块：

```
Expert Outputs
       |
       v

Multi-view Failure Signature
       |
       v

Neural Failure Ecology Space
       |
       +----------------+
       |                |
 Strategy Selector   Repair Graph
       |                |
       +----------------+
              |
              v

Adaptive Expert Collaboration
```

------

# 三、和已有论文的差异

| 已有方法         | 你的新方法                                     |
| ---------------- | ---------------------------------------------- |
| RouteLLM         | query→expert                                   |
| LLM-Bench        | benchmark label routing                        |
| MODULAR-CoE      | category→expert                                |
| ContextualRouter | similar query                                  |
| ECC Improve5     | fixed answer correction                        |
| FATE++           | learn failure ecology and collaboration policy |

核心创新：

> Existing routers model expert preference. We model expert failure dynamics.

这个角度目前明显更少。

------

# 四、建议下一轮实验顺序

## 第一优先级（1–2天）

### Meta-FATE

原因：

最快。

利用已有：

FATE/ECC/ECR/DARE结果。

只学习：

选择策略。

目标：

三套 benchmark 全超过 best single。

------

## 第二优先级（3–5天）

### Neuro-ECC

这是最像论文核心创新。

------

## 第三优先级

### FATE Distillation++

解决部署问题。

------

## 第四优先级

### V-FATE

解决 MMStar。

------

# 五、最终论文故事建议

不要写：

> We propose a better router.

应该写：

> We reveal that OOD expert collaboration is governed by latent failure ecology rather than task semantics. We propose a failure-aware collaboration framework that discovers, models, and exploits expert failure dynamics without target labels.

这比当前 Bench-CoE 高一个层次。

------

综合 Improve1–5，我认为下一步最值得投入的不是继续设计 routing score，而是：

1. **Meta-FATE（自动选择协作策略）**
2. **Neural ECC（学习专家纠错结构）**
3. **FATE Distillation（把 output-level oracle 转成低成本 router）**
4. **Visual Failure Ecology（解决 MMStar）**

其中第 1 个最容易快速验证，第 2 个最有原创论文潜力。