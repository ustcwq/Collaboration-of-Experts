我仔细分析了 Improve1–4 的实验结果后，认为**下一步不能继续沿着“设计更好的 router”走**。原因非常清楚：

- Improve1 证明：**目标标签 probe 能释放巨大互补性，但不可用**；
- Improve2/3 证明：**仅从 query 表征、subject、latent capability 很难跨 benchmark 转移**；
- Improve4 证明：**真正有效的信息来自目标样本上的专家行为（expert outputs），而不是输入本身**。尤其 ECC decoder 在 BBH 达到 71.34%，超过最强单模型 68.05%；GPQA 也达到 32.84%，超过单模型 32.17%。

因此新的核心问题应该重新定义：

> **如何在不访问目标标签的情况下，从专家无标签行为中恢复类似 oracle 的互补选择能力？**

也就是说：

过去：

[
q\rightarrow expert
]

失败。

现在应该：

[
{expert\ outputs}
\rightarrow
latent\ failure\ state
\rightarrow
expert\ correction
]

下面提出一些基于 Improve4 结果进一步推导出的、更原创的方向。

------

# 方向1（最推荐）

# FATE-CoE: Failure-Aware Test-time Expert Ecology

## 核心思想

Improve4 的 ECC decoder 已经证明：

> 专家输出之间的关系包含大量目标任务信息。

但是 ECC 仍然是：

```
答案模式
↓
选择专家
```

进一步升级：

不是学习“哪个专家正确”。

而是学习：

> 当前样本处于哪一种专家生态状态。

------

## 核心假设

对于一个 query：

14 个专家输出：

[
O(q)=
(o_1,o_2,...,o_M)
]

形成一个状态：

[
z_q
]

不同状态对应不同 expert policy。

例如：

### State A:

多数专家一致

```
A B C D:
same answer
```

说明：

低风险。

策略：

选择最快专家。

------

### State B:

两个答案簇

```
A B C:
answer1

D E F:
answer2
```

说明：

存在能力分叉。

策略：

选择历史中最擅长该冲突模式的专家。

------

### State C:

全部分散

说明：

hard sample。

策略：

选择 reasoning expert。

------

# 方法

建立：

Expert Output State Space。

训练阶段：

source benchmark：

获得：

[
X_s=
{
outputs,
correctness
}
]

但是：

target correctness 不用。

------

## Step 1

无监督聚类：

输入：

专家输出特征：

- answer agreement
- semantic embedding
- length
- reasoning length
- uncertainty words
- option distribution

得到：

failure states:

[
C_1,C_2,...,C_K
]

------

## Step 2

源域建立：

state → expert success

例如：

[
P(expert|state)
]

------

## Step 3

目标：

只计算：

state。

然后选择：

[
argmax P(expert|state)
]

------

# 为什么比 ECC 更原创？

ECC：

> 当前答案分布像什么？

FATE：

> 当前样本属于哪种专家失效生态？

区别：

ECC 是 decoder。

FATE 是：

**failure state modeling**。

------

# 与已有工作区别

最接近：

- self-consistency
- mixture voting
- uncertainty routing

但它们：

关注：

答案正确概率。

FATE：

关注：

专家错误结构。

目前 CoE 很少建模：

expert failure ecology。

------

# 实验

无需重新推理。

你已有：

expert outputs。

新增：

```python
build_failure_state.py
state_router.py
```

即可。

------

# 预期

我认为最可能：

BBH:

71.34%

有机会：

72–74%。

GPQA:

32.84%

可能进一步提升。

------

# 方向2

# ECR-CoE: Expert Correction Relationship Learning

## 核心思想

Improve4 只考虑：

谁可能正确。

但是忽略：

专家之间存在“修复关系”。

例如：

专家A错误时：

专家B经常正确。

定义：

Correction Graph。

------

## 构造

源域：

对于每个 query：

如果：

A wrong

B correct

建立：

[
A\rightarrow B
]

权重：

[
w_{AB}
]

形成：

专家修复图。

例如：

```
Mathstral
   |
   | repair
   ↓
DeepSeek

Qwen
   |
   ↓
Gemma
```

------

## 推理

目标：

如果发现：

专家输出模式：

类似 A 的失败模式。

不要问：

谁最好。

问：

谁经常修复 A。

------

## 为什么原创？

当前：

expert ranking。

这里：

expert interaction。

类似：

fault diagnosis 中：

component dependency。

------

## 优势

特别适合：

GPQA。

因为：

科学推理中：

一个模型错误不是随机。

------

# 方向3

# DARE-CoE: Domain-Agnostic Reliability Estimation

## 核心思想

当前最大问题：

source ranking ≠ target ranking。

但是：

专家可靠性变化不是随机。

存在：

domain-independent reliability。

------

定义：

专家：

[
R_m
]

由：

- answer stability
- reasoning consistency
- output entropy
- contradiction rate

构成。

------

例如：

某专家：

source:

数学强。

但是：

输出经常：

自相矛盾。

OOD:

容易失败。

------

## 推理

选择：

[
score=
local\ compatibility
\times
reliability
]

------

区别：

不是 calibration。

Calibration:

预测概率。

DARE:

专家行为稳定性。

------

# 方向4

# Oracle Gap Minimization via Conservative Routing

这是我认为理论价值最高。

## 当前问题

oracle:

BBH:

94.87%

best single:

68.05%

差：

26.82%

说明：

大量机会。

但是错误路由可能伤害。

------

提出：

不是追求最大提升。

而是：

保证不输。

定义：

conservative utility:

# [ U_m

## E(score_m)

\lambda Risk_m
]

其中：

Risk:

专家失效概率。

------

## 新贡献

证明：

在一定条件：

[
E[U]
\geq
BestSingle-\epsilon
]

即：

不会明显低于 strongest expert。

------

## 为什么重要？

实际 CoE 最大问题：

如果没有收益，为什么不用最大模型？

这个方向回答：

安全部署。

------

# 方向5

# Sparse ECC Routing

## 针对 Improve4 最大缺点：

需要全部专家。

成本高。

------

## 思想

ECC decoder：

14 experts。

但是：

是否真的需要14？

分析：

expert output correlation。

选择：

最小纠错码。

------

例如：

14个专家：

实际只需要：

```
A B C D E
```

因为：

覆盖不同错误模式。

------

## 方法

源域：

构建：

expert diversity matrix。

优化：

[
min |S|
]

subject:

[
Oracle(S)>threshold
]

------

得到：

专家子集。

------

## 优势

解决：

Improve4 部署问题。

------

# 方向6

# Teacher-Free Distilled Output Router

解决：

Improve4 需要调用所有专家。

------

## 两阶段：

### Offline

source:

完整 ECC router。

得到：

pseudo labels:

[
expert^*
]

------

### Online

训练：

question-only router。

输入：

只：

query/image。

预测：

ECC router。

------

关键：

不是预测 correctness。

预测：

**output router decision**。

------

## 为什么比普通蒸馏新？

普通：

teacher answer distillation。

这里：

teacher 是：

multi-expert failure decoder。

------

# 方向7

# Cross-Benchmark Expert Transfer Graph

针对：

MMLU→BBH/GPQA。

------

## 核心

不同 benchmark：

不是独立。

建立：

benchmark graph。

节点：

benchmark。

边：

能力迁移。

例如：

MMLU:

knowledge reasoning

GPQA:

expert reasoning

------

学习：

benchmark-to-benchmark transfer matrix。

------

推理：

目标 benchmark：

虽然没有标签。

但知道：

它类似哪些 source benchmark。

然后调整 expert prior。

------

# 方向8

# Output Semantic Contradiction Routing

这个方向很容易实现。

## 当前 ECC：

看答案。

但是：

忽略答案内容。

例如：

两个专家：

选择同一个 option。

但是：

解释完全冲突。

------

加入：

semantic contradiction。

计算：

专家输出：

[
sim(o_i,o_j)
]

构造：

contradiction graph。

------

发现：

高风险区域。

选择：

历史中：

该区域最可靠专家。

------

# 综合排序

结合你已有结果：

| 方案                  | 创新  | 成功概率 | 实现 |
| --------------------- | ----- | -------- | ---- |
| FATE failure ecology  | ★★★★★ | ★★★★★    | 容易 |
| ECR correction graph  | ★★★★★ | ★★★★☆    | 容易 |
| Sparse ECC            | ★★★★☆ | ★★★★★    | 容易 |
| Teacher-free distill  | ★★★★  | ★★★★★    | 容易 |
| Conservative routing  | ★★★★★ | ★★★★     | 中   |
| DARE reliability      | ★★★★  | ★★★★     | 容易 |
| Contradiction routing | ★★★★  | ★★★      | 容易 |
| Benchmark graph       | ★★★★★ | ★★★      | 中   |

------

# 我认为下一篇论文最强故事

不是：

> Better Bench-CoE Router

而是：

# **Failure Ecology Guided Expert Collaboration**

核心框架：

```
Source Benchmark
       |
       v

Expert Behavior Matrix
       |
       +----------------+
       |                |
 Failure Ecology   Correction Graph
       |                |
       +----------------+
              |
              v

Target Unlabeled Expert Outputs
              |
              v

Failure State Recognition
              |
              v

Robust Expert Selection
```

贡献：

## 1

首次从 expert success 转向 expert failure ecology。

## 2

无需目标标签，仅利用目标无标签专家行为。

## 3

将 output-level oracle signal 压缩为低成本 sparse router。

------

# 最建议立即验证顺序

## 第一优先级（1天）

实现：

### FATE-CoE

因为：

已有 Improve4 数据。

不用重新推理。

比较：

```
ECC
FAME
FATE
Best single
```

------

## 第二优先级（1天）

实现：

### Sparse ECC

回答：

“是否必须调用所有专家？”

------

## 第三优先级（3-5天）

实现：

### Teacher-free output distillation

把 Improve4 从：

offline oracle-like

变成：

online practical CoE。

------

根据目前结果，我认为**真正可能超过 ECC 71.34% 的不是再设计一个 router，而是学习 ECC 背后的“专家失效结构”。** Improve4 已经证明方向正确，下一步应该从“答案纠错”升级到“失效生态建模”。