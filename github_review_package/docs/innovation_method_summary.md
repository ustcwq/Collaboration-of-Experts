# Bench-CoE 改进方法总结

下面按“方法递进关系”总结。核心思路都是：不重新训练大模型，只利用目标测试集的一小部分 probe 样本和已有单模型缓存结果，学习更好的专家选择规则，再在 heldout 部分评估。

## 1. 学科/分组 Probe Mapping

方法：按数据集元信息分组，例如 `category`、`task`、`subdomain`、`skills`。每组抽少量 probe 样本，看哪个专家在该组 probe 上准确率最高，然后该组 heldout 样本都路由到这个专家。

例子：

- GPQA 有 `subdomain=Organic Chemistry`
- probe 里该子领域 20 道题
- `MAmmoTH2-8B-Plus` 对 8 道，`Qwen3.5-9B` 对 6 道，`granite` 对 5 道
- heldout 中所有 `Organic Chemistry` 样本路由到 `MAmmoTH2-8B-Plus`

优点：简单、可解释。

缺点：分组太粗时不够灵活，probe 太小时容易过拟合。

## 2. Risk-Controlled Mapping / Fallback

方法：仍按分组选专家，但不是“谁 probe 最高就选谁”，而是和全局最强专家比较。只有当某组专家的置信下界明显超过全局专家时才切换，否则回退到全局专家。

例子：

- 全局最强专家是 `Qwen3.5-9B`
- 某个 `task=logical_deduction` 上，`granite` probe 准确率略高
- 如果 Wilson 下界没有超过 `Qwen3.5-9B`，不切换
- 如果超过，则该任务路由到 `granite`

优点：减少因为 probe 偶然波动导致的错误切换。

缺点：仍依赖人工/数据集给出的分组。

## 3. TF-IDF + Metadata 多标签正确性预测

方法：对每个专家训练一个二分类器，输入是题目文本 + 元信息，输出“该专家是否会答对”。推理时选预测正确概率最高的专家。

输入特征示例：

```text
meta_task=boolean_expressions
meta_routed_subject=数学
Question: Evaluate the result of ...
```

训练目标示例：

```text
Qwen3.5-9B: 1
granite-3.3-8b-instruct: 0
MAmmoTH2-8B-Plus: 1
...
```

推理时：

```text
P(Qwen3.5-9B correct)=0.62
P(granite correct)=0.71
P(MAmmoTH correct)=0.58
```

选择 `granite`。

优点：能利用题面和元数据，不局限于固定学科映射。

缺点：本质是轻量 correctness predictor，已有相关路由思想，原创性不够强。

## 4. kNN-TFIDF 局部相似题路由

方法：对 heldout 题目，找 probe 中最相似的 k 道题，看各专家在这些近邻上的局部表现，选择局部表现最好的专家。

例子：

当前题：

```text
A physics problem about electric potential and field direction.
```

在 probe 中找到最近的 10 道物理/电磁题：

```text
Qwen3.5-9B: 6/10
MAmmoTH2-8B-Plus: 4/10
granite: 7/10
```

则路由到 `granite`。

优点：比按学科更细。

缺点：普通 kNN 路由本身不算原创，容易被认为是已有检索路由思想。

## 5. Complementarity-Weighted Probe Routing

这是第一版更有价值的改进。

方法：先选一个默认强专家，例如全局最强的 `Qwen3.5-9B`。然后不直接问“哪个专家局部准确率最高”，而是问“切换到某专家相对默认专家是否有净收益”。

定义配对增益：

```text
delta = +1       候选专家答对，默认专家答错
delta = -lambda  候选专家答错，默认专家答对
delta = 0        二者同对或同错
```

对当前样本找相似 probe 题，计算候选专家的局部加权平均 `delta`。只有平均 `delta > 0` 时才切换。

例子：

默认专家：`Qwen3.5-9B`

候选专家：`MAmmoTH2-8B-Plus`

当前题：GPQA 化学题

最近 5 个 probe 样本中：

| 样本 | Qwen3.5 | MAmmoTH | delta |
| --- | --- | --- | ---: |
| 1 | 错 | 对 | +1 |
| 2 | 错 | 对 | +1 |
| 3 | 对 | 错 | -1 |
| 4 | 错 | 错 | 0 |
| 5 | 对 | 对 | 0 |

平均 delta：

```text
(+1 + 1 - 1 + 0 + 0) / 5 = 0.2
```

因为大于 0，所以切换到 `MAmmoTH2-8B-Plus`。

优点：关注“互补错误”，不是简单追求局部准确率。

实验上提升明显，例如 `gaokao_bert_gpqa` 达到 43.20%，比 heldout 最强单模型高 +9.01%。

## 6. Paired Local LCB Complementarity Routing

这是最后一版、最适合主打的改进。

它在方法 5 基础上加入统计风险门控：不仅要求局部平均 `delta > 0`，还要求它的下置信界 LCB > 0。

公式：

```text
LCB = mean(delta) - z * sqrt(var(delta) / n_eff)
```

只有：

```text
LCB > 0
```

才切换专家，否则保留默认强专家。

例子：

当前题：MathVista 几何视觉题

默认专家：`Qwen3-VL-4B-Instruct`

候选专家：某个在相似几何题上更互补的模型

近邻 probe 的 delta：

```text
+1, +1, +1, -1.25, 0
```

平均值：

```text
mean = 0.35
```

如果波动较大，LCB 可能是：

```text
LCB = 0.35 - 0.25 * sqrt(var / n_eff) = 0.12
```

因为 `0.12 > 0`，允许切换。

如果另一个专家：

```text
mean = 0.20
LCB = -0.03
```

虽然平均看起来有收益，但下置信界小于 0，不切换。

优点：

- 显式控制“切换导致退化”的风险。
- 用的是配对反事实收益，不只是预测专家正确率。
- 比普通 kNN routing / correctness prediction 更有论文贡献点。

最终它刷新了几个结果：

- `gaokao_bert_mmlu_pro`: 66.86%，比最强单模型 +1.04%
- `gaokao_bert_bbh`: 74.19%，比最强单模型 +5.97%
- `qwen3vl_mathvista`: 67.17%，比最强单模型 +1.50%

最终结果文件：

```text
outputs/bench_coe/innovation_experiments_paired_lcb_combined_filtered/materialized/materialized_summary.md
```

## 论文主线建议

建议论文主线写成：

```text
Probe-Adaptive Bench-CoE with Paired Local Lower-Confidence Complementarity Gating
```

不要主打“kNN 路由”或“正确性预测”，这些已有相近工作；主打“基于目标 probe 的配对反事实互补收益 + 局部下置信界风险门控”。
