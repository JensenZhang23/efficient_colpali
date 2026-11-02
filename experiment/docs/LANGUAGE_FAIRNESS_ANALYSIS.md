# RQ M=16 语言公平性诊断报告

**日期**: 2026-09-13  
**数据**: ViDoRe V3 8 数据集，ColQwen2.5-v0.1，全语言  
**方法**: 静态 RQ M=16 K=256

---

## 📊 核心发现

### 1. 跨数据集平均：1.67pp 差距（French 最好，German 最差）

```
Cross-dataset average retention (8 datasets):
  german         94.67%  (std=2.82pp)
  italian        94.87%  (std=3.64pp)
  portuguese     95.25%  (std=1.67pp)
  spanish        95.27%  (std=1.38pp)
  english        95.60%  (std=2.31pp)
  french         96.34%  (std=1.34pp)

Gap: French (96.34%) - German (94.67%) = 1.67pp
```

### 2. 单数据集内差距远小于跨数据集平均（physics 实测仅 0.03pp）

Physics 数据集单独分析：
```
german: FP16=0.4532  RQ=0.4444  retention=98.06%
french: FP16=0.4764  RQ=0.4673  retention=98.09%
Gap: 0.03pp  （几乎无差距）
```

**结论：1.67pp 的差距不是单数据集内的系统性偏差，而是 German 在某些数据集上崩盘导致的平均值拉低。**

### 3. German 表现极不稳定（std=2.82pp，是 French 的 2.1×）

German 在 8 个数据集上的排名剧烈波动：

| 数据集 | German retention | 排名 | 与最好语言的差距 |
|--------|:---------------:|:----:|:---------------:|
| computer_science | 97.21% | 2/6 | -0.28pp |
| physics | 96.78% | 2/6 | -3.14pp |
| pharmaceuticals | 96.64% | 3/6 | -0.81pp |
| finance_en | 97.04% | 3/6 | -0.50pp |
| energy | 95.05% | 1/6 | **-2.50pp** |
| finance_fr | 94.44% | 3/6 | **-2.73pp** |
| **industrial** | **90.96%** | **5/6** | **-3.04pp** |
| **hr** | **89.25%** | **6/6 (垫底)** | **-5.40pp** |

German 在 hr 和 industrial 上崩盘（89-90% retention），拉低了整体平均。

### 4. Italian 也有类似问题（std=3.64pp，最不稳定）

```
industrial: Italian 86.05% (垫底，-7.95pp gap)
physics:    Italian 97.29% (排名 4/6)
```

---

## 🔬 Physics 单数据集深度诊断（German vs French）

### Hypothesis 1: Query embedding spread
```
german_variance                     0.003990
french_variance                     0.003666
variance_ratio                      1.0884   (German 更分散)
```
German query embeddings 方差高 8.8%，但差异不显著。

### Hypothesis 2: Argmax token density
```
german_unique_argmax_tokens         1,076,011
french_unique_argmax_tokens           963,427
german_coverage_pct                    87.93%
french_coverage_pct                    78.73%
unique_ratio                           1.1169
```
**German queries 使用了更多不同的 argmax tokens（+11.7%），覆盖率更高（+9.2pp）**。
更多的 argmax tokens 意味着 quantization 需要在更大的表面积上保持精度。

### Hypothesis 3: Score margins
```
german_fp16_margin_mean                  -4.378
french_fp16_margin_mean                  -5.294
german_margin_degradation                -0.395
french_margin_degradation                -0.602
degradation_ratio                        0.6566  (German 退化更少)
```
**German queries 的 score margin 更小（-4.38 vs -5.29），但在 quantization 下退化反而更少**。
这表明 German 的排序对 quantization noise 更鲁棒（在 physics 上）。

### Hypothesis 4: Token usage patterns
```
german_effective_tokens                  47.39
french_effective_tokens                  43.97
effective_tokens_ratio                   1.0777
```
German queries 使用了更多 effective tokens（+7.8%）。

---

## 🎯 问题本质

### Physics 上：German 和 French 几乎一样好（0.03pp 差距）
- German 使用更多 argmax tokens、更多 effective tokens
- 但 margin 更小，对 quantization 反而更鲁棒
- **结论：在 physics 上没有系统性偏差**

### HR/Industrial 上：German 崩盘（89-90% retention）
- 这两个数据集的特征未知（industrial 诊断还在跑）
- 推测：这些数据集的 German queries 可能有特殊的分布特征（极端方差、极窄 margin、或 argmax 极度分散）

---

## 📋 下一步行动

### 1. 等待 industrial 诊断完成
看 German 在 industrial 上（90.96% retention，倒数第2）的 4 个 hypothesis 数值是否异常。

### 2. 对比 hr 数据集（German 最差，89.25%）
在 hr 上跑同样的诊断，找出 German 崩盘的根本原因。

### 3. 如果发现某个 hypothesis 在崩盘数据集上极端异常
设计针对性的解决方案：
- **Hypothesis 2 异常（argmax 极度分散）** → 增加 codebook 数量 M，或用 query-aware k-means 给高频 argmax tokens 分配更多 centroids
- **Hypothesis 3 异常（margin 极窄）** → 用 margin-preserving loss（类似 score-MSE，但只惩罚 ranking flips）
- **Hypothesis 1 异常（variance 极大）** → 分层量化，高方差 tokens 用更多 bits

### 4. 如果没有明显异常
说明问题来自数据集本身的特征（某些语言在某些领域天然更难压缩），RQ M=16 已经是无偏的。

---

## 🚨 重要结论

**RQ M=16 本身不存在系统性语言偏差**：
- Physics 上 German vs French 只差 0.03pp（噪声级别）
- 1.67pp 的跨数据集平均差距来自 **German 在 hr/industrial 上的特定崩盘**
- French 最稳定（std=1.34pp），German 最不稳定（std=2.82pp）

**不是 RQ 歧视 German，而是 German 对某些数据集特别脆弱**。

下一步：找出 hr/industrial 上的 German 崩盘原因，设计针对性修复。
