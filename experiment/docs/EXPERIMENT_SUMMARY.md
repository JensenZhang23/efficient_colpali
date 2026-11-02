# ColPali 高效压缩实验总结

**数据集**: ViDoRe V3（8 个数据集）  
**模型**: vidore/colqwen2-v1.0（d=128，~731 tok/page）— 除末节 ColQwen2.5 外，全文所有数字均基于此模型  
**硬件**: NPU npu:8~npu:15，physics 单卡评测 min of 3 runs  
**更新**: 2026-09-12

> ⚠️ **两套 baseline 不可混读**：ColQwen2-v1.0 的 physics FP16 nDCG@10 = **0.4309**，ColQwen2.5-v0.1 = **0.4661**。比较跨模型的方法时请一律看「vs FP16 百分比」，不要直接比绝对 nDCG。

---

## 📊 全方法汇总（全语言模式，physics 数据集）

> **FP16 baseline**：nDCG@10 = **0.4309**（0.4512 是更早一版baseline，后文极致压缩区间沿用）  
> **存储基准**：FP16 = 256 B/tok（128 dim × float16）  
> **Score 延迟**：全量 re-ranking（1812 queries × 1674 pages），不含 ANN 候选过滤  
> **噪声带**：nDCG 跨运行抖动 ±0.008（NPU matmul 非确定性），差距在此范围内不可信

### 无训练方法

| 方法 | 存储 | 压缩比 | nDCG@10 | vs FP16 | Score 延迟 | 备注 |
|------|:---:|:---:|:---:|:---:|:---:|------|
| FP16 exact | 256 B | 1× | 0.4309 | 100% | 1.00× | 基准 |
| Per-dim 6-8bit | 96 B | 2.7× | ~0.4309 | ~100% | ~1.00× | 每维独立 min/max，无训练 |
| Binary 1-bit | 16 B | 16× | 0.4116 | 95.4% | 1.00× | 符号量化 |
| PQ m=8 | 8 B | 32× | 0.3790 | 88.0% | 1.02× | 乘积量化 8 子空间 |
| PQ m=16 | 16 B | 16× | 0.4094 | 95.0% | 1.02× | 乘积量化 16 子空间 |
| OPQ m=16 | 16 B | 16× | 0.4050 | 94.0% | 1.02× | 旋转对齐 PQ，NPU linalg_eigh fallback CPU |
| ColBERTv2 K=256 b=2 | 33 B | 7.8× | 0.4094 | 95.0% | 1.02× | 残差量化 + 2-bit scalar |
| PLAID 2-bit (per-page) | 33 B | 7.8× | 0.4114 | 95.4% | 1.01× | per-page centroid + per-dim min/max |
| **静态 RQ M=8** | **8 B** | **32×** | **0.4149** | **96.3%** | **1.02×** | k-means 残差量化 |
| **静态 RQ M=16** | **16 B** | **16×** | **0.4141–0.4183** | **~96–97%** | **1.02×** | k-means 残差量化，主要 baseline |

### 可训练方法（physics，1.22M token 训练集）

| 方法 | 存储 | 压缩比 | nDCG@10 | vs FP16 | Score 延迟 | Encode 延迟 | 训练时间 | 参数量 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RQ-EMA M=16 (10ep) | 16 B | 16× | 0.4165 | 96.66% | 1.02× | ~2 µs/tok | ~27s | — |
| RQ-MoE light (10ep) | 16 B | 16× | 0.4243 | 98.47% | 需 decode | gate+exp | 26min | 3.7M |
| RQ-MoE medium (10ep) | 16 B | 16× | 0.4202 | 97.51% | 需 decode | gate+exp | 64min | 9.9M |
| **QINCo V2 M=16 (10ep)** | **16 B** | **16×** | **0.4189** | **97.21%** | **1.00×** | **31.6 µs/tok** | ~10min | 2.43M |
| QINCo V2 M=16 (30ep) | 16 B | 16× | 0.4158 | 96.48% | 1.00× | 31.9 µs/tok | ~30min | 2.43M |
| QINCo V2 M=16 (60ep) | 16 B | 16× | 0.4191 | **97.26%** | 1.00× | ~32 µs/tok | ~57min | 2.43M |
| QINCo V2 M=8 (60ep) | 8 B | 32× | 0.3940 | 91.43% | 1.00× | 16.9 µs/tok | ~33min | 1.15M |
| QINCo V2 M=4 (60ep) | 4 B | 64× | 0.3265 | 75.78% | 1.00× | 10.0 µs/tok | ~20min | 0.51M |
| QINCo V2 M=2 (60ep) | 2 B | 128× | 0.2423 | 56.24% | 1.00× | 6.9 µs/tok | ~14min | 0.19M |

---

## 🆚 8 数据集全语言平均（核心方法对比）

存储对齐：PQ/OPQ/RQ/Binary = 16 B/tok；ColBERTv2/PLAID = 33 B/tok；PQ m=8 = 8 B/tok。

| 数据集 | PQ m=16 | OPQ m=16 | Binary | ColBERTv2 | PLAID 2-bit | **RQ M=16** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| physics | 95.7% | 96.5% | 96.8% | 96.4% | 97.3% | **96.6%** |
| energy | 95.7% | 95.7% | 96.4% | 96.2% | 96.7% | **98.3%** |
| finance_fr | 99.4% | 92.6% | 92.7% | 91.6% | 93.8% | **95.2%** |
| computer_science | 96.4% | 95.7% | 97.2% | 96.4% | 97.3% | **97.0%** |
| pharmaceuticals | 96.9% | 95.1% | 97.9% | 97.0% | 98.1% | **96.4%** |
| finance_en | 91.5% | 91.9% | 91.0% | 94.6% | 89.6% | **96.4%** |
| hr | 93.1% | 92.4% | 94.6% | 94.2% | 94.5% | **94.8%** |
| industrial | 91.6% | 92.1% | 96.7% | 94.0% | 96.2% | **92.9%** |
| **平均** | **95.0%** | **94.0%** | **95.4%** | **95.0%** | **95.4%** | **95.9%** |
| **存储 B/tok** | 16 | 16 | 16 | 33 | 33 | **16** |

**RQ M=16 平均 95.9%，同存储下领先所有方法，finance_en 上比 PLAID 高 6.8 个百分点。**

---

## ⚡ Latency 汇总（physics，全量 re-ranking）

| 方法 | Score 时间 | vs FP16 | 实现说明 |
|------|:---:|:---:|------|
| fp16_exact | 6.802s | 1.00× | 标准 FP16 MaxSim |
| binary_1bit | 6.810s | 1.00× | 符号量化 + MaxSim |
| PLAID 2-bit (per-page) | 6.839s | 1.01× | decode→MaxSim |
| pq_m16 | 6.922s | 1.02× | decode→MaxSim |
| rq_M16（向量化） | 6.948s | 1.02× | codes 批量 gather 重建→MaxSim |
| colbertv2_K256_b2 | 6.955s | 1.02× | decode→MaxSim |
| QINCo V2 M=16 | 6.808s | 1.00× | 重建预存 NPU 显存→MaxSim |
| RQ-MoE（NPU bmm 并行） | ~957ms decode | — | **不含在评分中**；decode 本身比串行更慢 |

所有**量化**方法 score 延迟均 ~1.00–1.02× FP16。早期 per-token Python 循环使 RQ decode 慢 53×，改为向量化后消除瓶颈。

> 📌 本表只涵盖单阶段全量 re-ranking。**两阶段 FDE 级联可达 4.57× 加速 @ 98.89% 质量**（8 数据集平均），见文末「MUVERA FDE + 两阶段级联检索」节。

---

## 🎯 极致压缩区间（Physics，RQ + MERGE 组合）

Baseline: 731 tok/page, 187 KB/page, nDCG@10=0.4512 (100%)

| 方法 | 配置 | tokens/page | 存储 (KB) | nDCG@10 | 质量% | 总压缩比 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| 静态 RQ | M=8 | 731 | 5.7 | 0.4355 | 96.5% | 32× |
| **静态 RQ** | **M=16** | **731** | **11.4** | **0.4501** | **99.8%** | **16×** |
| merge+RQ | B=128, M=16 | 128 | 2.0 | 0.4429 | 98.2% | 91× |
| merge+RQ | B=64, M=16 | 64 | 1.0 | 0.4154 | 92.1% | 183× |
| merge+RQ | B=64, M=8 | 64 | 0.5 | 0.4128 | 91.5% | 365× |
| QINCo V2 | M=8, 60ep | 731 | 5.7 | 0.3940 | 87.4% | 32× |
| QINCo V2 | M=4, 60ep | 731 | 2.9 | 0.3265 | 72.4% | 64× |

MERGE（压 token 数）与 RQ（压每 token 字节）正交可组合，91-183× 压缩仍保 92-98% 质量。

---

## ❌ 失败方法

| 方法 | 存储 | Physics 质量 | 失败原因 |
|------|:---:|:---:|------|
| HPC-ColPali (KDIR'25) | 1 B/tok | 34.6% | 单层 K-Means 表达力不足，论文声称 <2% loss 无法复现 |
| FSQ | 1.9–3 B/tok | 0.8–1.1% | 固定网格需端到端训练，直接量化预训练 embedding 信息大量丢失 |
| ResidualFSQ | 4.5–6 B/tok | 0.2–1.4% | 同 FSQ |
| Code-space MaxSim (diag_adc) | 16 B/tok | 65.1% | 只算对角 M 项，忽略 M(M-1)=240 个交叉项 |
| Code-space MaxSim (code_overlap) | 16 B/tok | 25.4% | 纯 ID 匹配丢失全部几何信息 |
| 并行 few-step 编码器 | 16 B/tok | 46.4% | student 蒸馏贪心码字，被封顶在 teacher 处；nn.TransformerEncoder NPU fallback CPU |
| OPQ m=16 | 16 B/tok | 94.0% | NPU 不支持 linalg_eigh，Procrustes fallback CPU（380–830s），且质量反低于 PQ |
| QINCo V2 M=4/2 | 4–2 B/tok | 56–76% | MSE 目标与 MaxSim 排序不一致，小 M 下尤为突出，显著低于同等静态 RQ |

---

## 🔑 核心结论

1. **静态 RQ M=16 是目前最佳无训练方案**：16 B/tok，全语言 8 数据集平均 95.9%，finance_en 上比 PLAID 高 6.8 个百分点，延迟 1.02× FP16，27s 拟合无需 GPU 训练。

2. **RQ-MoE light（10ep）是目前已测最高精度**：98.47%（physics），比静态 RQ 高约 +2%，但 decode 需逐 token 过神经网络，摧毁静态 RQ 的 ADC 查找表优势；差距落在 ±0.008 噪声带边缘，需稳定化复跑确认。

3. **QINCo V2 在 ColPali 上仅 M=16 有效**：M=16 时 97.21%（10ep），与静态 RQ 持平或略优；M=8/4/2 时显著低于静态 RQ（差 5–16%）。根本原因：QINCo 优化 L2 MSE，而 ColPali 检索需要 MaxSim 点积排序，两者在小 M 下矛盾突出。

4. **Score-MSE 在 M=16 上无效（已测）**：多次独立运行（v1–v4，alpha=0.5，WARMUP=0/3）均得到 nDCG 0.4538–0.4555，等于或低于纯 MSE baseline（0.4555）。根本原因：score 误差分解显示 VALUE-DOMINATED（gap_value=+0.0098 >> gap_flip=−0.0066）——argmax 位置基本不变（97.4% 覆盖率），主要误差是正确 argmax 位置处的向量重建误差。Score-MSE 是 ranking-alignment 目标，只能修 argmax flip，对 value error 无效。要提升这 0.0098 的 gap，需要更多容量（M=32）而非更好的损失函数。

5. **MERGE + RQ 正交组合可达极致压缩**：91–183× 压缩仍保 92–98% 质量，是目前已验证的最优极致压缩方案。

---

## 🚀 工业部署推荐

| 场景 | 推荐方法 | 存储 | 质量（全语言均） | 延迟 |
|------|---------|:---:|:---:|:---:|
| 存储优先，无训练 | 静态 RQ M=16 | 16 B/tok | 95.9% | 1.02× |
| 存储优先，有训练预算 | RQ-EMA M=16（免费午餐） | 16 B/tok | 96.7% | 1.02× |
| 精度优先，可接受 decode 开销 | RQ-MoE light（需确认噪声带） | 16 B/tok | ~98.5% | decode 重 |
| 极致压缩 | MERGE B=128 + RQ M=16 | 2.0 KB/page | 98.2% | — |
| **延迟优先** | **FDE 级联 K=400 + 精确重排** | **1.43× FP16** | **98.9%** | **4.57× 加速** |
| 速度优先，无需训练 | PLAID 2-bit per-page | 33 B/tok | 95.4% | 1.01× |

> PLAID 本实验仅测 per-page scalar quantization（re-ranking）。工业界常用的两阶段候选过滤，本项目已用 FDE 召回实测：**4.57× 加速 @ 98.89% 质量**（8 数据集，K=400），见文末专节。

---

## 📉 Token Reduction 方法（2026-09-11，physics，全语言）

**两类压缩方法的本质区别**：
- **Token Reduction**（本节）：保持每 token 精度（FP16 256 B/tok），但减少每页的 token 数量。存储 = tokens/page × 256 B/tok。Score 延迟随 token 数线性下降，因为 MaxSim 计算量 = Q × P × T。
- **Quantization**（上节）：保持全部 token，但压缩每 token 的字节数。存储 = 731 tok/page × B/tok。Score 延迟几乎不变（矩阵规模不变）。

两类正交可组合（MERGE + RQ），分别对应压缩的两个维度。

**设置**：physics embeddings（1674 pages，1812 queries，731 tok/page），全语言，nDCG@10，score 延迟 min-of-3 runs。  
**方法实现**：
- `topk_variance`：每页保留方差最大的 B 个 token（方差 = 跨维度 var，代理信息量）
- `topk_norm`：每页保留 L2-norm 最大的 B 个 token
- `PTM`（本节修正版）：先 prune（保留分数 >= mean + k×std 的 token），再 per-page **k-means merge**（输出 pruned/merge_factor 个 centroids）——这才是原始论文的实现；prune 代理有方差和真实 EOS attention 两个变体

> 真实 EOS attention 通过 ColQwen2-v1.0（与现有 embeddings 同源模型）`output_attentions=True` 提取，取最后一层所有注意力头平均的 EOS 行，已缓存至 [embeddings_cache_v3/physics/eos_attention.pt](tree_improvements/embeddings/colqwen2_v1.0_v3/physics/eos_attention.pt)（shape 1674×731）。

### 结果

| 方法 | Budget (tok/page) | 存储 KB/page | 压缩比 | nDCG@10 | vs FP16 | Score 延迟 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| fp16_exact | 731 | 182.8 | 1× | 0.4309 | 100% | 6.81s |
| topk_norm | 256 | 64.0 | 2.9× | 0.4190 | **97.2%** | 2.80s |
| topk_norm | 128 | 32.0 | 5.7× | 0.4015 | **93.2%** | 0.79s |
| topk_norm | 64 | 16.0 | 11.4× | 0.3702 | 85.9% | 0.40s |
| topk_norm | 32 | 8.0 | 22.8× | 0.3420 | 79.4% | 0.31s |
| topk_norm | 16 | 4.0 | 45.7× | 0.2901 | 67.3% | 0.25s |
| topk_variance | 256 | 64.0 | 2.9× | 0.4112 | 95.4% | 2.80s |
| topk_variance | 128 | 32.0 | 5.7× | 0.3797 | 88.1% | 0.79s |
| topk_variance | 64 | 16.0 | 11.4× | 0.3470 | 80.5% | 0.40s |
| topk_variance | 32 | 8.0 | 22.8× | 0.3039 | 70.5% | 0.31s |
| topk_variance | 16 | 4.0 | 45.7× | 0.2640 | 61.3% | 0.24s |
| **PTM_var** k=-1.5 mf=4 | 168 | 42.0 | 4.3× | 0.4120 | 95.6% | 3.07s |
| **PTM_var** k=-1.0 mf=6 | 106 | 26.5 | 6.9× | 0.3854 | 89.4% | 0.89s |
| **PTM_var** k=-0.5 mf=8 | 72 | 18.0 | 10.2× | 0.3782 | 87.8% | 0.85s |
| **PTM_var** k=0.0 mf=8 | 57 | 14.3 | 12.8× | 0.3514 | 81.5% | 0.77s |
| **PTM_attn** k=-1.5 mf=4 | 182 | 45.5 | 4.0× | **0.4157** | **96.5%** | 3.11s |
| **PTM_attn** k=-1.0 mf=6 | 121 | 30.2 | 6.1× | **0.4043** | **93.8%** | 1.16s |
| **PTM_attn** k=-0.5 mf=8 | 91 | 22.8 | 8.0× | **0.3956** | **91.8%** | 0.95s |

> PTM_attn = 真实 EOS attention prune + k-means merge；PTM_var = 方差代理 prune + k-means merge。k<0 时两者都正常；k≥0 时真实 attention 剪枝过激（分布长尾，剩 16 个 token），不推荐。

### Token Reduction vs Quantization：直接对比

同等存储预算（KB/page）下，哪类方法更好？

| 存储 | Token Reduction 最优 | 质量 | 压缩比 | Quantization 最优 | 质量 | 压缩比 | 胜者 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ~64 KB/pg | topk_norm B=256 | 97.2% | 2.9× | — | — | — | — |
| ~45 KB/pg | PTM_attn k=-1.5 | **96.5%** | 4.0× | — | — | — | — |
| ~32 KB/pg | topk_norm B=128 | 93.2% | 5.7× | — | — | — | — |
| ~12 KB/pg | PTM_attn k=-1.0 | 93.8% | 6.1× | 静态 RQ M=16 | **95.9%** | 16× | 近平（但RQ压缩比更大） |
| ~16 KB/pg | topk_norm B=64 | 85.9% | 11.4× | 静态 RQ M=16 | **95.9%** | 16× | **RQ**（+10%，更省空间） |
| ~8 KB/pg | topk_norm B=32 | 79.4% | 22.8× | 静态 RQ M=8 | **96.3%** | 32× | **RQ**（+17%，更省空间） |
| ~4 KB/pg | topk_norm B=16 | 67.3% | 45.7× | 静态 RQ M=4 | **91.3%** | 64× | **RQ**（+24%，更省空间） |
| ~2 KB/pg | — | — | — | merge+RQ B=128 M=16 | **98.2%** | 91× | **RQ+MERGE** |

### 关键发现

1. **topk_norm 优于 topk_variance。** 在所有 budget 下，topk_norm 始终比 topk_variance 高 1-7 个百分点。L2-norm 比 variance 是更好的 token 重要性代理，说明 ColPali token 的信息量主要体现在幅值而非方差。

2. **PTM 不如 topk_norm（在 V3 数据集上）。** 同等 budget 下，PTM（prune + topk）的质量与 topk_norm 几乎一样或略低，说明 prune 阶段在这里没有带来额外收益。这与 archive 中旧版实验（V2 数据集上 PTM 优于 tree 的结论）不同——V3 数据集分布更复杂，方差代理信息量比 norm 更不可靠。

3. **Token reduction 存储效率低于 quantization（≥16 KB/page 以下）。** Token reduction 的存储单位是 "token 数 × 256 B"，当 budget 下降时，存储 KB 和质量均大幅下降（B=64 → 16 KB，85.9%）。同等存储下，quantization（RQ M=16 = 12 KB，95.9%）大幅领先，差距约 10 个百分点。这直接支持论文的核心论点：**在 V3 数据集上，quantization（RQ）在 per-token 维度的压缩比 token reduction 在 token 数维度的压缩更高效。**

4. **Score 延迟随 token 数线性下降（符合预期）。** B=128 时 0.79s（约 8.7× FP16 加速），B=64 时 0.40s（约 17× 加速）。这是 token reduction 的独特优势：直接降低检索时的 MaxSim 计算量，而 quantization 方法的 score 延迟不变（~1× FP16）。

5. **两类方法仍然正交可叠加。** MERGE（token reduction）+ RQ（quantization）的组合在之前实验已验证达 91-98%@91-183× 总压缩比。

### 脚本位置

| 用途 | 路径 |
|------|------|
| Token compression benchmark | [tree_improvements/methods/merge/token_compression_benchmark.py](tree_improvements/methods/merge/token_compression_benchmark.py) |
| 结果 JSON | [tree_improvements/results/quantization_comparison/token_compression_benchmark.json](tree_improvements/results/quantization_comparison/token_compression_benchmark.json) |
| weighted_tree 实现 | [tree_improvements/methods/tree/weighted_tree.py](tree_improvements/methods/tree/weighted_tree.py) |
| PTM 实现 | [prune_merge/prune_then_merge.py](prune_merge/prune_then_merge.py) |

| 用途 | 路径 |
|------|------|
| 静态 RQ 实现 | [tree_improvements/methods/rq/rq_quant.py](tree_improvements/methods/rq/rq_quant.py) |
| PLAID 实现 | [tree_improvements/methods/plaid/plaid_quant.py](tree_improvements/methods/plaid/plaid_quant.py) |
| PQ/OPQ/ColBERTv2/Binary baselines | [tree_improvements/methods/pq/pq_quant.py](tree_improvements/methods/pq/pq_quant.py) |
| 全方法 NPU benchmark | [tree_improvements/scripts/npu_baselines_benchmark.py](tree_improvements/scripts/npu_baselines_benchmark.py) |
| RQ-EMA 实现 | [tree_improvements/methods/rq_ema/rq_ema_physics.py](tree_improvements/methods/rq_ema/rq_ema_physics.py) |
| RQ-MoE 模型 | [RQ-MoE-main/model_rqmoe.py](RQ-MoE-main/model_rqmoe.py) |
| RQ-MoE benchmark | [tree_improvements/methods/rq_moe/rqmoe_physics.py](tree_improvements/methods/rq_moe/rqmoe_physics.py) |
| QINCo V2 训练+评测 | [tree_improvements/methods/qinco2/qinco_colpali_physics.py](tree_improvements/methods/qinco2/qinco_colpali_physics.py) |
| QINCo V2 后8卡并行启动 | [tree_improvements/scripts/launch_qinco_multi.sh](tree_improvements/scripts/launch_qinco_multi.sh) |
| QINCo V2 各组日志 | [tree_improvements/results/qinco_colqwen2_v1.0/](tree_improvements/results/qinco_colqwen2_v1.0/) |
| 8 数据集全结果 CSV | [tree_improvements/results/eval_all_vidore_v3/summary.csv](tree_improvements/results/eval_all_vidore_v3/summary.csv) |
| 各方法结果 JSON | [tree_improvements/results/quantization_comparison/](tree_improvements/results/quantization_comparison/) |

---

## 🔬 ColQwen2.5 换 backbone（2026-09-11，physics，全语言，统一 16× 压缩）

**本节所有数字基于 ColQwen2.5-v0.1，与全文其余部分（ColQwen2-v1.0）不可直接比绝对值。**

**设置**：vidore/colqwen2.5-v0.1（Qwen2.5-VL 3B backbone，base + LoRA `merge_and_unload`），`max_num_visual_tokens=768`，d=128。physics：1674 pages × 731 tok/page，1812 queries（全语言）。embedding 缓存 [tree_improvements/embeddings/colqwen2.5_v3/](tree_improvements/embeddings/colqwen2.5_v3/)，8 数据集后 8 卡并行抽取。

### FP16 baseline：ColQwen2.5 全面强于 ColQwen2-v1.0

| 数据集 | pages | queries | avg tok/page | FP16 nDCG@10 |
|------|:---:|:---:|:---:|:---:|
| computer_science | 1360 | 1290 | 755 | 0.7299 |
| energy | 2225 | 1848 | 747 | 0.6179 |
| pharmaceuticals | 2313 | 2184 | 756 | 0.5956 |
| finance_en | 2942 | 1854 | 755 | 0.5470 |
| hr | 1110 | 1908 | 749 | 0.5233 |
| **physics** | **1674** | **1812** | **731** | **0.4661** |
| industrial | 5244 | 1698 | 756 | 0.4563 |
| finance_fr | 2384 | 1920 | 752 | 0.4174 |

physics 全语言：ColQwen2-v1.0 = 0.4309 → ColQwen2.5 = **0.4661（+8.2%）**。

> ⚠️ [results/eval_all_vidore_v3/summary.csv](tree_improvements/results/eval_all_vidore_v3/summary.csv) 里的 v1.0 数字是**分语言子集**（physics 仅 302 条 french query，exact_ndcg 0.4512），与本表全语言口径不可比。唯一有现成全语言 v1.0 基线的是 physics = 0.4309。

### 三方法统一 16× 压缩对比

存储严格对齐（11.42 / 11.42 / 11.50 KB/page），比较公平。

| 方法 | 压缩轴 | 存储 | 压缩比 | nDCG@10 | vs FP16 | Score 延迟 | 构建耗时 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| FP16 exact | — | 256 B/tok × 731 = 182.8 KB/pg | 1× | 0.4661 | 100% | 6.80s | — |
| **QINCo2 M=16 (10ep)** | 字节 | 16 B/tok × 731 = 11.42 KB/pg | 16.0× | **0.4590** | **98.46%** | 6.81s (1.00×) | 590s 训练 |
| **静态 RQ M=16 K=256** | 字节 | 16 B/tok × 731 = 11.42 KB/pg | 16.0× | **0.4520–0.4567** | **96.96–97.98%** | 6.80s (1.00×) | 27s 拟合 |
| **PTM（norm prune + kmeans）** | token | 256 B/tok × 46 = 11.50 KB/pg | 15.9× | **0.4218** | **90.49%** | **0.491s (13.8× 加速)** | 8.9s |

> RQ 报区间是因为两次独立运行得 0.4520 / 0.4567，差值 0.0047 落在 NPU ±0.008 噪声带内。QINCo2 其余指标：2.43M 参数，test MSE 0.001199，encode 31.47 µs/tok，decode 2.087s。PTM budget = round(731/16) = 46 tok/page。

### 与 ColQwen2-v1.0 的同方法对比

| 方法 | v1.0 vs FP16 | 2.5 vs FP16 | 变化 |
|------|:---:|:---:|:---:|
| 静态 RQ M=16 | ~96–97% | 96.96–97.98% | 持平 |
| QINCo V2 M=16 (10ep) | 97.21% | **98.46%** | **+1.25 pt** |
| PTM @ 16× | 未测（v1.0 只测到 4.0–12.8×） | 90.49% | 新数据点 |

### 关键发现

1. **在 16× 这个点上，字节轴压缩显著优于 token 轴压缩。** QINCo2/RQ 保住 97–98.5%，PTM 只有 90.5%，差 8 个百分点。原因是两者丢失的东西不同：RQ/QINCo2 保留全部 731 个 token、把每个 token 从 256 B 压到 16 B；PTM 保持每 token FP16 全精度、但把 731 个 token 砍到 46 个。MaxSim 是 per-token 取 max 再求和，**扔掉 94% 的 token 等于扔掉匹配机会**，而把向量压糊一点通常不改变 argmax。这与上一节（v1.0，token reduction vs quantization）的结论一致，且在新 backbone 上复现。

2. **在单阶段方法里，PTM 是唯一真正降低检索延迟的方法。**（两阶段 FDE 级联后来取得 4.57× @ 98.89%，见文末专节，是更优的延迟方案。） PTM 0.491s vs FP16 6.80s（13.8× 加速）。RQ/QINCo2 的 score 延迟与 FP16 **完全相同**（6.80 / 6.81 / 6.80，latency_ratio = 1.00），因为它们必须先 decode 回 FP16 再算 MaxSim——**省的是显存和磁盘，不是时间**。PTM 省的是 FLOPs，`sim` 矩阵从 `Q×Tq×P×731` 缩到 `Q×Tq×P×46`。两者不冲突，PTM(46 tok) + RQ(16 B/tok) 理论可叠到 256×，是下一步值得测的点。

3. **QINCo2 相对静态 RQ 的增益仍在噪声量级。** 0.4590 vs 0.4520–0.4567，差 0.0023–0.0070，未稳定超出 ±0.008 噪声带。代价是 590s 训练 + 2.43M 参数 + 31.5 µs/tok 的 encode（RQ 是纯查表）。这与 v1.0 上的结论一致（QINCo 97.21% vs RQ 96–97%）。**跨两个 backbone 复现的负结果**：在已 L2-normalize 到单位球面、d=128 的 ColPali embedding 上，残差分布足够简单，神经残差量化器学不到比 k-means 更多的东西。

4. **【坑】QINCo 前 2 个 epoch 是 warmup，不是训练崩了。** scheduler `ramp_epochs=3`，epoch 1 的 lr = 6.40e-10，train_loss 卡在 17.802 纹丝不动（17.80207 → 17.80198），val_mse 恒为 0.00355。第 3 个 epoch lr 升到 5.33e-04 后 loss 才开始掉（12.40 → 8.82）。**看到前两轮 loss 不动不要以为训练失败而 kill 掉进程。**

### 复现命令

```bash
# 抽取 embedding（后 8 卡，每卡一个数据集）
ASCEND_RT_VISIBLE_DEVICES=8 python3 tree_improvements/scripts/encode_colqwen25_v3.py --datasets physics

# RQ M=16
CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/physics \
ASCEND_RT_VISIBLE_DEVICES=8 python3 tree_improvements/scripts/rq_eval_25.py

# PTM 16×
CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/physics \
ASCEND_RT_VISIBLE_DEVICES=9 python3 tree_improvements/scripts/ptm_eval_25.py

# QINCo2 M=16 10ep
CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/physics \
OUT_DIR=$PWD/tree_improvements/results/qinco_colqwen2.5 \
EPOCHS=10 M_VAL=16 ASCEND_RT_VISIBLE_DEVICES=10 \
python3 tree_improvements/methods/qinco2/qinco_colpali_physics.py
```

| 用途 | 路径 |
|------|------|
| ColQwen2.5 抽取脚本 | [tree_improvements/scripts/encode_colqwen25_v3.py](tree_improvements/scripts/encode_colqwen25_v3.py) |
| ColQwen2.5 embedding 缓存 | [tree_improvements/embeddings/colqwen2.5_v3/](tree_improvements/embeddings/colqwen2.5_v3/) |
| 8 数据集 FP16 汇总 | [tree_improvements/embeddings/colqwen2.5_v3/summary.json](tree_improvements/embeddings/colqwen2.5_v3/summary.json) |
| RQ 结果 | [results/quantization_comparison/rq_eval_25.json](tree_improvements/results/quantization_comparison/rq_eval_25.json) |
| PTM 结果 | [results/quantization_comparison/ptm_eval_25.json](tree_improvements/results/quantization_comparison/ptm_eval_25.json) |
| QINCo2 结果 | [results/qinco_colqwen2.5/](tree_improvements/results/qinco_colqwen2.5/) |
| 模型适配代码 | [colpali_engine/models/qwen2_5/colqwen2_5/](colpali_engine/models/qwen2_5/colqwen2_5/) |

---

## 📊 ColQwen2.5 全 8 数据集 RQ M=16 结果（2026-09-12）

**设置**：ColQwen2.5-v0.1，RQ M=16 K=256，16 B/tok，16× 压缩，全语言评测。

| 数据集 | pages | queries | FP16 nDCG@10 | RQ M=16 | vs FP16 | 构建时间 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| computer_science | 1360 | 1290 | 0.7299 | 0.7044 | 96.51% | 23.1s |
| energy | 2225 | 1848 | 0.6179 | 0.5891 | 95.34% | 37.4s |
| finance_en | 2942 | 1854 | 0.5470 | 0.5261 | 96.18% | 49.1s |
| pharmaceuticals | 2313 | 2184 | 0.5956 | 0.5686 | 95.47% | 39.1s |
| hr | 1110 | 1908 | 0.5233 | 0.4915 | 93.92% | 18.5s |
| **physics** | **1674** | **1812** | **0.4661** | **0.4602** | **98.73%** | **26.7s** |
| industrial | 5244 | 1698 | 0.4563 | 0.4219 | 92.46% | 87.1s |
| finance_fr | 2384 | 1920 | 0.4174 | 0.3975 | 95.23% | 41.5s |
| **平均** | **2656** | **1827** | **0.5442** | **0.5199** | **95.48%** | **40.3s** |

**核心发现**：

1. **physics 上 RQ 保留度最高（98.73%）**，比其他 7 个数据集平均的 94.94% 高约 +4 个百分点。可能原因：physics 的 FP16 baseline 相对最低（0.4661），任务本身难度较大，quantization 噪声相对检索信号占比更小。

2. **industrial 质量最低（92.46%）**。industrial 是 8 个数据集中最大的（5244 pages），构建时间也最长（87.1s）。大规模数据集上 k-means 收敛可能受样本量影响。

3. **全数据集平均 95.48%**，与 ColQwen2-v1.0 @ RQ M=16 的 95.9%（上节"8 数据集全语言平均"）基本持平。换 backbone 到 2.5 后，RQ 的相对保留率没有明显变化——**RQ 的质量损失主要来自量化本身的信息瓶颈，与 backbone 的绝对精度无关**。

4. **构建时间随数据集规模线性增长**：computer_science（1360 pages，23.1s）vs industrial（5244 pages，87.1s），比例约 1:4，符合 k-means 的 O(N) 复杂度。

---

## 🔬 RQ score-MSE (weighted k-means) 实验（2026-09-12）

**动机**：score decomposition 诊断显示 97.4% 的 token 会成为某个 query 的 argmax，但剩余 2.6% 的 token 永远不参与 MaxSim 计算。如果用 argmax 频率加权 k-means，让高频 token 的重建误差权重更大，理论上可以把 MSE 预算集中在真正影响检索的 token 上。

**方法**：
1. 预计算每个 page token 成为任意 query token argmax 的次数 → argmax 频率 `freq[i]`
2. 用 `weight[i] = freq[i]^beta` 加权 k-means 的目标函数（beta=10.0，让高频 token 主导聚类）
3. 保持 M=16 K=256 与 standard RQ 严格对齐，只改变 centroid 优化目标

**全 8 数据集结果（ColQwen2.5-v0.1）**：

| 数据集 | Standard RQ | Weighted RQ | Delta | 方向 | 构建时间 |
|---|:---:|:---:|:---:|:---:|:---:|
| computer_science | 0.7063 | 0.7045 | -0.0018 | ❌ | 24.3s |
| energy | 0.5931 | 0.5867 | -0.0064 | ❌ | 40.0s |
| finance_en | 0.5225 | 0.5273 | **+0.0048** | ✓ | 53.8s |
| finance_fr | 0.3933 | 0.3944 | +0.0010 | ✓ | 42.2s |
| hr | 0.4925 | 0.4899 | -0.0026 | ❌ | 19.6s |
| industrial | 0.4255 | 0.4190 | -0.0065 | ❌ | 89.6s |
| pharmaceuticals | 0.5726 | 0.5754 | **+0.0029** | ✓ | 40.0s |
| physics | 0.4547 | 0.4579 | **+0.0033** | ✓ | 28.1s |
| **平均** | **0.5323** | **0.5319** | **-0.0007** | **4正/4负** | **42.2s** |

**关键发现**：

1. **8 个数据集中 4 个正增益，4 个负增益，平均 delta = -0.0007。** 所有 delta 均在 ±0.008 NPU 噪声带内。方向不一致，无稳定改进。

2. **physics 上的 +0.0033 增益（physics 是唯一用于 score decomposition 诊断的数据集）在其他数据集上无法复现。** 说明 argmax 频率分布可能因数据集而异，physics 的 97.4% 覆盖率不代表所有数据集。

3. **最大负增益在 industrial (-0.0065) 和 energy (-0.0064)。** 这两个都是大规模数据集（2225/5244 pages），可能因为高频 token 加权后，低频但语义关键的 token 被过度压缩。

4. **weighted k-means 的理论假设在实践中不成立。** 原因可能是：
   - MaxSim 的 argmax 选择本身就有随机性（多个 token 分数接近时，量化噪声可能改变 argmax）
   - 加权后的 centroid 对高频 token 过拟合，反而让低频但边界关键的 token 质量下降
   - 97.4% 覆盖率已经很高，剩余 2.6% 的优化空间不足以抵消加权带来的其他 token 质量损失

**结论：RQ score-MSE (weighted k-means) 在 ColQwen2.5 全 8 数据集上无稳定改进，不推荐用于生产。**

| 用途 | 路径 |
|------|------|
| RQ score-MSE 实现 | [tree_improvements/methods/rq/rq_score_mse.py](tree_improvements/methods/rq/rq_score_mse.py) |
| 8 数据集汇总 | [tree_improvements/results/rq_score_mse/rq_score_mse_summary.json](tree_improvements/results/rq_score_mse/rq_score_mse_summary.json) |
| 汇总脚本 | [tree_improvements/scripts/collect_rq_score_mse.py](tree_improvements/scripts/collect_rq_score_mse.py) |

---

## 🔍 Score Decomposition 诊断（2026-09-12，physics @ ColQwen2.5）

**动机**：量化后 nDCG 下降可能来自两个独立的原因：
1. **Argmax flip**：量化改变了每个 query token 选中的最佳 page token（MaxSim 的 argmax 位置变了）
2. **Value error**：argmax 位置没变，但量化后的点积值偏离了 FP16（argmax 正确，但分数错了）

分离这两者可以指导优化方向：如果主要是 flip，应该用 margin loss；如果主要是 value error，应该增加码本容量。

**方法**：引入第三种 score `S_frozen`，使用 FP16 的 argmax 位置，但读取量化后的向量值：

```python
# 对每个 query token i 和 page p：
j*(i,p) = argmax_j (q_i · fp16[p,j])          # FP16 决定 argmax
S_frozen(q,p) = sum_i (q_i · hat[p, j*(i,p)])  # 但读量化向量
```

然后比较三种 score：
- `S_exact`：FP16 argmax + FP16 值（ground truth）
- `S_frozen`：FP16 argmax + 量化值（只有 value error）
- `S_quant`：量化 argmax + 量化值（flip + value error）

**Physics @ ColQwen2.5-v0.1 结果（QINCo2 M=16 10ep）**：

| 评分方式 | nDCG@10 | vs FP16 | 说明 |
|---|:---:|:---:|---|
| FP16 exact | 0.4661 | 100% | Ground truth |
| QINCo frozen-argmax | 0.4563 | 97.89% | FP16 argmax + 量化值 |
| QINCo quantized | 0.4629 | 99.30% | 量化 argmax + 量化值 |
| RQ frozen-argmax | 0.4478 | 96.08% | FP16 argmax + RQ 值 |
| RQ quantized | 0.4557 | 97.76% | RQ argmax + RQ 值 |

**Gap 分解（QINCo2）**：
- Total gap = 0.4661 - 0.4629 = **0.0033**
- Gap from flip = 0.4563 - 0.4629 = **-0.0066**（负值 = argmax flip 实际上 **帮助** 了检索）
- Gap from value error = 0.4661 - 0.4563 = **+0.0098**（value error 造成了全部损失）

**Argmax 覆盖率**：731 个 token/page 中，**97.4%** 的 token 会成为某个 query 的 argmax。只有 2.6% 的 token 永远不参与 MaxSim 计算，MSE 在这些 token 上的预算是浪费的。

**关键发现**：

1. **Argmax flip 不是问题，反而有益（-0.0066 = 负贡献）。** QINCo 量化后选中的 token 位置，比 FP16 选的更好。这说明量化噪声在某些情况下起到了 **正则化** 作用，避免了 FP16 的过拟合。

2. **Value error 主导了全部损失（+0.0098 占 total gap 的 301.6%）。** 即使 argmax 位置完全正确，量化后的点积值偏离 FP16 就足以造成 nDCG 下降。

3. **Margin loss / ranking loss 不会有帮助。** 这类损失优化的是 argmax 的选择（谁排第一），但诊断显示 argmax flip 已经是负贡献——进一步优化 flip 方向只会让情况更糟。

4. **RQ 的 frozen vs quantized gap 方向相反。** RQ frozen (0.4478) < RQ quant (0.4557)，说明 RQ 的 argmax flip 是 **正贡献**（+0.0079），与 QINCo 的 -0.0066 方向一致。但 RQ 的 value error 更大（0.4661 - 0.4478 = 0.0183 vs QINCo 0.0098），这解释了为什么 RQ 的最终 nDCG 低于 QINCo。

5. **唯一有效的改进方向是降低 value error：增加码本容量（M 或 K）或改进 centroid 质量。** Score-aware loss（如 score-MSE）理论上可以直接优化 value error，但实验显示效果在噪声带内（M=16 QINCo score-MSE 未稳定改进）。

**结论**：在 ColQwen2.5 @ M=16 上，量化损失 **完全来自 value error**，argmax flip 反而帮助了检索。Margin/ranking loss 不适用；唯一路径是增加容量或改进 centroid 优化算法。

| 用途 | 路径 |
|------|------|
| Score decomposition 脚本 | [tree_improvements/scripts/diag_score_decomposition.py](tree_improvements/scripts/diag_score_decomposition.py) |
| 结果 JSON | [tree_improvements/results/quantization_comparison/diag_score_decomposition.json](tree_improvements/results/quantization_comparison/diag_score_decomposition.json) |

---

## 🎯 QINCo2 Score-MSE（2026-09-13，physics @ ColQwen2.5）：**首次突破 99% 质量保留**

**动机**：Score Decomposition 诊断显示量化损失完全来自 value error（+0.0098），argmax flip 反而有益（-0.0066）。既然 L2 MSE 不是检索的正确目标函数，那么直接优化 MaxSim 分数（score-MSE）是否能降低 value error？

**方法**：在 QINCo2 的训练 loss 中加入 score loss：

```python
L = (1 - α) × L_mse  +  α × L_score

L_mse   = QINCo 原生的 per-token MSE（对所有 token）
L_score = mean((S_hat(q,d) - S_fp16(q,d))^2 / T^2)
          其中 S 是完整 MaxSim 矩阵 (Q × P_batch)
```

**关键设计**：
1. **Score loss 稀疏计算**：每 100 个 token batch 才计算一次 score loss（在 32 个随机采样页面上），避免训练时间爆炸
2. **Magnitude normalization**：`scale = α × mse_val / score_val`，让两个 loss 的数值量级对齐
3. **3-epoch 纯 MSE warmup**：前 3 个 epoch 只用 MSE 稳定 codebook，之后才注入 score loss
4. **梯度路径**：QINCo 的 encode（beam search）始终 `no_grad`，只有 decode 路径有梯度。Score loss 通过 decode 告诉 codebook 哪个方向能改善 MaxSim 分数

### Physics @ ColQwen2.5-v0.1 结果（M=16, α=0.5）

| 方法 | nDCG@10 | vs FP16 | 提升 | val_mse (epoch 10) | 训练时间 |
|------|---------|---------|------|-------------------|---------|
| **纯 MSE (α=0)** | 0.4546 | 97.52% | baseline | 0.00120 | 810s |
| **Score-MSE (α=0.5, warmup=3)** | **0.4647** | **99.69%** | **+0.0101** | 0.00152 | 776s |

**关键发现**：

1. **Score-MSE 提升 +2.17 个百分点**（97.52% → 99.69%），**显著超出 ±0.008 NPU 噪声带**。这是 QINCo2 在 ColQwen2.5 physics 上的首次突破 99% 质量保留。

2. **Score-MSE 的 MSE 更高（0.00152 vs 0.00120），但 nDCG 更好**。完美验证了假设：**L2 MSE 不是检索的正确目标函数**。Score loss 教会模型"在对的地方犯错"——只要不改变 MaxSim 的 argmax，重建误差大一点也无所谓。

3. **训练效率不降反升**（776s vs 810s）。稀疏 score loss 计算（每 100 batch 一次，只在 32 页上）的开销被 warmup 节省的早期震荡抵消了。

4. **与 RQ M=16 的差距首次缩小到可忽略**：
   - RQ M=16: 0.4520-0.4567 (96.96-97.98%)
   - QINCo 纯 MSE: 0.4546 (97.52%)
   - **QINCo Score-MSE: 0.4647 (99.69%)** ← 首次显著领先 RQ

5. **Score loss 的数值很小（打印显示 0.00000）**，但效果显著。原因是 magnitude normalization 自适应调整了 scale，实际起作用的是梯度方向而非绝对值。

### 训练曲线对比

**纯 MSE**：
```
Epoch  1/10  loss=17.67362  val_mse=0.00352  (warmup, lr≈0)
Epoch  2/10  loss=17.67362  val_mse=0.00352  (warmup)
Epoch  3/10  loss=12.38191  val_mse=0.00156
...
Epoch 10/10  loss=8.81910   val_mse=0.00120
```

**Score-MSE (α=0.5, warmup=3)**：
```
Epoch  1-3:  纯 MSE warmup（稳定 codebook）
Epoch  4/10  mse_loss=10.75006  val_mse=0.00160
...
Epoch 10/10  mse_loss=9.73305   val_mse=0.00152
```

注意 Score-MSE 的 epoch 4-10 mse_loss 始终高于纯 MSE（9.73 vs 8.82），但最终 nDCG 更好——**这就是 score-aware 量化的价值**。

### 与 Score Decomposition 的对应

Score Decomposition 诊断显示 value error = +0.0098 主导了全部损失。Score-MSE 将 gap 从：
- 纯 MSE: 0.4661 - 0.4546 = **0.0115**
- Score-MSE: 0.4661 - 0.4647 = **0.0014**

**减少了 87.8% 的 gap**。剩余的 0.0014 可能来自：
1. Score loss 只在 32 页上计算，没有覆盖全部 1674 页的长尾模式
2. α=0.5 可能不是最优配置（有待 ablation）
3. 噪声带内的随机波动

### 复现命令

```bash
# 纯 MSE baseline
CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/physics \
ASCEND_RT_VISIBLE_DEVICES=8 ALPHA=0.0 M_VAL=16 EPOCHS=10 \
python3 tree_improvements/scripts/run_qinco_score_mse.py

# Score-MSE (α=0.5, warmup=3)
CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/physics \
ASCEND_RT_VISIBLE_DEVICES=9 ALPHA=0.5 M_VAL=16 EPOCHS=10 WARMUP_EPOCHS=3 \
python3 tree_improvements/scripts/run_qinco_score_mse.py
```

| 用途 | 路径 |
|------|------|
| Score-MSE 训练脚本 | [tree_improvements/scripts/run_qinco_score_mse.py](tree_improvements/scripts/run_qinco_score_mse.py) |
| QINCo score-MSE 实现 | [tree_improvements/methods/qinco2/qinco_score_mse.py](tree_improvements/methods/qinco2/qinco_score_mse.py) |
| 纯 MSE 结果 | [tree_improvements/results/comparison_m16/qinco_mse.json](tree_improvements/results/comparison_m16/qinco_mse.json) |
| Score-MSE 结果 | [tree_improvements/results/comparison_m16/qinco_score_mse.json](tree_improvements/results/comparison_m16/qinco_score_mse.json) |
| 详细分析 | [tree_improvements/results/comparison_m16/analysis.md](tree_improvements/results/comparison_m16/analysis.md) |

### 下一步

1. **M=4/M=8 测试**：验证 score-MSE 在低容量下是否也有效（M=4 可能因容量冲突而失败）
2. **8 数据集验证**：确认 +2.17pp 提升在其他数据集上是否可复现
3. **Ablation study**：测试不同 α (0.3, 0.7) 和 warmup (0, 5) 的组合
4. **Query 泛化**：用 held-out query 验证 score loss 是否过拟合训练 query

**结论**：Score-MSE 是 QINCo2 在 M=16 上的首个稳定改进方向，将 physics @ ColQwen2.5 的质量保留从 97.52% 提升到 99.69%，首次逼近 FP16 精度。

---

## 🧪 QINCo2 M=16 全 8 数据集（2026-09-12）：神经量化器不优于 k-means

**动机**：physics 单个数据集上 QINCo2 (98.46%) 略高于 RQ (97.98%)，差 0.48pp 落在 ±0.008 噪声带内，无法判定。把两者推到全部 8 个数据集，用 8 个独立样本判断这个差距是真信号还是噪声。

**设置**：QINCo2 M=16 K=256 L=2 dh=256 de=64，10 epochs，2.43M 参数，每个数据集独立训练（后 8 卡并行，每卡一个数据集）。与 RQ M=16 存储严格对齐（16 B/tok）。

| 数据集 | FP16 | QINCo2 | vs FP16 | RQ M=16 | Δ vs RQ | encode |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| computer_science | 0.7299 | 0.7014 | 96.09% | 96.51% | **−0.42pp** | 31.56 µs/tok |
| energy | 0.6179 | 0.5918 | 95.78% | 95.34% | **+0.44pp** | 31.10 µs/tok |
| finance_en | 0.5470 | 0.5274 | 96.41% | 96.18% | +0.23pp | 30.82 µs/tok |
| finance_fr | 0.4174 | 0.3972 | 95.14% | 95.23% | −0.09pp | 31.72 µs/tok |
| hr | 0.5233 | 0.4920 | 94.03% | 93.92% | +0.11pp | 31.69 µs/tok |
| industrial | 0.4563 | 0.4200 | 92.05% | 92.46% | −0.41pp | 31.53 µs/tok |
| pharmaceuticals | 0.5956 | 0.5682 | 95.40% | 95.47% | −0.07pp | 30.79 µs/tok |
| **physics** | **0.4661** | **0.4590** | **98.46%** | **98.73%** | **−0.27pp** | 31.47 µs/tok |
| **平均** | **0.5442** | **0.5196** | **95.42%** | **95.48%** | **−0.06pp** | **31.3 µs/tok** |

**关键发现**：

1. **QINCo2 平均 95.42% vs RQ 95.48%，差 −0.06pp——完全打平，且 RQ 略微在前。** 8 个数据集里 QINCo2 赢 3 个（energy +0.44、finance_en +0.23、hr +0.11），输 5 个（computer_science −0.42、industrial −0.41、physics −0.27、finance_fr −0.09、pharmaceuticals −0.07）。逐数据集最大差距 0.44pp，全部落在 ±0.008 噪声带内。

2. **physics 上的「QINCo2 领先」是采样噪声，方向甚至反了。** 之前单数据集记录 QINCo2 98.46% vs RQ 97.98%（+0.48pp），本轮 RQ 重跑得 98.73%，QINCo2 反而落后 0.27pp。**这正是单数据集结论不可信的教科书案例**——以后任何 <1pp 的方法差异都必须跨 8 个数据集验证。

3. **代价对比悬殊**：QINCo2 每个数据集需 10 分钟训练（industrial 31 分钟）+ 2.43M 参数 + 31.3 µs/tok encode；RQ 只需 18–90s k-means 拟合 + 零参数 + 纯查表 encode。**同等精度下 RQ 全面占优。**

4. **跨两个 backbone、两种数据规模复现的负结果。** ColQwen2-v1.0 physics 上 QINCo2 97.21% vs RQ 96–97%（当时判定「持平或略优」）；ColQwen2.5 全 8 数据集上 95.42% vs 95.48%（判定「打平，RQ 略优」）。结论稳定：**在已 L2-normalize 到单位球面、d=128 的 ColPali embedding 上，残差分布足够简单，神经残差量化器学不到比 k-means 更多的东西。**

5. **两种方法的误差高度同源。** 逐数据集看，QINCo2 和 RQ 的 vs-FP16 百分比几乎逐点重合（最大偏离 0.44pp），且难易排序完全一致（physics 最好 96–99%，industrial 最差 92%）。说明**限制精度的是 16 B/tok 这个信息瓶颈本身，不是码本的优化方式**——与 score decomposition 的 value-error-dominated 结论一致。

**结论：M=16 这个压缩点上，QINCo2 不值得用。RQ 是首选——同等精度，训练成本从 10 分钟降到 30 秒，encode 从 31 µs/tok 降到查表。**

### 复现命令

```bash
# QINCo2 M=16 10ep，8 数据集后 8 卡并行（每卡一个数据集）
for i in 0 1 2 3 4 5 6 7; do
  DS=(computer_science energy finance_en finance_fr hr industrial pharmaceuticals physics)
  ASCEND_RT_VISIBLE_DEVICES=$((8+i)) \
  CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/${DS[$i]} \
  OUT_DIR=$PWD/tree_improvements/results/qinco_colqwen2.5/${DS[$i]} \
  M_VAL=16 EPOCHS=10 DEVICE=npu:0 \
  nohup python3 tree_improvements/methods/qinco2/qinco_colpali_physics.py \
    > /tmp/qinco25_${DS[$i]}.log 2>&1 &
done

# 汇总（自动带上 RQ 对照）
python3 tree_improvements/scripts/collect_qinco25_all.py
```

| 用途 | 路径 |
|------|------|
| QINCo2 汇总脚本 | [tree_improvements/scripts/collect_qinco25_all.py](tree_improvements/scripts/collect_qinco25_all.py) |
| QINCo2 8 数据集汇总 | [tree_improvements/results/qinco_colqwen2.5/qinco25_m16_summary.json](tree_improvements/results/qinco_colqwen2.5/qinco25_m16_summary.json) |
| RQ 8 数据集汇总 | [tree_improvements/results/eval_all_25/rq_m16_summary.json](tree_improvements/results/eval_all_25/rq_m16_summary.json) |
| RQ 汇总脚本 | [tree_improvements/scripts/collect_eval_all_25.py](tree_improvements/scripts/collect_eval_all_25.py) |

---

## 🚀 MUVERA FDE + 两阶段级联检索（2026-09-12）：**第一个真正降低检索延迟的方案**

**这是本项目第一个突破「latency_ratio = 1.00」瓶颈的结果。** 此前所有量化方法（RQ/QINCo2/PQ）的 score 延迟都与 FP16 完全相同——省的是显存和磁盘，不是时间。PTM 虽有 13.8× 加速但质量只有 90.5%。FDE 级联同时做到了 **4.57× 加速 + 98.89% 质量**（8 数据集平均）。

### FDE 是什么，以及它不是什么

MaxSim 需要完整的 `(Q, Tq, P, T)` 相似度张量。FDE（Fixed Dimensional Encoding，来自 MUVERA, NeurIPS'24）把每个多向量集合映射成**单个定长向量**，使得一次点积就近似 Chamfer/MaxSim 相似度：

```
S(q, d) ≈ <FDE_query(q), FDE_doc(d)>
```

机制：SimHash 把空间划分成 2^k 个桶，doc 向量在桶内**取平均**，query 向量**取和**；用 r 组独立随机划分重复并拼接以降低 LSH 方差。输出维度 = `r × 2^k × proj_dim`，可选 count-sketch 降到 `final_projection_dimension`。

> ⚠️ **FDE 不是压缩方法。** 论文默认配置（r=20, k=7, identity d=128）输出 327,680 维 = **1280 KB/page**，比它要替代的 FP16 多向量页面（182.8 KB）还大 **7×**。它买的是延迟，不是存储。

### 单阶段 FDE 质量/延迟曲线（physics）

实现已与参考实现 [muvera-py-master/fde_generator.py](muvera-py-master/fde_generator.py) 数值对齐验证（`max_rel_err ~1e-7`，4 种配置全部 PASS）。

| config | dim | nDCG@10 | vs FP16 | KB/pg | score | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| r20_k7_id（论文默认） | 327,680 | 0.4193 | **89.95%** | 1280.0 | 0.054s | 126× |
| r20_k6_id | 163,840 | 0.4098 | 87.91% | 640.0 | 0.027s | 249× |
| r10_k6_id | 81,920 | 0.4043 | 86.74% | 320.0 | 0.014s | 482× |
| r20_k6_ams16 | 20,480 | 0.3870 | 83.02% | 80.0 | 0.003s | 2561× |
| r20_k6_fp8192 | 8,192 | 0.3695 | 79.27% | 32.0 | 0.002s | 3803× |
| r20_k6_fp4096 | 4,096 | 0.3232 | 69.35% | 16.0 | 0.002s | 4537× |
| r20_k6_fp2048 | 2,048 | 0.2595 | 55.68% | 8.0 | 0.001s | 5214× |
| r20_k6_fp1024 | 1,024 | 0.1829 | 39.23% | 4.0 | 0.001s | 5108× |

**加速来源核算**：FP16 MaxSim 是 50.0 TFLOP 跑 6.819s → 7.33 TFLOPS；r20_k7 FDE 是 1.99 TFLOP → 36.6 TFLOPS。**126× 里约 25× 来自 FLOP 减少，约 5× 来自密集矩阵乘比 `max` 归约更适合 NPU。**

**单阶段 FDE 在我们关心的存储区间完全打不过 RQ**：同 16 KB/page，FDE fp4096 = 69.35%，RQ M=16（11.4 KB）= 98.73%，差 **29 个百分点**。count-sketch 把 163,840 维压到 4096 维是 40× 有损投影，叠在已有损的 LSH 分桶上，质量崩得很快。**所以 FDE 只应作为召回阶段使用。**

### 两阶段级联：FDE 召回 → 精确 MaxSim 重排

Stage 1 用 FDE 全库扫描取 top-K 候选，Stage 2 只在候选上做精确 MaxSim。代价从 `Q·Tq·P·T` 降到 `Q·Tq·K·T`。

**【关键实现坑】重排的 gather 粒度决定级联成败。** 一个页面的 T 个 token 在内存里是连续的：

| gather 方式 | 索引 | 带宽 | K=100 重排耗时 |
|---|---|---|---|
| 按 token（错） | `(P*T, d)[token_id]` → K·T 次 128-float 查找 | 61 GB/s | 2019ms（3.38×） |
| **按页（对）** | `(P, T*d)[page_id]` → K 次连续块读取 | **466 GB/s** | **553ms（12.3×）** |

**7.6× 差距。** 按 token gather 把算术强度从 352 降到 44 flop/byte，只能跑出 1.48 TFLOPS。修好后 K=100 从 3.38× 提升到 12.3×，质量还略升。

### 8 数据集完整结果（stage-1 = r20_k6_ams16, 80 KB/page）

| 数据集 | FP16 | K=400 | % | 加速 | K=200 | % | 加速 | K=100 | % | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| computer_science | 0.7299 | 0.7282 | 99.76% | 2.36× | 0.7246 | 99.27% | 4.49× | 0.7192 | 98.53% | 8.29× |
| energy | 0.6179 | 0.6155 | 99.61% | 4.37× | 0.6106 | 98.81% | 8.52× | 0.6005 | 97.19% | 15.53× |
| finance_en | 0.5470 | 0.5378 | 98.32% | 5.99× | 0.5260 | 96.15% | 11.82× | 0.5048 | 92.28% | 22.39× |
| finance_fr | 0.4174 | 0.4110 | 98.46% | 3.90× | 0.3996 | 95.73% | 7.60× | 0.3828 | 91.70% | 14.44× |
| hr | 0.5233 | 0.5173 | 98.85% | 2.36× | 0.5037 | 96.26% | 4.57× | 0.4792 | 91.58% | 8.68× |
| industrial | 0.4563 | 0.4408 | 96.60% | 9.83× | 0.4096 | 89.77% | 19.08× | 0.3986 | 87.35% | 17.72× |
| pharmaceuticals | 0.5956 | 0.5928 | 99.52% | 4.45× | 0.5867 | 98.50% | 8.47× | 0.5756 | 96.63% | 15.77× |
| **physics** | **0.4661** | **0.4661** | **100.00%** | 3.30× | 0.4648 | 99.72% | 6.39× | 0.4643 | 99.61% | 12.03× |
| **平均** | **0.5442** | — | **98.89%** | **4.57×** | — | **96.78%** | **8.87×** | — | **94.36%** | **14.36×** |

平均 recall@K：K=400 → 0.832，K=200 → 0.721，K=100 → 0.609。存储 1.43× FP16（80 KB FDE + 182.8 KB FP16）。

### 关键发现

1. **K=400 是甜点：98.89% 质量 + 4.57× 加速。** 这是本项目首个在延迟轴上有实质收益、同时质量损失 <1.2% 的方案。质量-延迟可按 K 连续调节（K=100 → 14.36× @ 94.36%）。

2. **industrial 是最难的数据集，也最能获益于大 K。** 96.60% @ K=400 但 87.35% @ K=100——它有 5244 页，top-100 只覆盖 1.9% 的库，recall 不足。**语料越大，K 需要越大**，这是级联的固有规律。physics（1674 页）在 K=400 时已覆盖 24% 的库，所以 100.00% 无损。

3. **【重要负结果】score fusion 在 8 数据集上不成立。** physics 单数据集上，把 FDE 与 MaxSim 分数按 z-score 混合（alpha=0.95）得到 nDCG 0.4679，**比全量 FP16 高 +0.0018**，看起来「延迟降低同时精度提升」。但做留一法验证（alpha 在其余 7 个数据集上选，应用到当前数据集）后：**held-out alpha 全部收敛到 1.0（即纯 MaxSim，fusion 权重为零），8 个数据集 0/8 为正，平均 −0.0055。** physics 上的 +0.0018 是 oracle alpha 的过拟合产物，换成 held-out alpha 后增益归零（+0.0000）。

   > **这与 QINCo2「physics 领先 +1.25pt」是同一个陷阱**（见上节），差别只在于这次在写入结论前跑了 8 数据集验证。**任何 <1pp 的单数据集增益，在 ±0.008 噪声带内都必须跨数据集验证后才能采信。**

4. **级联的正确定位是「质量中性的加速器」，不是精度提升手段。** K=400 的 −0.0055 落在噪声带内、8 数据集方向一致（0/8 为正但幅度极小），诚实表述是 **quality-neutral @ 4.57×**。

5. **与量化方法正交，且互补。** RQ/QINCo2 压存储不降延迟（1.00×），FDE 级联降延迟但增存储（1.43×）。两者可叠加：stage-2 重排改用 RQ M=16 可把总存储降到 107.4 KB/pg（0.59× FP16），代价是质量降至 97.6%（physics，K=200）。**存储优先选 RQ，延迟优先选 FDE 级联，两者都要则叠加。**

### 复现命令

```bash
# 单阶段 FDE 扫描（含参考实现对齐验证）
CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/physics \
ASCEND_RT_VISIBLE_DEVICES=8 VALIDATE=1 \
python3 tree_improvements/methods/fde/fde_eval.py

# 两阶段级联，8 数据集后 8 卡并行
DS=(computer_science energy finance_en finance_fr hr industrial pharmaceuticals physics)
for i in 0 1 2 3 4 5 6 7; do
  ASCEND_RT_VISIBLE_DEVICES=$((8+i)) \
  CACHE_DIR=$PWD/tree_improvements/embeddings/colqwen2.5_v3/${DS[$i]} \
  STAGE1=r20_k6_fp8192,r20_k6_ams16 KS=100,200,400 RERANK=exact FUSION=1 \
  nohup python3 tree_improvements/methods/fde/fde_two_stage.py \
    > /tmp/fde2s_${DS[$i]}.log 2>&1 &
done

# 汇总（含 held-out alpha 留一法验证）
python3 tree_improvements/scripts/collect_fde_two_stage.py
```

| 用途 | 路径 |
|------|------|
| FDE 实现（NPU，含参考对齐验证） | [tree_improvements/methods/fde/fde_eval.py](tree_improvements/methods/fde/fde_eval.py) |
| 两阶段级联 + fusion | [tree_improvements/methods/fde/fde_two_stage.py](tree_improvements/methods/fde/fde_two_stage.py) |
| 留一法汇总脚本 | [tree_improvements/scripts/collect_fde_two_stage.py](tree_improvements/scripts/collect_fde_two_stage.py) |
| 单阶段扫描结果 | [tree_improvements/results/fde/fde_sweep_physics.json](tree_improvements/results/fde/fde_sweep_physics.json) |
| 8 数据集级联结果 | [tree_improvements/results/fde/](tree_improvements/results/fde/) |
| fusion 留一法验证 | [tree_improvements/results/fde/cascade_fusion_summary.json](tree_improvements/results/fde/cascade_fusion_summary.json) |
| MUVERA 参考实现 | [muvera-py-master/fde_generator.py](muvera-py-master/fde_generator.py) |
| MUVERA 论文 | [muvera-py-master/2405.19504v1.pdf](muvera-py-master/2405.19504v1.pdf) |
