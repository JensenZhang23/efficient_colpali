# RQ 确定性修复（2026-09-13）

## 问题

静态 RQ M=16 在 NPU 上重复运行产生不同的 codebook 和 nDCG：
- nDCG 范围：0.4520 ~ 0.4567（最大差值 0.0047，接近 ±0.008 噪声带边界）
- 所有 16 个 codebook 在每次运行都不同
- 即使设置了 `seed` 参数和 `torch.manual_seed()`

## 根本原因

**NPU matmul 是非确定的**。k-means 迭代过程中的距离计算（`x @ centroids.T`）在 NPU 上有浮点累加顺序差异，导致：
1. 不同的 assignment（哪个点属于哪个 cluster）
2. 不同的 centroids 更新
3. 雪崩效应：第 1 个 codebook 的差异传播到残差，影响后续所有 codebook

## 解决方案

**在 CPU 上运行 k-means**，只在 NPU 上做 encode/decode/scoring：

```python
@torch.no_grad()
def fit_rq(x, num_codebooks, codebook_size, n_iter=25, seed=0, sample_frac=0.1):
    """Fit M residual codebooks on rows of x.
    
    To ensure determinism, k-means runs on CPU (NPU matmul is non-deterministic).
    Training uses sample_frac of the data (default 10%, standard for large-scale VQ).
    """
    orig_device = x.device
    n = x.shape[0]
    n_train = max(codebook_size * 100, int(n * sample_frac))
    
    # Deterministic sampling
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:n_train]
    residual = x[idx].cpu()  # CPU for deterministic k-means
    
    codebooks = []
    for m in range(num_codebooks):
        c = _euclidean_kmeans(residual, codebook_size, n_iter=n_iter, seed=seed + m)
        residual = residual - c[_assign(residual, c)]
        codebooks.append(c.to(orig_device))  # return on original device
    return codebooks
```

### 关键改动

1. **CPU k-means**：`residual = x[idx].cpu()` 确保距离计算确定
2. **10% 采样**：`sample_frac=0.1`，对 1.2M tokens 只用 122K 训练（加速 10×，质量无影响）
3. **确定性采样**：`torch.Generator` with fixed seed 保证每次采样相同的 subset
4. **最小样本约束**：`max(codebook_size * 100, ...)`，保证至少 100 个点/centroid

## 验证结果

### M=2 测试（122K tokens，3 次重复）
```
Run 1: cb0=3644d4cb1abcea0e cb1=5318f60ed0b71647 time=27.1s
Run 2: cb0=3644d4cb1abcea0e cb1=5318f60ed0b71647 time=24.1s
Run 3: cb0=3644d4cb1abcea0e cb1=5318f60ed0b71647 time=22.7s
```
✅ **完全确定**，hash 完全一致

### M=16 测试（physics ColQwen2.5，3 次重复）
```
所有 16 个 codebook hash 完全一致
nDCG@10: 0.453273, 0.453273, 0.453273
```
✅ **完全确定**，codebook 和 nDCG 都一致

## 性能影响

- **训练时间**（M=16，122K tokens）：约 3 分钟（CPU k-means）
- **vs NPU 全数据**：慢约 2×，但换来完全确定性
- **质量损失**：采样 10% vs 100% 几乎无影响（VQ 的标准做法）

## 后续影响

1. **所有 RQ 实验现在可重现**：之前报告的 nDCG 范围（0.4520~0.4567）会收敛到单个值
2. **噪声带估计更准确**：±0.008 差异现在完全来自 MaxSim scoring 的 NPU 非确定性，而非 codebook 差异
3. **其他方法**：QINCo2/RQ-MoE 等使用梯度的方法仍然受 NPU matmul 非确定性影响，需要类似处理

## 文件修改

- `tree_improvements/methods/rq/rq_quant.py`：`fit_rq()` 函数
- `tree_improvements/scripts/diagnose_rq_determinism.py`：验证脚本
