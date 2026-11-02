# Efficient ColPali — 实验目录

本目录包含在 ColQwen2.5 + ViDoRe V3 上复现 token 量化与 token 压缩实验所需的全部内容。

## 目录结构

```
experiment/
├── docs/                          # 文档（即本文件）
│   └── requirements.txt           # Python 依赖
├── embeddings/                    # 嵌入向量缓存（由 embedding.py 生成）
│   ├── colqwen2.5_v3/             # ColQwen2.5 全语言嵌入（8 个数据集）
│   └── colqwen2_v1.0_v3/          # ColQwen2 v1.0 嵌入（PTM 专用）
├── methods/
│   ├── muvera/
│   │   ├── fde/                   # FDE 召回 + 两阶段级联
│   │   └── fde_generator.py       # MUVERA FDE 生成器（原 muvera-py-master）
│   ├── token_prune_merge/
│   │   ├── ptm/                   # 基于真实 EOS 注意力的 Prune-Then-Merge
│   │   ├── merge/                 # Token 压缩基准（topk、local_merge）
│   │   ├── tree/                  # 加权树 token 合并
│   │   └── prune_merge/           # PTM 网格搜索实验与结果
│   └── token_quantization/
│       ├── rq/                    # 残差量化（核心方法）
│       ├── qinco2/
│       │   └── qinco/             # QINCo 训练核心（原 Qinco-main/qinco）
│       ├── rq_moe/
│       │   └── model_rqmoe.py     # RQ-MoE 模型（原 RQ-MoE-main）
│       ├── colbertsar.py          # ColBERT SAR 工具（原 sparse/）
│       ├── binary/、colbertv2/、opq/、pq/、plaid/  # 基线方法
│       └── rq_ema/                # RQ-EMA 变体
├── results/                       # 评测结果输出（由 test.py 写入）
└── scripts/
    ├── download.py                # 下载模型权重和数据集
    ├── embedding.py               # 提取 ColQwen2.5 嵌入向量
    ├── encode_colqwen25_v3.py     # 被 embedding.py 调用
    ├── train.py                   # 训练量化模型（QINCo M=16）
    └── test.py                    # 全方法评测（FP16 + RQ + FDE + PTM）
```

## 完整流程：从零开始到得到结果

### 第一步 — 下载模型权重与数据集

```bash
python experiment/scripts/download.py \
    --cache-dir /path/to/.cache/huggingface \
    --datasets all
```

下载 ColQwen2.5-base 权重（分片 safetensors）、ColQwen2.5-v0.1 适配器，以及全部 8 个 ViDoRe V3 数据集（parquet → Arrow 缓存）。默认使用 hf-mirror.com；可通过 `--hf-endpoint https://huggingface.co` 覆盖。

主要参数：
- `--datasets physics energy` — 仅下载指定子集
- `--skip-weights` / `--skip-datasets` — 跳过其中一半
- `--hf-endpoint` — 覆盖 HuggingFace 镜像地址

### 第二步 — 提取嵌入向量

```bash
python experiment/scripts/embedding.py \
    --data-dir /path/to/.cache/huggingface \
    --out-dir   /path/to/embeddings \
    --datasets  all \
    --npu-phys  8
```

在数据集图像和查询上运行 ColQwen2.5，将 `page_embeddings.pt`、`query_embeddings.pt` 和 `relevance.pt` 写入 `--out-dir/<dataset>/`。若缓存文件已存在则跳过该数据集。

主要参数：
- `--model-base` / `--model-adpt` — 模型快照目录的显式路径（默认为 `--data-dir/hub/` 下）
- `--npu-phys` — 使用的物理 NPU 卡号（默认：8）
- `--datasets physics energy` — 指定子集

### 第三步 — 训练量化模型

```bash
python experiment/scripts/train.py \
    --emb-dir experiment/embeddings/colqwen2.5_v3 \
    --out-dir  experiment/results/quant \
    --method   all \
    --wait
```

在全部 8 个数据集上并行训练量化模型，每个数据集独占一张 NPU 卡（默认 npu:8~15）。

支持的方法（`--method`）：
- `rq` — RQ M=16（CPU k-means，无 GPU 要求，约 5 分钟/数据集）
- `qinco` — QINCo2 M=16 MSE（NPU 混合精度，约 30~60 分钟/数据集）
- `all` — 两者都训练（默认）

主要参数：
- `--datasets physics energy` — 仅训练指定子集
- `--npus 8 9` — 覆盖卡号（数量须与 `--datasets` 一致）
- `--epochs 10 --m 16 --k 256` — QINCo 训练超参
- `--qinco-dir` — QINCo 包目录（默认 `experiment/methods/token_quantization/qinco2`）
- `--wait` — 阻塞等待所有训练完成

### 第四步 — 全方法评测

```bash
python experiment/scripts/test.py \
    --emb-dir experiment/embeddings/colqwen2.5_v3 \
    --out-dir  experiment/results \
    --mode     all \
    --wait
```

在全部 8 个数据集上评测所有方法。评测顺序：FP16 上界（主进程内联，极快）→ RQ M=16 → FDE 两阶段级联（等 RQ 退出后启动，同卡无冲突）→ PTM（CPU 并行）。

支持的模式（`--mode`）：
- `fp16` — 仅 FP16 exact baseline
- `rq` — 仅 RQ M=16
- `fde` — 仅 FDE 两阶段级联
- `ptm` — 仅 PTM prune-then-merge
- `all` — 全部（默认）

主要参数：
- `--datasets physics energy` — 仅评测指定子集
- `--npus 8 9` — 覆盖卡号
- `--log-dir /tmp/my_logs` — 日志目录
- `--wait` — 阻塞等待所有任务完成并打印汇总

---

## scripts 目录说明

```
experiment/scripts/
├── download.py             # 第一步：下载模型权重 + ViDoRe V3 数据集
├── embedding.py            # 第二步：提取 ColQwen2.5 全语言嵌入向量
├── encode_colqwen25_v3.py  # 被 embedding.py 内部调用，不直接使用
├── train.py                # 第三步：训练量化模型（RQ + QINCo2）
└── test.py                 # 第四步：全方法评测（FP16 + RQ + FDE + PTM）
```

各方法的核心实现在 `methods/` 下：
- RQ 训练/评测：[methods/token_quantization/rq/train_rq.py](../methods/token_quantization/rq/train_rq.py)
- QINCo 训练：[methods/token_quantization/qinco2/train_qinco.py](../methods/token_quantization/qinco2/train_qinco.py)
- FDE 评测：[methods/muvera/fde/fde_two_stage.py](../methods/muvera/fde/fde_two_stage.py)
- PTM 评测：[methods/token_prune_merge/ptm/ptm_real_attention.py](../methods/token_prune_merge/ptm/ptm_real_attention.py)

### FP16 exact baseline
直接用 FP16 向量计算 MaxSim，作为质量上界。内联在 `test.py` 中，无需单独训练。

### RQ M=16（[run_rq_m16.py](../methods/token_quantization/rq/rq_quant.py)）
确定性 CPU k-means 拟合 M=16 K=256 残差量化，对所有页面编码后评估 nDCG@10。无需训练，每次从头拟合（~5 分钟）。

### QINCo2 M=16 MSE（`train.py` → `test.py`）
以纯 MSE 损失在 NPU 上混合精度训练 QINCo2（M=16 K=256，L=2，dh=256），10 个 epoch，约 30~60 分钟/数据集。训练产出模型文件，`test.py` 加载推理评测。

### FDE 两阶段级联（[fde_two_stage.py](../methods/muvera/fde/fde_two_stage.py)）
FDE（SimHash 固定维度编码）召回 → 精确或 RQ 重排。多种 FDE 配置（amsSketch 4k/8k、full）× 多个召回 K 值扫描。输出 `fde_two_stage_<dataset>.json`。

参数：`--stage1 r20_k7_id,... --ks 10,25,100 --rerank exact,rq --no-fusion --m 16`

### PTM（[ptm_real_attention.py](../methods/token_prune_merge/ptm/ptm_real_attention.py)）
基于真实 EOS 注意力的 Prune-Then-Merge，仅 CPU。需要 ColQwen2-v1.0 的嵌入（注意力行须与嵌入行对齐）。首次运行会提取并缓存注意力到 `eos_attention.pt`。

```bash
python experiment/methods/token_prune_merge/ptm/ptm_real_attention.py \
    --cache-dir    /path/to/embeddings_v1.0/physics \
    --corpus-arrow /path/to/hf_cache/vidore___vidore_v3_physics/corpus/data.arrow \
    --model-base   /path/to/colqwen2-base/snapshots/... \
    --model-adpt   /path/to/colqwen2-v1.0/snapshots/... \
    --out          experiment/results/ptm/ptm_physics.json
```

---

## 核心实验发现

**Token 量化（M=16，8 个数据集）**

RQ M=16 与 QINCo2 M=16 MSE 在 ViDoRe V3 上表现相近（QINCo 均值 Δ ≈ −0.0022，在 ±0.008 NPU 噪声带内）。与 FP16 的差距是容量瓶颈，而非损失函数问题——由三项独立分析共同证实：

1. 分数分解：在 RQ 下，97.4% 的查询 token 已命中正确的 argmax 页面 token；差距由值主导，而非翻转主导。
2. Score-MSE 损失：以 argmax 翻转为目标，但翻转净为正向（`gap_flip = −0.0066`），无改进空间。
3. 特征谱对齐：页面 VQ 分辨率与查询质量分布已充分对齐（有效增益 = 1.02），白化/各向异性 MSE 同样无改进空间。

结论：下一步应提升容量至 M=32，而非调整损失函数。

**FDE 级联（两阶段）**

FDE 召回 → 精确 MaxSim 重排（K=100）将评分代价降低约 16 倍（从 1674 页减至 100 个候选）。在紧凑 FDE 尺寸下（AMS sketch，4–8k 维），级联 nDCG 可超过完整 FP16 MaxSim，原因是 FDE 抑制了 MaxSim 中单 token 峰值页面所利用的假正例模式。

**PTM（Prune-Then-Merge）**

physics（v3）上最优配置 `ptm(k=−1.5, merge_factor=8)`：nDCG@5 = 0.7335，每页 83.8 个 token，FP16 基准为 0.7435（保留率 98.6%）。方差代理与真实 EOS 注意力在该阈值下结果相近，但真实注意力才是正确实现。

---

## 依赖项

**Python 包**（见 [requirements.txt](requirements.txt)）：
`torch`、`torch_npu`、`transformers==4.46.3`、`peft`、`accelerate`、`pyarrow`、`numpy`、`omegaconf`、`Pillow`

**第三方库（已内置于 `experiment/methods/` 下，无需单独安装）：**

| 用途 | 位置 |
|---|---|
| QINCo2 训练核心（`qinco` 包）| [methods/token_quantization/qinco2/qinco/](../methods/token_quantization/qinco2/qinco/) |
| MUVERA FDE 生成器（`fde_generator.py`）| [methods/muvera/](../methods/muvera/) |
| RQ-MoE 模型（`model_rqmoe.py`）| [methods/token_quantization/rq_moe/](../methods/token_quantization/rq_moe/) |
| ColBERT SAR 工具（`colbertsar.py`）| [methods/token_quantization/](../methods/token_quantization/) |

各脚本通过 `--qinco-dir` / `--muvera-dir` / `--rqmoe-dir` 等参数指定这些目录，默认值已指向上表位置，正常情况下无需手动传入。

**`colpali_engine`**（仓库根目录的本地包）通过 `_find_root()` 自动加入 `sys.path`，无需额外配置。
