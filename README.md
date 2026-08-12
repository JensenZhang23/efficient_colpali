# Efficient ColPali：最小训练与检索工作流

这是一个以推理为主的视觉文档检索项目，仅对外保留 **ColPali** 和 **ColQwen2** 两种多向量模型，以及数据获取、训练、推理和评测四个入口。文档页面无需 OCR，模型直接编码图片，并通过 ColBERT MaxSim 与文本查询匹配。

## 文件结构

| 功能 | 入口或实现 |
| --- | --- |
| 下载训练/评测数据 | [scripts/download_data.py](scripts/download_data.py) |
| LoRA 训练 | [scripts/train.py](scripts/train.py) |
| 本地图片检索 | [scripts/infer.py](scripts/infer.py) |
| 检索评测 | [scripts/evaluate.py](scripts/evaluate.py) |
| 模型加载、批量编码和 MaxSim | [colpali_engine/workflow.py](colpali_engine/workflow.py) |
| ColPali 模型与处理器 | [modeling_colpali.py](colpali_engine/models/paligemma/colpali/modeling_colpali.py)、[processing_colpali.py](colpali_engine/models/paligemma/colpali/processing_colpali.py) |
| ColQwen2 模型与处理器 | [modeling_colqwen2.py](colpali_engine/models/qwen2/colqwen2/modeling_colqwen2.py)、[processing_colqwen2.py](colpali_engine/models/qwen2/colqwen2/processing_colqwen2.py) |
| 训练 batch | [visual_retriever_collator.py](colpali_engine/collators/visual_retriever_collator.py) |
| ColBERT 损失 | [late_interaction_losses.py](colpali_engine/loss/late_interaction_losses.py) |
| Hugging Face Trainer 扩展 | [contrastive_trainer.py](colpali_engine/trainer/contrastive_trainer.py) |
| CPU/CUDA/Ascend 设备选择 | [torch_utils.py](colpali_engine/utils/torch_utils.py) |

常用自动化脚本：

| 脚本 | 用途 |
| --- | --- |
| [download_assets.sh](scripts/download_assets.sh) | 缓存两种模型的推理/训练权重并获取数据集 |
| [train_colqwen2.sh](scripts/train_colqwen2.sh) | 推荐的 ColQwen2 LoRA 训练设置 |
| [train_colpali.sh](scripts/train_colpali.sh) | ColPali LoRA 训练设置 |
| [evaluate_models.sh](scripts/evaluate_models.sh) | 依次评测两个官方 checkpoint |
| [smoke_test.sh](scripts/smoke_test.sh) | 8 条评测样本与 16 条训练样本的端到端检查 |
| [env.sh](scripts/env.sh) | 路径、设备和 batch size 的公共环境变量 |

## 安装

Python 3.9+：

```bash
pip install -e ".[train]"
```

如需将训练和验证曲线上报到 Weights & Biases：

```bash
pip install -e ".[train,monitor]"
wandb login
```

默认代码不依赖 FlashAttention、bitsandbytes 或 NVIDIA NVML，因此普通 CUDA、CPU 与 Ascend 环境共用同一套入口。

### Ascend NPU

先按照昇腾版本矩阵安装驱动、CANN、匹配版本的 PyTorch 和 `torch_npu`，再安装本项目。验证环境：

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

脚本中的 `--device auto` 按 NPU、CUDA、MPS、CPU 的顺序自动选择设备。为保证可移植性，模型默认使用 `eager` attention。

## 1. 获取数据

一次性获取权重和数据：

```bash
bash scripts/download_assets.sh
```

权重写入项目内的 `.cache/huggingface/`，评测集和训练集写入 `data_dir/`。训练集约 41 GB；如果只做推理和评测，可以跳过它：

```bash
DOWNLOAD_TRAIN_DATA=0 bash scripts/download_assets.sh
```

评测集较小，建议先下载它验证流程：

```bash
python scripts/download_data.py --dataset eval
```

输出到 `data_dir/eval/`，对应公开数据集 `vidore/docvqa_test_subsampled`。

完整训练集较大，确认磁盘空间后下载：

```bash
python scripts/download_data.py --dataset train
```

输出到 `data_dir/train/`，对应 `vidore/colpali_train_set`。也可不预下载，后续脚本直接填写 Hugging Face 数据集 ID，由 `datasets` 自动缓存。

## 2. 推理

ColQwen2 是默认模型。输入一个查询和若干页面图片，输出按相关度降序排列的结果：

```bash
python scripts/infer.py \
  --model-type colqwen2 \
  --query "What is the total revenue?" \
  --images page1.png page2.png page3.png
```

使用 ColPali：

```bash
python scripts/infer.py \
  --model-type colpali \
  --query "What is the total revenue?" \
  --images page1.png page2.png
```

默认 checkpoint：

- ColPali：`vidore/colpali-v1.2`
- ColQwen2：`vidore/colqwen2-v1.0`

加载本地训练结果或其他兼容 checkpoint：

```bash
python scripts/infer.py \
  --model-type colqwen2 \
  --model-name outputs/retriever \
  --query "query text" \
  --images page1.png page2.png
```

可用 `--device npu:0`、`--device cuda:0` 或 `--device cpu` 显式指定设备。

## 3. 评测

使用推荐设置依次评测两个官方模型：

```bash
bash scripts/evaluate_models.sh
```

低成本验证：

```bash
EVAL_LIMIT=16 EVAL_BATCH_SIZE=1 bash scripts/evaluate_models.sh
```

先用少量样本进行 smoke test：

```bash
python scripts/evaluate.py \
  --model-type colqwen2 \
  --dataset data_dir/eval \
  --limit 32
```

完整在线评测：

```bash
python scripts/evaluate.py \
  --model-type colpali \
  --dataset vidore/docvqa_test_subsampled \
  --output results.json
```

评测会对物理页面去重，编码全部 query/page，并输出 `Recall@K`、`MRR@K` 和 `NDCG@K`。指标实现直接位于 [scripts/evaluate.py](scripts/evaluate.py)。

## 4. 最小 LoRA 训练

推荐直接使用自适应训练脚本：

```bash
# 单卡
bash scripts/train_colqwen2.sh

# 8 卡 CUDA 或 Ascend；自动将有效 batch size 调整到约 256
NUM_PROCESSES=8 bash scripts/train_colqwen2.sh
```

ColQwen2 默认有效 batch size 为 256；ColPali 使用更节省资源的 32。ColPali 对应执行 `bash scripts/train_colpali.sh`。显存不足时可设置 `PER_DEVICE_BATCH_SIZE=1`，也可用 `TARGET_BATCH_SIZE` 覆盖默认值。所有变量见 [env.sh](scripts/env.sh)。

训练默认保留最多 500 条验证样本，每个 epoch 计算并在终端打印 `eval_loss`，保存一次 checkpoint，并在训练结束时恢复 `eval_loss` 最低的 checkpoint。可通过 `EVAL_SAMPLES` 调整验证规模：

```bash
EVAL_SAMPLES=1000 bash scripts/train_colqwen2.sh
```

启用 W&B 监控：

```bash
WANDB_PROJECT=efficient-colpali \
REPORT_TO=wandb \
RUN_NAME=colqwen2-lora-exp01 \
NUM_PROCESSES=8 \
bash scripts/train_colqwen2.sh
```

W&B 会记录 `loss`、`eval_loss`、学习率、epoch、训练吞吐和运行时间。完整的 Recall/MRR/NDCG 需要编码整个验证语料，成本明显高于验证 loss，因此仍通过 [evaluate_models.sh](scripts/evaluate_models.sh) 在训练后运行；评测本地 checkpoint 时设置 `COLQWEN2_MODEL=outputs/colqwen2-lora`。

先用少量数据确认环境和显存：

```bash
python scripts/train.py \
  --model-type colqwen2 \
  --dataset data_dir/train \
  --output-dir outputs/colqwen2-smoke \
  --max-samples 32 \
  --batch-size 1
```

完整训练示例：

```bash
python scripts/train.py \
  --model-type colqwen2 \
  --dataset vidore/colpali_train_set \
  --output-dir outputs/colqwen2 \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --epochs 1
```

ColPali 只需将 `--model-type` 改为 `colpali`。训练入口使用：

- LoRA：`r=32`、`alpha=32`，训练语言模型中的注意力/MLP 投影，并保存 `custom_text_proj` 检索投影层；
- ColBERT pairwise loss 和 batch 内负样本；
- BF16（加速设备）与 gradient checkpointing；
- `eager` attention，以兼容 CUDA 和 Ascend。

默认训练基座为 `vidore/colpaligemma-3b-pt-448-base` 或 `vidore/colqwen2-base`，可通过 `--model-name` 替换。ColPali 的 PaliGemma 基座可能要求先在 Hugging Face 登录并接受 Gemma 使用协议：

```bash
hf auth login
```

多卡训练可继续使用 Hugging Face Accelerate：

```bash
accelerate config
accelerate launch scripts/train.py --model-type colqwen2 --dataset data_dir/train
```

Ascend 多卡需要环境中的 Accelerate/PyTorch 使用 HCCL，建议先依次完成单条推理、单卡评测和单卡 smoke training，再启动多卡任务。

## 数据格式

训练数据至少包含：

```text
image: PIL image
query: text
```

评测数据还需要页面标识字段 `image_filename`，也兼容 `docId`。相同页面可以对应多条查询，评测脚本会只编码一次页面。

## 测试

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

项目基于 [ColPali](https://arxiv.org/abs/2407.01449)，引用信息见 [CITATION.cff](CITATION.cff)，许可证见 [LICENSE](LICENSE)。

## 快速入手

```bash
DOWNLOAD_TRAIN_DATA=0 bash scripts/download_assets.sh
bash scripts/evaluate_models.sh
```