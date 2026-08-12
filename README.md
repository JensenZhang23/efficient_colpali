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

## 安装

Python 3.9+：

```bash
pip install -e ".[train]"
```

默认代码不依赖 FlashAttention、bitsandbytes 或 NVIDIA NVML，因此普通 CUDA、CPU 与 Ascend 环境共用同一套入口。

### Ascend NPU

先按照昇腾版本矩阵安装驱动、CANN、匹配版本的 PyTorch 和 `torch_npu`，再安装本项目。验证环境：

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

脚本中的 `--device auto` 按 NPU、CUDA、MPS、CPU 的顺序自动选择设备。为保证可移植性，模型默认使用 `eager` attention。

## 1. 获取数据

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
