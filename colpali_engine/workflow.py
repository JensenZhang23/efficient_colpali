"""Small, inference-first helpers shared by the public scripts."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device


MODEL_REGISTRY = {
    "colpali": (ColPali, ColPaliProcessor, "vidore/colpali-v1.2"),
    "colqwen2": (ColQwen2, ColQwen2Processor, "vidore/colqwen2-v1.0"),
}


def move_batch(batch, device: str):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def load_model(model_type: str, model_name: Optional[str] = None, device: str = "auto"):
    """Load a supported retriever and processor on CUDA, Ascend NPU, MPS or CPU."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type must be one of {sorted(MODEL_REGISTRY)}")

    model_class, processor_class, default_name = MODEL_REGISTRY[model_type]
    model_name = model_name or default_name
    device = get_torch_device(device)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Eager attention is slower than CUDA FlashAttention but portable to Ascend.
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    processor = processor_class.from_pretrained(model_name)
    return model, processor, device


def encode(
    values: Iterable,
    model,
    processor,
    device: str,
    kind: str,
    batch_size: int = 2,
) -> list[torch.Tensor]:
    """Encode images or queries and return variable-length embeddings on CPU."""
    values = list(values)
    process = processor.process_images if kind == "images" else processor.process_queries
    loader = DataLoader(ListDataset(values), batch_size=batch_size, collate_fn=process)
    embeddings: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            output = model(**move_batch(batch, device)).cpu()
            embeddings.extend(torch.unbind(output))
    return embeddings


def score_queries(processor, queries, documents, device: str, batch_size: int = 8):
    return processor.score_multi_vector(
        queries,
        documents,
        batch_size=batch_size,
        device=device,
    ).cpu()
