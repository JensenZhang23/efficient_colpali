import gc
import logging
from typing import List, TypeVar

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
T = TypeVar("T")


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

    return device


def tear_down_torch():
    """
    Teardown for PyTorch.
    Clears GPU cache for both CUDA and MPS.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]
    

def find_min_max_indices(input_ids, k):
    # input_ids 的形状为 [N, d]
    # 创建一个与 input_ids 形状相同的布尔矩阵，其中 input_ids 等于 k 的位置为 True
    mask = (input_ids == k)
    
    # 使用 torch.nonzero 找到所有等于 k 的元素的索引
    # nonzero_indices 的形状为 [num_occurrences, 2]，其中 num_occurrences 是 k 出现的总次数
    nonzero_indices = torch.nonzero(mask, as_tuple=False)
    
    # 初始化 outputs 为一个全零的 [N, 2] 形状的张量
    outputs = torch.zeros((input_ids.size(0), 2), dtype=torch.long)
    
    # 对于每个样本 i，找到最小和最大的索引
    for i in range(input_ids.size(0)):
        # 找到第 i 行中所有等于 k 的元素的索引
        indices = nonzero_indices[nonzero_indices[:, 0] == i, 1]
        if indices.numel() > 0:  # 如果存在等于 k 的元素
            # 将最小索引赋值给 outputs[i][0]
            outputs[i][0] = torch.min(indices)
            # 将最大索引赋值给 outputs[i][1]
            outputs[i][1] = torch.max(indices)
    
    return outputs