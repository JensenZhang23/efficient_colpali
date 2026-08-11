from math import ceil
from typing import Optional

import torch
from torch import nn


class Avg1DPoolingMerger(nn.Module):
    def __init__(self, pool_size) -> None:
        super().__init__()
        self.pool_size = pool_size
    

    def forward(
        self,
        hidden_states,
        attention_mask,
        patch_range_list: Optional[torch.Tensor]=None,
    ):
        vision_end_token_index =  patch_range_list[0, 1]+1
        hidden_states_tail = hidden_states[:, vision_end_token_index:]
        attention_mask_tail = attention_mask[:, vision_end_token_index:]

        hidden_states_images = hidden_states[:, :vision_end_token_index]
        attention_mask_images = attention_mask[:, :vision_end_token_index]
        pad_length = ceil(vision_end_token_index/self.pool_size)*self.pool_size-vision_end_token_index

        paddings = torch.zeros((hidden_states_images.size(0), pad_length, hidden_states_images.size(-1)), device=hidden_states_images.device, dtype=hidden_states_images.dtype)
        hidden_states_images = torch.cat([paddings, hidden_states_images], dim=1)
        hidden_states_images = hidden_states_images.view(hidden_states_images.size(0),hidden_states_images.size(1)//self.pool_size, self.pool_size, hidden_states_images.size(2))
        hidden_states_images = hidden_states_images.mean(dim=2)

        paddings = torch.zeros(attention_mask_images.size(0), pad_length, device=attention_mask_images.device, dtype=attention_mask_images.dtype)
        attention_mask_images = torch.cat([paddings, attention_mask_images], dim=1)
        attention_mask_images = attention_mask_images.view(attention_mask_images.size(0),attention_mask_images.size(1)//self.pool_size, self.pool_size)
        attention_mask_images = torch.any(attention_mask_images, dim=2)

        hidden_states = torch.cat([hidden_states_images, hidden_states_tail], dim=1)
        attention_mask = torch.cat([attention_mask_images, attention_mask_tail], dim=1)

        return hidden_states, attention_mask