from math import sqrt
from typing import Optional

import torch
from torch import nn


class Avg2DPoolingMerger(nn.Module):
    def __init__(self, merge_size) -> None:
        super().__init__()
        self.kernel_size = int(sqrt(merge_size))


    def forward(
        self,
        hidden_states,
        attention_mask,
        patch_range_list: Optional[torch.Tensor]=None,
        patch_indices_list_list: Optional[torch.Tensor]=None,
        remove_index_list_list: Optional[torch.Tensor]=None,
    ):
        max_tokens = 300
        outputs = torch.zeros((hidden_states.size(0), max_tokens, hidden_states.size(-1)), device=hidden_states.device, dtype=hidden_states.dtype)
        outputs_attention = torch.zeros((attention_mask.size(0), max_tokens), device=attention_mask.device, dtype=attention_mask.dtype)

        vision_end_token_index =  patch_range_list[0, 1]+1
        hidden_states_tail = hidden_states[:, vision_end_token_index:]
        attention_mask_tail = attention_mask[:, vision_end_token_index:]
        
        if remove_index_list_list is None:
            remove_index_list_list = [None for _ in range(hidden_states.size(0))]
        for i, (patch_range, hidden_state, patch_indices_list, remove_index_list) in enumerate(zip(patch_range_list, hidden_states, patch_indices_list_list, remove_index_list_list)):
            start, end = patch_range
            hidden_state = hidden_state[start:(end+1)]

            mask = (patch_indices_list!=-1)
            if remove_index_list is not None:
                remove_index_list = remove_index_list[remove_index_list!=-1]
                mask = mask & ~torch.isin(patch_indices_list, remove_index_list)
            patch_indices_list = patch_indices_list[torch.any(mask, dim=-1)]
            mask = mask[torch.any(mask, dim=-1)]

            hidden_state_reshaped = torch.sum(hidden_state[patch_indices_list], dim=1)/mask.sum(dim=1).unsqueeze(-1)
            print("hidden_state: {}; hidden_state_reshaped: {}; patch_indices_list: {}".format(
                hidden_state.size(), hidden_state_reshaped.size(), patch_indices_list.size())
            )
            outputs[i, -hidden_state_reshaped.size(0):] = hidden_state_reshaped
            outputs_attention[i, -hidden_state_reshaped.size(0):] = 1

        outputs = torch.cat([outputs, hidden_states_tail], dim=1)
        outputs_attention = torch.cat([outputs_attention, attention_mask_tail], dim=1)
        print(outputs.size(), outputs_attention.size())
        return outputs, outputs_attention