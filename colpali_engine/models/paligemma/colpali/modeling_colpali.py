from typing import ClassVar, Optional

import torch
from torch import nn
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)
from colpali_engine.merge import Avg1DPoolingMerger, Avg2DPoolingMerger, ClusteringMerger


class ColPali(PaliGemmaPreTrainedModel):
    """
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
            self, 
            config: PaliGemmaConfig,
            pooling_strategy: Optional[str]=None,
            pool_size: int=10,
            dim: int=128,
        ):
        super().__init__(config=config)

        model = PaliGemmaForConditionalGeneration(config=config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model
        # self.model.language_model.to(torch.bfloat16)

        # TODO: Wait for ColPali2 to create a ColPaliConfig to allow specifying the embedding dimension.
        # We could do it now but it would break all the models trying to load the model from the checkpoint.
        self.dim = dim
        self.pool_size = pool_size
        self.pooling_strategy = pooling_strategy

        if self.pooling_strategy in ["post_proj_2dpool", "pre_llm_2dpool", "pre_proj_2dpool"]:
            self.merger = Avg2DPoolingMerger(self.pool_size)
        if self.pooling_strategy in ["post_proj_flatten", "pre_llm_flatten", "pre_proj_flatten"]:
            self.merger = Avg1DPoolingMerger(self.pool_size)
        if self.pooling_strategy in ["post_proj_cluster", "pre_llm_cluster", "pre_proj_cluster"]:
            self.merger = ClusteringMerger(self.pool_size)

        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size, self.dim)
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        patch_ranges = kwargs.pop("patch_ranges", None)
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)

        outputs = self.model(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)
        
        if "pixel_values" in kwargs:
            if self.pooling_strategy=="post_proj_selected":
                from torch.nn.utils.rnn import pad_sequence
                from math import ceil
                indices_list = list()
                for patch_range in patch_ranges:
                    k = min(ceil((patch_range[1]+1)/self.pool_size), patch_range[1]+1-patch_range[0])
                    indices = torch.randperm(patch_range[1]-patch_range[0]+1, device=proj.device)[:k] + patch_range[0]
                    indices_list.append(indices)
                indices_list = pad_sequence(indices_list, batch_first=True, padding_value=0).to(proj.device)
                indices_list = torch.sort(indices_list, dim=1).values
                proj_tail = proj[:, (patch_range[1]+1):]
                proj = proj[torch.arange(proj.size(0), device=proj.device).unsqueeze(-1), indices_list]
                proj = torch.cat([proj, proj_tail], dim=1)
                # adapt attention_mask to the size of proj
                kwargs["attention_mask"] = torch.cat([(indices_list!=0).long(), torch.ones_like(proj_tail)[..., -0]], dim=1)

            if self.pooling_strategy=="post_proj_cluster":
                proj, kwargs["attention_mask"] = self.merger(
                    proj,
                    kwargs["attention_mask"],
                    patch_range_list=patch_ranges,
                )

        # L2 normalization
        proj = (proj+1e-10) / (proj.norm(dim=-1, keepdim=True)+1e-10)  # (batch_size, sequence_length, dim)

        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        return proj

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of=None,
    ) -> nn.Embedding:
        model_embeds = self.model.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.model.vocab_size = model_embeds.num_embeddings

        return model_embeds

    @property
    def patch_size(self) -> int:
        return self.model.vision_tower.config.patch_size
