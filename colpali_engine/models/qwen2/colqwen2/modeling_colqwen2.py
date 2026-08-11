from math import ceil
from typing import ClassVar, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from colpali_engine.merge import Avg1DPoolingMerger, Avg2DPoolingMerger, ClusteringMerger


class MyVisonEncoder(Qwen2VisionTransformerPretrainedModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, ):
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        return self.merger(hidden_states)


class ColQwen2(Qwen2VLForConditionalGeneration):
    """
    ColQwen2 model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(
        self, 
        config: Qwen2VLConfig,
        pooling_strategy: Optional[str]=None,
        clustering_strategy: Optional[str]="hierarchical",
        pool_size: int=10,
        dim: int=128,
    ):
        super().__init__(config=config)
        self.dim = dim
        self.pool_size = pool_size
        self.pooling_strategy = pooling_strategy
        self.clustering_strategy = clustering_strategy
        # self.visual = MyVisonEncoder(config.vision_config)

        if self.pooling_strategy in ["post_proj_2dpool", "pre_llm_2dpool", "pre_proj_2dpool"]:
            self.merger = Avg2DPoolingMerger(self.pool_size)
        if self.pooling_strategy in ["post_proj_flatten", "pre_llm_flatten", "pre_proj_flatten"]:
            self.merger = Avg1DPoolingMerger(self.pool_size)
        if self.pooling_strategy in ["post_proj_cluster", "pre_llm_cluster", "pre_proj_cluster"]:
            self.merger = ClusteringMerger(self.pool_size, self.clustering_strategy)
        
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.post_init()


    def inner_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            patch_ranges: Optional[torch.Tensor]=None,
            patch_indices_list: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if pixel_values is not None and self.pooling_strategy=="pre_llm_flatten":
            inputs_embeds, attention_mask = self.merger(
                inputs_embeds,
                attention_mask,
                patch_range_list=patch_ranges,
            )
            position_ids = None

        if pixel_values is not None and self.pooling_strategy=="pre_llm_2dpool":
            inputs_embeds, attention_mask = self.merger(
                inputs_embeds,
                attention_mask,
                patch_range_list=patch_ranges,
                patch_indices_list_list=patch_indices_list,
            )
            position_ids = None

        if pixel_values is not None and self.pooling_strategy=="pre_llm_cluster":
            inputs_embeds, attention_mask = self.merger(
                inputs_embeds,
                attention_mask,
                patch_range_list=patch_ranges,
                patch_indices_list_list=patch_indices_list,
            )
            position_ids = None

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if output_attentions:
            hidden_states, self_attns = outputs[0], outputs[-1]
            return hidden_states, self_attns
        else:
            hidden_states = outputs[0]
            return hidden_states, attention_mask


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Delete output_hidden_states from kwargs
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("remove_index_list", None)

        # The following code is a hack to make sure the scatter in DDP is done correctly when training on multiple GPUs
        if "pixel_values" in kwargs:
            # compute pixel_values offsets
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pv[:o] for pv, o in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        position_ids, rope_deltas = self.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            video_grid_thw=None,
            attention_mask=kwargs.get("attention_mask", None),
        )
        if kwargs.get("output_attentions", False):
            last_hidden_states, self_attns = self.inner_forward(*args,
                                    **kwargs,
                                    position_ids=position_ids,
                                    use_cache=False,
                                    output_hidden_states=True)  # (batch_size, sequence_length, hidden_size)
        else:
            last_hidden_states, kwargs["attention_mask"] = self.inner_forward(*args,
                                    **kwargs,
                                    position_ids=position_ids,
                                    use_cache=False,
                                    output_hidden_states=True)  # (batch_size, sequence_length, hidden_size)

        if "pixel_values" in kwargs and self.pooling_strategy in ["pre_proj_flatten", "pre_proj_2dpool", "pre_proj_cluster"]:
            if self.pooling_strategy in ["pre_proj_2dpool"]:
                last_hidden_states, kwargs["attention_mask"] = self.merger(
                    last_hidden_states,
                    kwargs["attention_mask"],
                    patch_range_list=kwargs["patch_ranges"],
                    patch_indices_list_list=kwargs["patch_indices_list"],
                )
            elif self.pooling_strategy in ["pre_proj_flatten",  "pre_proj_cluster"]:
                last_hidden_states, kwargs["attention_mask"] = self.merger(
                    last_hidden_states,
                    kwargs["attention_mask"],
                    patch_range_list=kwargs["patch_ranges"],
                )
            else:
                raise("Pooling strategy not supported")
        
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.pooling_strategy in ["post_proj_flatten", "post_proj_2dpool", "post_proj_cluster", "post_proj_selected"]:
            if self.pooling_strategy=="post_proj_selected":
                from torch.nn.utils.rnn import pad_sequence
                indices_list = list()
                for patch_range in kwargs["patch_ranges"]:
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
            elif self.pooling_strategy in ["post_proj_flatten", "post_proj_cluster"]:
                proj, kwargs["attention_mask"] = self.merger(
                    proj,
                    kwargs["attention_mask"],
                    patch_range_list=kwargs["patch_ranges"]
                )
            elif self.pooling_strategy=="post_proj_2dpool":
                proj, kwargs["attention_mask"] = self.merger(
                    proj,
                    kwargs["attention_mask"],
                    patch_range_list=kwargs["patch_ranges"],
                    patch_indices_list_list=kwargs["patch_indices_list"],
                )
            else:
                raise("Pooling strategy not supported")
        
        # L2 normalization
        proj = (proj+1e-10) / (proj.norm(dim=-1, keepdim=True)+1e-10)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        print(proj.size(), proj.device)

        if kwargs.get("output_attentions", False):
            return proj, self_attns
        else:
            return proj

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size