from math import ceil
from typing import Optional, Tuple, Dict
from scipy.cluster.vq import kmeans, vq
from scipy.cluster.hierarchy import fcluster, linkage

import torch
from torch import nn


class ClusteringMerger(nn.Module):
    def __init__(self, merge_size, cluster_type="hierarchical") -> None:
        super().__init__()
        self.pool_size = merge_size
        self.cluster_type = cluster_type


    def pool_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Return the pooled embeddings and the mapping from cluster id to token indices.
        Input:
        - embeddings: tensor of shape (token_length, embedding_dim)
        Output:
        - pooled_embeddings: tensor of shape (num_clusters, embedding_dim)
        NOTE: This method doesn't support batched inputs because:
        - the sequence lengths can be different.
        - scipy doesn't support batched inputs.
        """

        pooled_embeddings = []
        token_length = embeddings.size(0)
        max_clusters = max(token_length // self.pool_size, 1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        if self.cluster_type == "hierarchical":
            similarities = torch.mm(embeddings, embeddings.t())
            distances = 1 - similarities
            Z = linkage(distances.to(torch.float16).clone().detach().cpu().numpy(), metric="euclidean", method="ward")  # noqa: N806
            cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")
        elif self.cluster_type == "kmeans":
            centroids, _ = kmeans(embeddings.to(torch.float).cpu().numpy(), max_clusters)
            cluster_labels, _ = vq(embeddings.to(torch.float).cpu().numpy(), centroids)
        else:
            raise ValueError(f"Invalid cluster type: {self.cluster_type}")
        cluster_id_to_indices: Dict[int, torch.Tensor] = {}
        for cluster_id in range(1, max_clusters + 1):
            cluster_indices = torch.where(torch.tensor(cluster_labels == cluster_id))[0]
            cluster_id_to_indices[cluster_id] = cluster_indices
            if cluster_indices.numel() > 0:
                pooled_embedding = embeddings[cluster_indices].mean(dim=0)
                pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)
                pooled_embeddings.append(pooled_embedding)
        pooled_embeddings = torch.stack(pooled_embeddings, dim=0)
        
        return pooled_embeddings, cluster_id_to_indices

    def forward(
        self,
        hidden_states,
        attention_mask,
        patch_range_list: Optional[torch.Tensor]=None,
    ):
        vision_end_token_index = patch_range_list[0, 1]+1
        hidden_states_tail = hidden_states[:, (vision_end_token_index+1):-1]
        attention_mask_tail = torch.ones((hidden_states_tail.size(0), hidden_states_tail.size(1)), device=hidden_states_tail.device, dtype=hidden_states_tail.dtype)

        max_tokens = ceil(hidden_states.size(1)/self.pool_size)
        new_hidden_states = torch.zeros((hidden_states.size(0), max_tokens, hidden_states.size(-1)), device=hidden_states.device, dtype=hidden_states.dtype)
        new_attention_mask = torch.zeros((hidden_states.size(0), max_tokens), device=hidden_states.device, dtype=hidden_states.dtype)
        for i, (p_embed, patch_range) in enumerate(zip(hidden_states, patch_range_list)):
            p_embed = p_embed[patch_range[0]:patch_range[1]+1]
            p_embed, _ = self.pool_embeddings(p_embed)
            new_hidden_states[i, -p_embed.size(0):] = p_embed
            new_attention_mask[i, -p_embed.size(0):] = 1
        new_hidden_states = torch.cat([new_hidden_states, hidden_states_tail], dim=1)
        new_attention_mask = torch.cat([new_attention_mask, attention_mask_tail], dim=1)
        return new_hidden_states, new_attention_mask


    # def forward(
    #     self,
    #     hidden_states,
    #     attention_mask,
    #     patch_range_list: Optional[torch.Tensor]=None,
    # ):
    #     print("2333")
    #     max_tokens = ceil(hidden_states.size(1)/self.pool_size)
    #     new_hidden_states = torch.zeros((hidden_states.size(0), max_tokens, hidden_states.size(-1)), device=hidden_states.device, dtype=hidden_states.dtype)
    #     new_attention_mask = torch.zeros((hidden_states.size(0), max_tokens), device=hidden_states.device, dtype=hidden_states.dtype)
    #     for i, (p_embed, patch_range) in enumerate(zip(hidden_states, patch_range_list)):
    #         p_embed, _ = self.pool_embeddings(p_embed)
    #         new_hidden_states[i, -p_embed.size(0):] = p_embed
    #         new_attention_mask[i, -p_embed.size(0):] = 1
    #     return new_hidden_states, new_attention_mask