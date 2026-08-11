import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import json
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, concatenate_datasets

import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.utils.torch_utils import get_torch_device, find_min_max_indices


@torch.no_grad()
def compute_query_embeddings(test_dataset, collator):
    idx_with_query = [idx for idx, sample in enumerate(test_dataset["query"]) if sample is not None]
    dataloader_with_query = DataLoader(
        test_dataset.select(idx_with_query),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    qs = []
    for batch in tqdm(dataloader_with_query):
        if "query_input_ids" in batch:
            query = model(
                input_ids=batch["query_input_ids"].to(device),
                attention_mask=batch["query_attention_mask"].to(device),
            )
            # variable len
            qs.extend(list(torch.unbind(query.to("cpu"))))
    return qs


@torch.no_grad()
def compute_passage_embeddings(test_dataset, collator):
    idx_with_query = [idx for idx, sample in enumerate(test_dataset["query"]) if sample is not None]
    idx_without_query = [idx for idx, sample in enumerate(test_dataset["query"]) if sample is None]
    dataloader_with_query = DataLoader(
        test_dataset.select(idx_with_query),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    dataloader_without_query = DataLoader(
        test_dataset.select(idx_without_query),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # dataset is ordered so that non-null queries come first
    test_dataset = concatenate_datasets(
        [test_dataset.select(idx_with_query), test_dataset.select(idx_without_query)]
    )

    ps = list()
    patch_range_list = list()
    hw_list = list()
    for dataloader in [dataloader_with_query, dataloader_without_query]:
        for batch in tqdm(dataloader):
            # feed only kwargs with 'doc_' prefix
            batch_inputs = {k[4:]: v.to(device) for k, v in batch.items() if k.startswith("doc")}
            doc = model(**batch_inputs)
            ps.extend(list(torch.unbind(doc.to("cpu"))))
            image_patch_ranges = find_min_max_indices(batch["doc_input_ids"], 151655)
            patch_range_list.extend(list(torch.unbind(image_patch_ranges.to("cpu"))))
            hw = batch["doc_image_grid_thw"][:, 1:]//2
            hw_list.extend(list(torch.unbind(hw.to("cpu")))) 
    return ps, patch_range_list, hw_list


def plot_similarity_map(
    image: Image.Image,
    similarity_map: torch.Tensor,
    figsize: Tuple[int, int] = (8, 8),
    show_colorbar: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot and overlay a similarity map over the input image.

    A similarity map is a 2D tensor where each element (i, j) represents the similarity score between a chosen query
    token and the associated image patch at position (i, j). Thus, the higher the similarity score, the brighter the
    color of the patch.

    To show the returned similarity map, use:

    ```python
    >>> fig, ax = plot_similarity_map(image, similarity_map)
    >>> fig.show()
    ```

    Args:
        image: PIL image
        similarity_map: tensor of shape (n_patches_x, n_patches_y)
        figsize: size of the figure
        show_colorbar: whether to show a colorbar
    """

    # Convert the image to an array
    img_array = np.array(image.convert("RGBA"))  # (height, width, channels)

    # Normalize the similarity map and convert it to Pillow image
    similarity_map = (similarity_map-similarity_map.min())/(similarity_map.max()-similarity_map.min())
    similarity_map_array = similarity_map.to(torch.float32).cpu().numpy() # (n_patches_x, n_patches_y)
    similarity_map_image = Image.fromarray((similarity_map_array * 255).astype("uint8")).resize(image.size, Image.Resampling.NEAREST)

    # Create the figure
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(img_array)
        im = ax.imshow(
            similarity_map_image,
            cmap=sns.color_palette("mako", as_cmap=True),
            alpha=0.5,
        )

        if show_colorbar:
            fig.colorbar(im)
        ax.set_axis_off()
        fig.tight_layout()

    return fig, ax


if __name__=="__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="docvqa")
    args = parser.parse_args()

    if not os.path.exists("./visualized_scores/{}".format(args.dataset_name)):
        os.makedirs("./visualized_scores/{}".format(args.dataset_name))

    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/hwfile/mllm/mayubo/colqwen2-v1.0/checkpoint-2310"
    pooling_strategy = None
    pool_size = 1

    max_length = 768
    batch_size = 4
    dataset_name = {
        "docvqa": "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled",
        "arxivqa": "/mnt/petrelfs/mayubo/colpali/data_dir/arxivqa_test_subsampled",
        "infovqa": "/mnt/petrelfs/mayubo/colpali/data_dir/infovqa_test_subsampled",
        "tabfquad": "/mnt/petrelfs/mayubo/colpali/data_dir/tabfquad_test_subsampled",
        "tatdqa": "/mnt/petrelfs/mayubo/colpali/data_dir/tatdqa_test",
        "shift": "/mnt/petrelfs/mayubo/colpali/data_dir/shiftproject_test",
    }[args.dataset_name]
    device = get_torch_device("auto")

    # Load the model and processor
    processor = ColQwen2Processor.from_pretrained(
        model_name,
        max_length=max_length,
    )
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        pooling_strategy=pooling_strategy,
        pool_size=pool_size,
    ).eval()
    model.load_adapter(peft_name)
    if "custom_text_proj.pt" in os.listdir(peft_name):
        proj_state_dict = torch.load(
            os.path.join(peft_name, "custom_text_proj.pt")
        )
        try:
            model.custom_text_proj.base_layer.weight = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.weight'])
            model.custom_text_proj.base_layer.bias = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.bias'])
        except:
            model.custom_text_proj.weight = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.weight'])
            model.custom_text_proj.bias = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.bias'])
    
    collator = VisualRetrieverCollator(
        processor=processor,
        max_length=max_length,
        pooling_strategy=pooling_strategy,
        pool_size=pool_size,
    )
    test_dataset = load_dataset(dataset_name)['test']
    test_dataset = test_dataset.select(range(50))

    ps, range_list, hw_list = compute_passage_embeddings(test_dataset, collator)
    qs = compute_query_embeddings(test_dataset, collator)

    max_ps_len = max([len(p) for p in ps])
    range_list = [max_ps_len-len(p)+ranges for p, ranges in zip(ps, range_list)]
    all_normalized_scores, all_normalized_scores_neg = list(), list()
    # Left-pad
    ps = torch.stack([torch.cat([torch.zeros(max_ps_len-len(p), p.shape[-1], dtype=p.dtype), p], dim=0) for p in ps])
    scores_pos = torch.einsum("bnd,bsd->bns", ps, pad_sequence(qs, batch_first=True))
    patch_scores = scores_pos.max(dim=-1).values

    similarity_maps = ps@ps.transpose(1, 2)
    for similarity_map in similarity_maps:
        valid_index = similarity_map.sum(dim=1)!=0
        similarity_map = similarity_map - torch.eye(similarity_map.shape[0], device=similarity_map.device) * 1e6
        # Drop all-zero rows
        similarity_map = similarity_map[valid_index]
        # For each row, count the values larger than 0.9
        print("sim_num: {}".format((similarity_map>0.9).sum(dim=1)))
        


    neg_type = "hard"
    if neg_type=="hard":
        scores_colpali = torch.einsum("bnd,csd->bcns", pad_sequence(qs, batch_first=True), ps).max(dim=-1).values.sum(dim=2)
        scores_colpali = scores_colpali - torch.eye(scores_colpali.shape[0], device=scores_colpali.device) * 1e6
        neg_page_indices = scores_colpali.max(dim=-1).indices
        scores_neg = torch.einsum("bnd,bsd->bns", ps[neg_page_indices], pad_sequence(qs, batch_first=True))
    else:
        scores_neg = torch.einsum("bnd,bsd->bns", ps.roll(10, dims=0), pad_sequence(qs, batch_first=True))
        neg_page_indices = torch.arange(len(ps)).to(scores_neg.device).roll(10)
    patch_scores_neg = scores_neg.max(dim=-1).values
    range_list_neg = [range_list[i] for i in neg_page_indices]

    rank_list = list()
    sim_num_list_90, sim_num_list_95 = list(), list()
    for patch_score, patch_score_neg, ranges, ranges_neg, hw, data in zip(patch_scores, patch_scores_neg, range_list, range_list_neg, hw_list, test_dataset):
        patch_score = patch_score[ranges[0]:ranges[1]+1]
        patch_score_neg = patch_score_neg[ranges_neg[0]:ranges_neg[1]+1]
        patch_score_2d = patch_score.view(*hw)
        image = data["image"].resize((hw[1]*56, hw[0]*56))
        # fig, ax = plot_similarity_map(image, patch_score_2d, show_colorbar=False)
        # fig.savefig("./visualized_scores/{}/{}.png".format(args.dataset_name, data["query"].split("/")[0]))

        score_max = max(patch_score.max(), patch_score_neg.max())
        score_min = min(patch_score.min(), patch_score_neg.min())
        normalized_score = (patch_score-score_min)/(score_max-score_min)
        print("pos: {}".format(normalized_score.topk(10).values))
        all_normalized_scores.extend(normalized_score.float().tolist())

        normalized_score_neg = (patch_score_neg-score_min)/(score_max-score_min)
        print("neg: {}".format(normalized_score_neg.topk(10).values))
        all_normalized_scores_neg.extend(normalized_score_neg.float().tolist())

        rank = sum([patch_score_neg.max()<score for score in patch_score])
        rank_list.append(rank)

        sim_num_90 = sum([score>0.9 for score in normalized_score])
        sim_num_list_90.append(sim_num_90)
        sim_num_95 = sum([score>0.95 for score in normalized_score])
        sim_num_list_95.append(sim_num_95)
    
    all_normalized_scores = [round(score, 3) for score in all_normalized_scores]
    with open("./normalized_score/{}_pos.json".format(args.dataset_name), "w") as f:
        json.dump(all_normalized_scores, f)

    all_normalized_scores_neg = [round(score, 3) for score in all_normalized_scores_neg]
    with open("./normalized_score/{}_neg_{}.json".format(args.dataset_name, neg_type), "w") as f:
        json.dump(all_normalized_scores_neg, f)
    
    print("Rank: {}".format(sum(rank_list)/len(rank_list)))
