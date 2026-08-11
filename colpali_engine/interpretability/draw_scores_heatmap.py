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
    show_colorbar: bool = False,
    top_k_percent: float = 0.05,
    highlight_style: str = "red_box",  # Options: "red_box", "hatch"
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

    width, height = image.size
    if width < height:
        figsize = (8, 8 * height / width)
    else:
        figsize = (8 * width / height, 8)

    # Create the figure
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(img_array)
        # im = ax.imshow(
        #     similarity_map_image,
        #     cmap=sns.color_palette("mako", as_cmap=True),
        #     alpha=0.5,
        # )

        # Calculate the threshold for top-K%
        highlight_threshold = np.percentile(similarity_map_array.flatten(), 100*(1-top_k_percent))
        top_mask = similarity_map_array >= highlight_threshold
        # Highlight top-K% regions
        if highlight_style == "red_box":
            # Calculate the scaling factor between similarity map and image
            scale_x = image.size[0] / similarity_map_array.shape[1]
            scale_y = image.size[1] / similarity_map_array.shape[0]

            # Draw red boxes around top-K% regions
            for i in range(top_mask.shape[0]):
                for j in range(top_mask.shape[1]):
                    if top_mask[i, j]:
                        rect = plt.Rectangle((j*scale_x, i*scale_y), scale_x, scale_y, fill=False, edgecolor='red', linewidth=2)
                        ax.add_patch(rect)
        elif highlight_style == "hatch":
            # Draw hatched regions for top-K% areas
            ax.imshow(top_mask, cmap='Reds', alpha=0.3, extent=(0, similarity_map_array.shape[1], similarity_map_array.shape[0], 0))

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
    test_dataset = test_dataset.select(range(20))

    ps, range_list, hw_list = compute_passage_embeddings(test_dataset, collator)
    max_ps_len = max([len(p) for p in ps])
    range_list = [max_ps_len-len(p)+ranges for p, ranges in zip(ps, range_list)]
    all_normalized_scores, all_normalized_scores_neg = list(), list()
    # Left-pad
    ps = torch.stack([torch.cat([torch.zeros(max_ps_len-len(p), p.shape[-1], dtype=p.dtype), p], dim=0) for p in ps])

    with open("./expanded_questions/{}.json".format(args.dataset_name)) as f:
        questions_dict = json.load(f)
    
    for i, (p_embed, ranges, hw) in enumerate(tqdm(zip(ps, range_list, hw_list))):
        if str(i) not in questions_dict:    # Skip the patches that are not in the expanded questions.
            continue
        questions = questions_dict[str(i)]  # The first question is the original question. Remove it.
        batch_queries = processor.process_queries(questions)
        q_embed = model(**{k:v.to(device) for k, v in batch_queries.items()})
        patch_scores = torch.einsum("nd,csd->cns", p_embed.to(device), q_embed).max(dim=-1).values
        patch_scores = patch_scores[:, ranges[0]:ranges[1]+1]
        
        for question, patch_score in zip(questions, patch_scores):
            patch_score_2d = patch_score.view(*hw)
            image = test_dataset[i]["image"].resize((hw[1]*56, hw[0]*56))
            fig, ax = plot_similarity_map(image, patch_score_2d, show_colorbar=False)
            if not os.path.exists("./contrastive_visualized_scores/{}_{}".format(args.dataset_name, i)):
                os.makedirs("./contrastive_visualized_scores/{}_{}".format(args.dataset_name, i))
            fig.savefig("./contrastive_visualized_scores/{}_{}/{}.png".format(args.dataset_name, i, question.split("/")[0]))