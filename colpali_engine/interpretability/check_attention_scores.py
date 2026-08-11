import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.utils.torch_utils import get_torch_device, find_min_max_indices


def draw_self_attn_map(
    image: Image.Image,
    self_attn_score: torch.Tensor,
    ratio = 0.2,
    patch_size = 28,
):
    """
    Given an image with size (patch_x*patch_size, patch_y*patch_size) and a self-attention score tensor with size (patch_x*patch_y), you are tasked to:
    (1) convert the self_attn_score tensor to a 2D tensor with size (patch_x, patch_y)
    (2) box the patch with the top-ratio% self-attention score
    (3) overlay the boxed image patch on the original image
    (4) return the modified image
    """


if __name__=="__main__":
    torch.manual_seed(42)

    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/hwfile/mllm/mayubo/colqwen2-v1.0/checkpoint-2310"
    # peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad_0129/checkpoint-2310"
    batch_size = 4
    dataset_name_out = "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled"
    max_length = 768
    pooling_strategy = None
    pool_size = 1
    device = get_torch_device("auto")

    # Load the model and processor
    processor = ColQwen2Processor.from_pretrained(
        model_name,
        max_length=max_length,
    )
    model = ColQwen2.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map=device,
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
    test_dataset_out = load_dataset(dataset_name_out)['test']

    self_attn_score_list = list()
    hw_list = list()
    for batch in tqdm(DataLoader(test_dataset_out, batch_size=batch_size, collate_fn=collator)):
        with torch.no_grad():
            outputs = model(**{
                "input_ids": batch["doc_input_ids"].to(device),
                "attention_mask": batch["doc_attention_mask"].to(device),
                "pixel_values": batch["doc_pixel_values"].to(device),
                "image_grid_thw": batch["doc_image_grid_thw"].to(device),
                "output_attentions": True,
            })
            self_attn_scores = outputs[1][-1]   # Last layer attention scores
            image_patch_ranges = find_min_max_indices(batch["doc_input_ids"], 151655)
            self_attn_scores = self_attn_scores.max(dim=1).values
            for image_patch_range, self_attn_score, image_grid_thw in zip(image_patch_ranges, self_attn_scores, batch["doc_image_grid_thw"]):
                self_attn_score = self_attn_score[-2, image_patch_range[0]:(image_patch_range[1]+1)]   # <EOS> token
                self_attn_score_list.append(self_attn_score)
                hw_list.append((image_grid_thw[1]*28, image_grid_thw[2]*28))
    
    for sample, self_attn_score, hw in zip(test_dataset_out, self_attn_score_list, hw_list):
        image_name = sample["image_filename"]
        image = sample["image"]
        height, width = hw
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        
        topk_values, topk_indices = torch.topk(self_attn_score, k=int(len(self_attn_score)*0.1))
        
        # Convert the indices from the flattened array to 2D coordinates (i, j)
        top_patches = []
        for idx in topk_indices:
            idx = idx.item()  # Convert tensor to integer
            i = idx // (width//(28*2))
            j = idx % (width//(28*2))
            top_patches.append((i, j))

        # Create a copy of the image to draw on
        modified_image = image.copy()
        draw = ImageDraw.Draw(modified_image)

        # Draw a red rectangle around each top patch
        for i, j in top_patches:
            x0 = j * (28*2)
            y0 = i * (28*2)
            x1 = x0 + (28*2)
            y1 = y0 + (28*2)
            draw.rectangle([x0, y0, x1, y1], outline='red', width=5)
        modified_image.save("./{}/{}.png".format(dataset_name_out.split("/")[-1], image_name))