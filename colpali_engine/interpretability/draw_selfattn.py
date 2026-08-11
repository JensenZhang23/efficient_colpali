import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from colpali_engine.interpretability import get_attn_maps, plot_one_similarity_map
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device


if __name__=="__main__":
    torch.manual_seed(42)

    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad_1207/checkpoint-2310"
    dataset_name = "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled"
    layer_num_list = [0, 8, 12, 16, 20, 24, 26, 27]

    if not os.path.exists(
        os.path.join("/mnt/petrelfs/mayubo/colpali/colpali_engine/interpretability", dataset_name.split("/")[-1])
    ):
        os.makedirs(os.path.join("/mnt/petrelfs/mayubo/colpali/colpali_engine/interpretability", dataset_name.split("/")[-1]))

    device = get_torch_device("auto")
    # Load the model
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    model.load_adapter(peft_name)

    if "custom_text_proj.pt" in os.listdir(peft_name):
        proj_state_dict = torch.load(os.path.join(peft_name, "custom_text_proj.pt"))
        model.custom_text_proj.base_layer.weight = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.weight'])
        model.custom_text_proj.base_layer.bias = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.bias'])

    model.pooling_strategy = None

    # Load the processor
    processor = ColQwen2Processor.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)['test']
    
    for data in tqdm(dataset):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 20))
        img_array = np.array(data["image"].convert("RGBA"))

        for i, layer_num in enumerate(layer_num_list):
            # Preprocess inputs
            batch_images = processor.process_images([data["image"]]).to(device)
            batch_images.update({"output_attentions": True})
            # Forward passes
            with torch.no_grad():
                # image_embeddings = model.forward(**batch_images)
                image_embeddings, self_attns = model.forward(**batch_images)
                last_self_attn = self_attns[layer_num]
            
            eos_idx = torch.nonzero(batch_images["input_ids"]==processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))[0][1]
            attn_weights = last_self_attn[:, :, eos_idx, :].sum(dim=1)
            # Get the number of image patches
            n_patches = processor.get_n_patches(image_size=data["image"].size, patch_size=model.patch_size, spatial_merge_size=2)
            # Get the tensor mask to filter out the embeddings that are not related to the image
            image_mask = processor.get_image_mask(batch_images)

            attn_maps = get_attn_maps(attn_weights, n_patches, image_mask)
            sub_fig, sub_ax = plot_one_similarity_map(
                image=data["image"],
                similarity_maps=attn_maps[0],
                figsize=(8, 10),
            )

            ax = axes[i//4, i%4]
            # 从子图中获取图像数据
            ax.imshow(img_array)
            for im in sub_ax.get_images():
                ax.imshow(im.get_array(), cmap=im.get_cmap(), norm=im.norm, alpha=0.5)            # 设置子图标题或其他样式
            ax.set_title("Layer {}".format(layer_num))
            
            query_id = data["questionId"] if "questionId" in data else data['image_filename'].split('/')[-1].split(".")[0]
        fig.savefig(
            os.path.join("/mnt/petrelfs/mayubo/colpali/colpali_engine/interpretability", dataset_name.split("/")[-1], "self_attn_{}.png".format(query_id))
        )