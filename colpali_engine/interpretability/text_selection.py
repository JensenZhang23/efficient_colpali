import os
import re
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device


if __name__=="__main__":
    torch.manual_seed(42)

    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad_1207/checkpoint-2310"
    dataset_name = "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled"

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
        pooling_strategy=None
    ).eval()
    model.load_adapter(peft_name)

    if "custom_text_proj.pt" in os.listdir(peft_name):
        proj_state_dict = torch.load(os.path.join(peft_name, "custom_text_proj.pt"))
        model.custom_text_proj.base_layer.weight = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.weight'])
        model.custom_text_proj.base_layer.bias = nn.Parameter(proj_state_dict['base_model.model.custom_text_proj.base_layer.bias'])

    # Load the processor
    processor = ColQwen2Processor.from_pretrained(model_name)
    dataset = load_dataset(dataset_name)['test']
    
    for data in tqdm(dataset):
        batch_images = processor.process_images([data["image"]]).to(device)
        batch_queries = processor.process_queries([data["query"]]).to(device)
        # Forward passes
        with torch.no_grad():
            # image_embeddings = model.forward(**batch_images)
            image_embeddings = model.forward(**batch_images).squeeze()
            query_embeddings = model.forward(**batch_queries).squeeze()
        
        query_embeddings = query_embeddings[(batch_queries["input_ids"]!=151643).nonzero(as_tuple=False)[:,1]]

        m_v_t = image_embeddings @ query_embeddings.transpose(0, 1)
        m_v_t = m_v_t.softmax(1).mean(0)
        t_token_idx = torch.where(m_v_t > m_v_t.mean())[0]

        print("--------------------------")
        print("Before Selection: {}".format(
            re.sub(r"<\|endoftext\|>", "", processor.decode(batch_queries["input_ids"][0]))
        ))
        print("After Selection: {}".format(
            processor.decode(batch_queries["input_ids"][0, t_token_idx])
        ))