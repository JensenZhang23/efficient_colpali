from typing import List, cast

import os
import json
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
sys.setrecursionlimit(10000)

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from colpali_engine.models import ColQwen2
from colpali_engine.models.qwen2.colqwen2.processing_colqwen2 import ColQwen2Processor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device


def draw_res(query, gt_image, pred_image, output_path):
    """
    query: str
    gt_image: PIL.Image
    pred_image: PIL.Image
    output_path: str

    Draw an image for the convenience of visualization. The gt_image and pred_image should be put horizontally within ONE figure.
    And the query should be at the top of the figure. Then save this figure to the output_path. Do not change the color of gt_image and pred_image.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    gt_image.save('tmp_1.png')
    pred_image.save('tmp_2.png')

    # 重新读取图像
    gt_image = Image.open('tmp_1.png').convert("RGB")
    pred_image = Image.open('tmp_2.png').convert("RGB")

    # 将图像转换为数组
    gt_image_array = np.array(gt_image)
    pred_image_array = np.array(pred_image)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)  # 可以调整图像大小
    ax[0].imshow(gt_image_array)
    ax[0].axis('off')
    ax[0].set_title("Ground Truth")
    ax[1].imshow(pred_image_array)
    ax[1].axis('off')
    ax[1].set_title("Prediction")
    fig.suptitle(query)
    plt.savefig(output_path)
    plt.close()


def main():
    """
    Example script to run inference with ColPali.
    """

    device = get_torch_device("auto")
    print(f"Device used: {device}")

    # Model name
    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad-proj-nolora_post_proj_flatten_1207/checkpoint-2310"
    # peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad-proj-nolora/checkpoint-2310"
    eval_name = "docvqa_test_subsampled"
    # eval_name = "infovqa_test_subsampled"
    # eval_name = "arxivqa_test_subsampled"
    # Load model
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    # model.pooling_strategy = "post_proj_flatten"
    # model.pool_size = 10
    model.load_adapter(peft_name)

    # Load processor
    processor = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))

    if not isinstance(processor, BaseVisualRetrieverProcessor):
        raise ValueError("Processor should be a BaseVisualRetrieverProcessor")

    dataset = cast(Dataset, load_dataset("/mnt/petrelfs/mayubo/colpali/data_dir/{}".format(eval_name), split="test"))
    images = dataset["image"]
    queries = dataset['query']

    # Run inference - docs
    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # Run inference - queries
    dataloader = DataLoader(
        dataset=ListDataset[str](queries),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # Run scoring
    scores = processor.score(qs, ds).cpu().numpy()
    idx_top_1 = scores.argmax(axis=1)
    print("Indices of the top-1 retrieved documents for each query:", idx_top_1)
    acc = sum([idx==i for i, idx in enumerate(idx_top_1)])/len(idx_top_1)
    print("Acc: {}".format(acc))

    with open("./results_{}_{}.json".format(eval_name, peft_name.split("/")[-2]), "w") as f:
        json.dump(idx_top_1.tolist(), f)
    
    # if not os.path.exists(f"./error_case/{eval_name}") :
    #     os.makedirs(f"./error_case/{eval_name}")
    # for gt_id, pred_id in enumerate(idx_top_1):
    #     if gt_id!=pred_id:
    #         draw_res(queries[gt_id], images[gt_id], images[pred_id], f"./error_case/{eval_name}/output_{gt_id}.png")


if __name__ == "__main__":
    main()