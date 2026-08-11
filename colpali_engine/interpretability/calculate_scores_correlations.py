import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import json
import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

import argparse
from tqdm import tqdm
from collections import defaultdict

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.utils.torch_utils import get_torch_device, find_min_max_indices


def top_k_overlap(list1, list2, k):
    # 取前k项并转换为集合
    top1 = set(list1[:k])
    top2 = set(list2[:k])
    
    # 计算交集比例
    overlap = len(top1 & top2)
    return overlap / k


def last_k_overlap(list1, list2, k):
    # 取后k项并转换为集合
    top1 = set(list1[-k:])
    top2 = set(list2[-k:])
    
    # 计算交集比例
    overlap = len(top1 & top2)
    return overlap / k


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

    ps, ps_prev, ps_after = list(), list(), list()
    range_list = list()
    for dataloader in [dataloader_with_query, dataloader_without_query]:
        for batch in tqdm(dataloader):
            # feed only kwargs with 'doc_' prefix
            batch_inputs = {k[4:]: v.to(device) for k, v in batch.items() if k.startswith("doc")}
            doc = model(**batch_inputs)
            ps.extend(list(torch.unbind(doc.to("cpu"))))
            range_list.extend(list(torch.unbind(find_min_max_indices(batch["doc_input_ids"], 151655).to("cpu"))))

    for i, (p_embed, range) in enumerate(zip(ps, range_list)):
        p_prev = p_embed[:range[0]]
        p_prev = p_prev[torch.sum(p_prev, dim=-1)!=0]
        ps_prev.append(p_prev)

        p_after = p_embed[range[1]+1:]
        ps_after.append(p_after)

        p_embed = p_embed[range[0]:range[1]+1]
        ps[i] = p_embed
    return ps, ps_prev, ps_after


if __name__=="__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="docvqa")
    args = parser.parse_args()

    dataset_path = {
        "docvqa": "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled",
        "arxivqa": "/mnt/petrelfs/mayubo/colpali/data_dir/arxivqa_test_subsampled",
        "infovqa": "/mnt/petrelfs/mayubo/colpali/data_dir/infovqa_test_subsampled",
        "tabfquad": "/mnt/petrelfs/mayubo/colpali/data_dir/tabfquad_test_subsampled",
    }[args.dataset_name]

    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/hwfile/mllm/mayubo/colqwen2-v1.0/checkpoint-2310"
    # peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad-proj-nolora_post_proj_2dpool_0130_rebuilt/checkpoint-2310"
    pooling_strategy = None
    pool_size = 1

    max_length = 768
    batch_size = 8
    device = get_torch_device("auto")

    with open("./expanded_questions/{}.json".format(args.dataset_name)) as f:
        questions_dict = json.load(f)

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
    test_dataset = load_dataset(dataset_path)['test']
    # test_dataset = test_dataset.select(range(20))

    ps, ps_prev, ps_after = compute_passage_embeddings(test_dataset, collator)
    corr_1_dict_max, corr_1_dict_mean = defaultdict(list), defaultdict(list)
    corr_2_dict_max, corr_2_dict_mean = defaultdict(list), defaultdict(list)
    corr_1_dict_max_last, corr_1_dict_mean_last = defaultdict(list), defaultdict(list)
    corr_2_dict_max_last, corr_2_dict_mean_last = defaultdict(list), defaultdict(list)

    scores_dict = defaultdict(list)
    for i, (p_embed, p_prev, p_after) in enumerate(zip(ps, ps_prev, ps_after)):
        if str(i) not in questions_dict:
            continue
        questions = questions_dict[str(i)]
        batch_queries = processor.process_queries(questions)
        q_embed = model(**{k:v.to(device) for k, v in batch_queries.items()})

        scores = torch.einsum("nd,csd->cns", p_embed.to(device), q_embed)
        scores_prev = torch.einsum("nd,csd->cns", p_prev.to(device), q_embed)
        scores_after = torch.einsum("nd,csd->cns", p_after.to(device), q_embed)

        scores_dict["prev_max"].append(scores_prev.max(dim=-1).values.mean().item())
        scores_dict["prev_mean"].append(scores_prev.mean().item())
        scores_dict["p_max"].append(scores.max(dim=-1).values.mean().item())
        scores_dict["p_mean"].append(scores.mean().item())
        scores_dict["after_max"].append(scores_after.max(dim=-1).values.mean().item())
        scores_dict["after_mean"].append(scores_after.mean().item())

        ranks = scores.max(dim=-1).values.argsort(dim=-1, descending=True)
        for ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]:
            overlap_matrix = torch.zeros((ranks.size(0), ranks.size(0)))
            for i in range(ranks.size(0)):
                for j in range(i+1, ranks.size(0)):
                    overlap_matrix[i, j] = top_k_overlap(ranks[i].tolist(), ranks[j].tolist(), k=int(p_embed.size(0)*ratio))
            corr_1 = overlap_matrix[0][1:].mean()
            corr_2 = overlap_matrix.sum()/((ranks.size(0)-1)*ranks.size(0)//2)
            corr_1_dict_max[ratio].append(corr_1.item())
            corr_2_dict_max[ratio].append(corr_2.item())

            overlap_matrix = torch.zeros((ranks.size(0), ranks.size(0)))
            for i in range(ranks.size(0)):
                for j in range(i+1, ranks.size(0)):
                    overlap_matrix[i, j] = last_k_overlap(ranks[i].tolist(), ranks[j].tolist(), k=int(p_embed.size(0)*ratio))
            corr_1_last = overlap_matrix[0][1:].mean()
            corr_2_last = overlap_matrix.sum()/((ranks.size(0)-1)*ranks.size(0)//2)
            corr_1_dict_max_last[ratio].append(corr_1.item())
            corr_2_dict_max_last[ratio].append(corr_2.item())


        ranks = scores.mean(dim=-1).argsort(dim=-1, descending=True)
        for ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]:
            overlap_matrix = torch.zeros((ranks.size(0), ranks.size(0)))
            for i in range(ranks.size(0)):
                for j in range(i+1, ranks.size(0)):
                    overlap_matrix[i, j] = top_k_overlap(ranks[i].tolist(), ranks[j].tolist(), k=int(p_embed.size(0)*ratio))
            corr_1 = overlap_matrix[0][1:].mean()
            corr_2 = overlap_matrix.sum()/((ranks.size(0)-1)*ranks.size(0)//2)
            corr_1_dict_mean[ratio].append(corr_1.item())
            corr_2_dict_mean[ratio].append(corr_2.item())

    
    with jsonlines.open("./correlations.jsonl".format(args.dataset_name), "a") as writer:
        writer.write(
            {
                "dataset_name": args.dataset_name,
                "prev_mean": sum(scores_dict["prev_mean"])/len(scores_dict["prev_mean"]),
                "prev_max": sum(scores_dict["prev_max"])/len(scores_dict["prev_max"]),
                "p_mean": sum(scores_dict["p_mean"])/len(scores_dict["p_mean"]),
                "p_max": sum(scores_dict["p_max"])/len(scores_dict["p_max"]),
                "after_mean": sum(scores_dict["after_mean"])/len(scores_dict["after_mean"]),
                "after_max": sum(scores_dict["after_max"])/len(scores_dict["after_max"]),
            }
        )
    
    for ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]:
        corr_1_list_max = corr_1_dict_max[ratio]
        corr_2_list_max = corr_2_dict_max[ratio]
        corr_1_list_mean = corr_1_dict_mean[ratio]
        corr_2_list_mean = corr_2_dict_mean[ratio]
        corr_1_list_max_last = corr_1_dict_max_last[ratio]
        corr_2_list_max_last = corr_2_dict_max_last[ratio]
        with jsonlines.open("./correlations.jsonl".format(args.dataset_name), "a") as writer:
            writer.write(
                {
                    "dataset_name": args.dataset_name,
                    "ratio": ratio,
                    "corr_1_max": sum(corr_1_list_max)/len(corr_1_list_max),
                    "corr_2_max": sum(corr_2_list_max)/len(corr_2_list_max),
                    "corr_1_max_last": sum(corr_1_list_max_last)/len(corr_1_list_max_last),
                    "corr_2_max_last": sum(corr_2_list_max_last)/len(corr_2_list_max_last),
                    "corr_1_mean": sum(corr_1_list_mean)/len(corr_1_list_mean),
                    "corr_2_mean": sum(corr_2_list_mean)/len(corr_2_list_mean),
                }
            )