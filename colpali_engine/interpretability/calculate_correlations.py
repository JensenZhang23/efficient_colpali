import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import concatenate_datasets

import argparse
from datasets import load_dataset
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator
from colpali_engine.utils.torch_utils import get_torch_device, find_min_max_indices
from colpali_engine.interpretability.identify_valid_patch import compute_passage_embeddings, compute_query_embeddings


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
    self_attns_rank = list()
    ranges_list = list()
    for dataloader in [dataloader_with_query, dataloader_without_query]:
        for batch in tqdm(dataloader):
            # feed only kwargs with 'doc_' prefix
            batch_inputs = {k[4:]: v.to(device) for k, v in batch.items() if k.startswith("doc")}
            batch_inputs.update({"output_attentions": True})
            doc, self_attns = model(**batch_inputs)
            ps.extend(list(torch.unbind(doc.to("cpu"))))
            rank_list = [rank for rank in self_attns[-2][:, :, -2, :].max(dim=1).values.sort().indices] # Last layer, last token (query), from small to large
            range_list = find_min_max_indices(batch["doc_input_ids"], 151655).tolist()
            ranges_list.extend(range_list)

            for rank, (start, end) in zip(rank_list, range_list):
                rank = rank[torch.isin(rank, torch.arange(start, end+1).to(rank.device))]-start
                self_attns_rank.append(rank)

    return ps, self_attns_rank, ranges_list


def top_k_overlap(list1, list2, k):
    # 取前k项并转换为集合
    top1 = set(list1[:k])
    top2 = set(list2[:k])
    
    # 计算交集比例
    overlap = len(top1 & top2)
    return overlap / k


if __name__=="__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_out", type=str, default="/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled")
    args = parser.parse_args()

    model_name = "/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct"
    peft_name = "/mnt/petrelfs/mayubo/colpali/models/colqwen2-ba256-ckpt-5e-nopad-proj-nolora_post_proj_2dpool_0130_rebuilt/checkpoint-2310"
    pooling_strategy = None
    pool_size = 1

    max_length = 768
    batch_size = 8
    dataset_name_in = "/mnt/petrelfs/mayubo/colpali/data_dir/colpali_train_set"
    dataset_name_out = args.dataset_name_out
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
    test_dataset_in = load_dataset(dataset_name_in)['test']
    test_dataset_out = load_dataset(dataset_name_out)['test']

    test_dataset_in = test_dataset_in.select(range(20))
    test_dataset_out = test_dataset_in.select(range(20))

    ps, self_attns_rank, ranges_list = compute_passage_embeddings(test_dataset_out, collator)
    max_ps_len = max([len(p) for p in ps])
    offset_list = [
        (start+pad_num, end+pad_num) for (start, end), pad_num in zip(ranges_list, [max_ps_len-len(p) for p in ps])
    ]

    # Left-pad
    ps = torch.stack([torch.cat([torch.zeros(max_ps_len-len(p), p.shape[-1], dtype=p.dtype), p], dim=0) for p in ps])
    qs_in = compute_query_embeddings(test_dataset_in, collator)
    scores_in = torch.einsum("bnd,csd->bcns", ps, pad_sequence(qs_in, batch_first=True))

    patch_scores = scores_in.max(dim=-1).values.max(dim=1).values
    # patch_scores = scores_in.mean(dim=-1).max(dim=1).values
    # patch_scores = scores_in.topk(dim=-1, k=3).values.mean(dim=-1).max(dim=1).values
    scores_rank = list()
    for score_rank, (start, end) in zip(patch_scores.sort().indices, ranges_list):  # From small to large
        score_rank = score_rank[torch.isin(score_rank, torch.arange(start, end+1).to(score_rank.device))]
        scores_rank.append(score_rank)

    overlap = 0.0
    for score_rank, self_attn_rank in zip(scores_rank, self_attns_rank):
        score_rank = score_rank.tolist()[::-1]  # From large to small
        self_attn_rank = self_attn_rank.tolist()[::-1]  # From large to small
        overlap += top_k_overlap(score_rank, self_attn_rank, 100)/len(scores_rank)
    print(overlap)