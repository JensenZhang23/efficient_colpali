import os
import sys
sys.path.append("/mnt/petrelfs/mayubo/colpali")
import json
import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, concatenate_datasets

import argparse
from tqdm import tqdm
from typing import Dict, Tuple

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.trainer.eval_utils import CustomRetrievalEvaluator
from colpali_engine.utils.torch_utils import get_torch_device


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
def compute_passage_embeddings(test_dataset, collator, output_attentions=False):
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
    for dataloader in [dataloader_with_query, dataloader_without_query]:
        for batch in tqdm(dataloader):
            # feed only kwargs with 'doc_' prefix
            batch_inputs = {k[4:]: v.to(device) for k, v in batch.items() if k.startswith("doc")}
            batch_inputs.update({"output_attentions": output_attentions})
            if output_attentions:
                doc, self_attns = model(**batch_inputs)
                self_attns_rank.extend(list(torch.unbind(self_attns[-1][:, :, -2, :].max(dim=1).values.argsort(dim=-1).to("cpu"))))
            else:
                doc = model(**batch_inputs)
            ps.extend(list(torch.unbind(doc.to("cpu"))))
    return ps, self_attns_rank if output_attentions else None


def compute_scores(processor, test_dataset, qs, ps):
    print("Embeddings computed, evaluating")
    scores = processor.score(qs, ps, device=device)

    # scores is 2d array of shape (n_queries, n_docs)
    # turn it into a dict
    relevant_docs = {}
    docidx_2_docid = {}
    qsidx_2_query = []
    for idx, sample in enumerate(test_dataset):
        doc_id = sample["image_filename"] if "image_filename" in sample else str(hash(sample["doc"]))
        # query_id = sample["query_id"] if "query_id" in sample else str(hash(sample["query"]))
        if sample["query"] is not None:
            relevant_docs[str(idx)] = {doc_id: 1}
            qsidx_2_query.append(str(idx))
        docidx_2_docid[str(idx)] = doc_id
    
    results = {}
    assert scores.shape[0] == len(qsidx_2_query)
    for idx, scores_per_query in enumerate(scores):
        results[qsidx_2_query[idx]] = {
            docidx_2_docid[str(docidx)]: float(score) for docidx, score in enumerate(scores_per_query)
        }

    # evaluate
    retrieval_evaluator = CustomRetrievalEvaluator()
    metrics = retrieval_evaluator.compute_mteb_metrics(relevant_docs, results)
    print("MTEB metrics:", metrics)
    return metrics


if __name__=="__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_out", type=str, default="docvqa")
    parser.add_argument("--model_name", type=str, default="/mnt/hwfile/mllm/mayubo/Qwen2-VL-2B-Instruct")
    parser.add_argument("--peft_name", type=str, default="/mnt/hwfile/mllm/mayubo/colqwen2-v1.0/checkpoint-2310")
    parser.add_argument("--prune_criteron", type=str, default="score", choices=["score", "attention", "random", "cluster", "cluster-score", "cluster-random"])
    parser.add_argument("--pooling_strategy", type=str, default=None)
    parser.add_argument("--pool_size", type=int, default=1)
    parser.add_argument("--is_reversed", action="store_true")
    args = parser.parse_args()

    pooling_strategy = args.pooling_strategy
    pool_size = args.pool_size

    max_length = 768
    batch_size = 8
    device = get_torch_device("auto")

    dataset_name_in = "/mnt/petrelfs/mayubo/colpali/data_dir/colpali_train_set"
    dataset_name_out = {
        "docvqa": "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled",
        "arxivqa": "/mnt/petrelfs/mayubo/colpali/data_dir/arxivqa_test_subsampled",
        "infovqa": "/mnt/petrelfs/mayubo/colpali/data_dir/infovqa_test_subsampled",
        "tabfquad": "/mnt/petrelfs/mayubo/colpali/data_dir/tabfquad_test_subsampled",
        "tatdqa": "/mnt/petrelfs/mayubo/colpali/data_dir/tatdqa_test",
        "shift": "/mnt/petrelfs/mayubo/colpali/data_dir/shiftproject_test",
    }[args.dataset_name_out]

    # Load the model and processor
    processor = ColQwen2Processor.from_pretrained(
        args.model_name,
        max_length=max_length,
    )
    model = ColQwen2.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2" if args.prune_criteron != "attention" else "eager",
        pooling_strategy="post_proj_cluster" if args.prune_criteron in ["cluster", "cluster-score", "cluster-random"] else pooling_strategy,
        pool_size=9 if args.prune_criteron in ["cluster", "cluster-score", "cluster-random"] else pool_size,
    ).eval()
    model.load_adapter(args.peft_name)
    if "custom_text_proj.pt" in os.listdir(args.peft_name):
        proj_state_dict = torch.load(
            os.path.join(args.peft_name, "custom_text_proj.pt")
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

    # test_dataset_in = test_dataset_in.select(range(50))
    # test_dataset_out = test_dataset_out.select(range(50))

    if args.prune_criteron != "attention":
        ps, _ = compute_passage_embeddings(test_dataset_out, collator)
    else:
        ps, self_attns_rank = compute_passage_embeddings(test_dataset_out, collator, output_attentions=True)

    qs_out = compute_query_embeddings(test_dataset_out, collator)

    max_ps_len = max([len(p) for p in ps])
    offset_list = [max_ps_len-len(p) for p in ps]
    # Left-pad
    ps = torch.stack([torch.cat([torch.zeros(max_ps_len-len(p), p.shape[-1], dtype=p.dtype), p], dim=0) for p in ps])

    metrics = compute_scores(processor, test_dataset_out, qs_out, ps)
    with jsonlines.open("./prune_results{}.jsonl".format("_reversed" if args.is_reversed else ""), 'a') as writer:
        writer.write({
            "prune_strategy": "{}_0.0".format(args.prune_criteron),
            "peft_name": args.peft_name,
            "pooling_strategy": pooling_strategy,
            "pool_size": pool_size,
            "dataset_name": args.dataset_name_out,
            "metrics": {
                "recall_at_1": metrics["recall_at_1"],
                "recall_at_5": metrics["recall_at_5"],
                "ndcg_at_5": metrics["ndcg_at_5"],
            }
        })

    if args.prune_criteron == "attention":
        for ratio in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            ps_copied = [p.clone() for p in ps]
            for p_embed, self_attn_rank, offset in zip(ps_copied, self_attns_rank, offset_list):
                self_attn_rank = self_attn_rank + offset
                if args.is_reversed:
                    self_attn_rank = self_attn_rank.flip(dims=(0,))
                remove_index = self_attn_rank[:int(len(self_attn_rank)*ratio)]
                p_embed[remove_index] = 0.0
            metrics = compute_scores(processor, test_dataset_out, qs_out, ps_copied)

            with jsonlines.open("./prune_results{}.jsonl".format("_reversed" if args.is_reversed else ""), 'a') as writer:
                writer.write({
                    "prune_strategy": "{}_{}".format(args.prune_criteron, ratio),
                    "peft_name": args.peft_name,
                    "pooling_strategy": pooling_strategy,
                    "pool_size": pool_size,
                    "dataset_name": args.dataset_name_out,
                    "metrics": {
                        "recall_at_1": metrics["recall_at_1"],
                        "recall_at_5": metrics["recall_at_5"],
                        "ndcg_at_5": metrics["ndcg_at_5"],
                    }
                })
    elif args.prune_criteron != "cluster":
        test_dataset_in = load_dataset(dataset_name_in)['test']
        qs_in = compute_query_embeddings(test_dataset_in, collator)
        scores_in = torch.einsum("bnd,csd->bcns", ps, pad_sequence(qs_in, batch_first=True))
        patch_scores = scores_in.max(dim=-1).values.max(dim=1).values
        # patch_scores = scores_in.max(dim=-1).values.std(dim=1)
        # patch_scores = scores_in.mean(dim=-1).max(dim=1).values
        # patch_scores = scores_in.topk(dim=-1, k=3).values.mean(dim=-1).max(dim=1).values

        with open("./expanded_questions/{}.json".format(args.dataset_name_out)) as f:
            questions_dict = json.load(f)
        ps_copied = [p.clone() for p in ps]
        for i, p_embed in enumerate(tqdm(ps_copied)):
            if str(i) not in questions_dict:    # Skip the patches that are not in the expanded questions.
                continue
            questions = questions_dict[str(i)][1:]  # The first question is the original question. Remove it.
            batch_queries = processor.process_queries(questions)
            q_embed = model(**{k:v.to(device) for k, v in batch_queries.items()})
            patch_scores[i] = torch.einsum("nd,csd->cns", p_embed.to(device), q_embed).max(dim=-1).values.max(dim=0).values

        for ratio in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            ps_copied = [p.clone() for p in ps]
            for p_embed, patch_score in zip(ps_copied, patch_scores):
                zero_num = patch_score.eq(0).sum().item()
                index_rank = patch_score.sort().indices[zero_num:]
                if args.is_reversed:
                    index_rank = index_rank.flip(dims=(0,))
                if args.prune_criteron in ["cluster-random", "random"]:
                    index_rank = index_rank[torch.randperm(len(index_rank))]
                # Pick patches either by score or by random
                remove_index = index_rank[:int(len(index_rank)*ratio)]
                p_embed[remove_index] = 0.0
            metrics = compute_scores(processor, test_dataset_out, qs_out, ps_copied)

            with jsonlines.open("./prune_results{}.jsonl".format("_reversed" if args.is_reversed else ""), 'a') as writer:
                writer.write({
                    "prune_strategy": "{}_{}".format(args.prune_criteron, ratio),
                    "peft_name": args.peft_name,
                    "pooling_strategy": pooling_strategy,
                    "pool_size": pool_size,
                    "dataset_name": args.dataset_name_out,
                    "metrics": {
                        "recall_at_1": metrics["recall_at_1"],
                        "recall_at_5": metrics["recall_at_5"],
                        "ndcg_at_5": metrics["ndcg_at_5"],
                    }
                })
    else:
        pass