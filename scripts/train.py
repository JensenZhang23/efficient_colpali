"""Minimal LoRA training for ColPali or ColQwen2."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import ColbertPairwiseCELoss
from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.torch_utils import get_torch_device


TRAIN_REGISTRY = {
    "colpali": (ColPali, ColPaliProcessor, "vidore/colpaligemma-3b-pt-448-base"),
    "colqwen2": (ColQwen2, ColQwen2Processor, "vidore/colqwen2-base"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=list(TRAIN_REGISTRY), default="colqwen2")
    parser.add_argument("--model-name", help="Base model ID or local checkpoint")
    parser.add_argument("--dataset", default="vidore/colpali_train_set")
    parser.add_argument("--output-dir", default="outputs/retriever")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, help="Useful for smoke tests")
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--report-to", choices=["none", "wandb"], default="none")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    model_class, processor_class, default_name = TRAIN_REGISTRY[args.model_type]
    model_name = args.model_name or default_name
    device = get_torch_device(args.device)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    path = Path(args.dataset)
    dataset = load_from_disk(str(path)) if path.exists() else load_dataset(args.dataset)
    if hasattr(dataset, "keys") and "train" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("test")
        if eval_dataset is None:
            eval_dataset = dataset.get("validation")
    else:
        train_dataset = dataset
        eval_dataset = None

    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    if eval_dataset is None:
        if len(train_dataset) < 2:
            raise ValueError("At least two training samples are required to create a validation split")
        eval_size = max(1, min(args.eval_samples, len(train_dataset) // 10))
        split = train_dataset.train_test_split(test_size=eval_size, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
    elif args.eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.eval_samples, len(eval_dataset))))

    model = model_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        use_cache=False,
    )
    processor = processor_class.from_pretrained(model_name)
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=r".*(language_model|model).*(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj).*$",
            modules_to_save=["custom_text_proj"],
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.025,
        logging_steps=10,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=device != "cpu",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=VisualRetrieverCollator(processor=processor, max_length=50, pool_size=1),
        loss_func=ColbertPairwiseCELoss(),
        is_vision_model=True,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
