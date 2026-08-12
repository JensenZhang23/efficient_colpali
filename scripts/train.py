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
    args = parser.parse_args()

    model_class, processor_class, default_name = TRAIN_REGISTRY[args.model_type]
    model_name = args.model_name or default_name
    device = get_torch_device(args.device)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    path = Path(args.dataset)
    dataset = load_from_disk(str(path)) if path.exists() else load_dataset(args.dataset)
    train_dataset = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))

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
        save_steps=500,
        save_total_limit=2,
        bf16=device != "cpu",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = ContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
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
