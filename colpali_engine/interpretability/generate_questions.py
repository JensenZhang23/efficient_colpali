import re
import json
import argparse

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model_and_processor(model_name_or_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


def generate(model, processor, image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="docvqa")
    args = parser.parse_args()

    # model_name_or_path = "/mnt/hwfile/mllm/huangqidong/lvlm/Qwen2-VL-72B-Instruct"
    model_name_or_path = "/mnt/hwfile/mllm/mayubo/Qwen2.5-VL-7B-Instruct"
    dataset_path = {
        "docvqa": "/mnt/petrelfs/mayubo/colpali/data_dir/docvqa_test_subsampled",
        "arxivqa": "/mnt/petrelfs/mayubo/colpali/data_dir/arxivqa_test_subsampled",
        "infovqa": "/mnt/petrelfs/mayubo/colpali/data_dir/infovqa_test_subsampled",
        "tabfquad": "/mnt/petrelfs/mayubo/colpali/data_dir/tabfquad_test_subsampled",
        "tatdqa": "/mnt/petrelfs/mayubo/colpali/data_dir/tatdqa_test",
        "shift": "/mnt/petrelfs/mayubo/colpali/data_dir/shiftproject_test",
    }[args.dataset_name]
    
    prompt = f"Given the screenshot of a document/poster, you are asked to generate five question that can be answered by looking at the image.  The questions should be relevant to the content of the image. The questions should be unique and be of varying question types. The output should be formatted as:\n1. [Question 1]\n2. [Question 2]\n3. [Question 3]\n4. [Question 4]\n5. [Question 5]"

    model, processor = load_model_and_processor(model_name_or_path)
    dataset = load_dataset(dataset_path, split="test")
    
    questions = dict()
    for i, data in enumerate(tqdm(dataset)):
        data["image"].save("./tmp.png")
        try:
            generated_questions = generate(model, processor, image_path="./tmp.png", prompt=prompt)
            generated_questions = re.findall(r'\d+\.\s+(.*)', generated_questions)
        except:
            print("Fail to pass {}".format(generated_questions))
            continue
        questions[i] = [data["query"]] + generated_questions
        with open(f"./expanded_questions/{args.dataset_name}.json", "w") as f:
            json.dump(questions, f)