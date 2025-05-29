# This script evaluates a trained model on a dataset and saves the results to a CSV file.
# It creates a pipeline for text generation using a pre-trained model and tokenizer,

import torch
import argparse
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
import csv

def load_model(base_model_name, trained_model_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # loading in 4 bit
        bnb_4bit_quant_type="nf4", # quantization type
        bnb_4bit_use_double_quant=True, # nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
        aattn_implementation='eager'
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False

    if trained_model_path == "None":
        model.eval()
        return model

    lora_config = LoraConfig.from_pretrained(trained_model_path)

    peft_model = PeftModel.from_pretrained(
        model=model,
        model_id=trained_model_path,
        peft_config=lora_config
    )
    peft_model.eval()
    return peft_model

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--base_model', type=str, default="cjvt/GaMS-9B-Instruct", help='Model name')
    argparser.add_argument('--trained_model_path', type=str, required=True, help='Path to the trained model. To evaluate only base model, set to "None"')
    argparser.add_argument('--dataset_path', type=str, default='test_dataset.jsonl', help='Path to the dataset jsonl')
    argparser.add_argument('--output_path', type=str, default='eval_results.csv', help='Path to save the evaluation results')
    args = argparser.parse_args()

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    model = load_model(args.base_model, args.trained_model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write("prompt,completion,output\n")

    for row in dataset:
        prompt = row["prompt"]
        print(f"Evaluating prompt: \n{prompt}")

        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500
            )

        all_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
        generated_text = all_text[len(prompt):].strip()
        print("\n----------------------------------\n" +generated_text+"\n")
        with open(args.output_path, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([prompt, row['completion'], generated_text])

