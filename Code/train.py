# This script trains a model using LoRA (Low-Rank Adaptation) with the TRL library.
#


import torch
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import argparse


def create_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # loading in 4 bit
        bnb_4bit_quant_type="nf4", # quantization type
        bnb_4bit_use_double_quant=True, # nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation='eager'
    )
    model.config.use_cache = False

    lora_alpha = 32
    lora_dropout = 0.1
    lora_r = 16

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM"
    )

    lora_model = get_peft_model(model, peft_config)
    return lora_model, peft_config

def train_model(model, peft_config, output_dir, max_steps):
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 1
    optim = "paged_adamw_32bit"  # specialization of the AdamW optimizer that enables efficient learning in LoRA setting.
    save_steps = 1000
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none"
    )

    max_seq_length = 512

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with LoRA")
    parser.add_argument("--model_name", type=str, default="cjvt/GaMS-9B-Instruct", help="Name of the model to train")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the training dataset in JSONL format")
    parser.add_argument("--max_steps", type=int, required=True, help="Maximum number of training steps")
    args = parser.parse_args()

    train_dataset = load_dataset("json", data_files="train_dataset.jsonl", split="train")

    model_name = args.model_name
    model, peft_config = create_model(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_model(model, peft_config, args.output_dir)
