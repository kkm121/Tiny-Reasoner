"""
Script: Local QLoRA Training Pipeline
Description: 
    Fine-tunes Qwen2.5-1.5B on the GSM8K dataset using Parameter-Efficient Fine-Tuning (PEFT).
    Designed for local execution: assumes dependencies are installed and uses local paths.
"""
import os
import sys
import yaml
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if "__file__" in globals():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    project_root = os.path.abspath(".")

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from data.gsm8k_loader import get_gsm8k_dataset, format_qwen_prompt
except ImportError:
    print("[-] Generating local data loader...")
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    with open(os.path.join(project_root, "data", "gsm8k_loader.py"), "w") as f:
        f.write("""from datasets import load_dataset

def format_qwen_prompt(example, tokenizer):
    texts = []
    questions = example['question'] if isinstance(example['question'], list) else [example['question']]
    answers = example['answer'] if isinstance(example['answer'], list) else [example['answer']]

    for q, a in zip(questions, answers):
        text = (
            f"<|im_start|>user\\n{q}<|im_end|>\\n"
            f"<|im_start|>assistant\\n{a}<|im_end|>"
        )
        texts.append(text + tokenizer.eos_token)
    
    return {"text": texts}

def get_gsm8k_dataset(split="train"):
    dataset = load_dataset("gsm8k", "main", split=split)
    return dataset
""")
    from data.gsm8k_loader import get_gsm8k_dataset, format_qwen_prompt

config_path = os.path.join(project_root, "configs", "qwen_qlora.yaml")
if not os.path.exists(config_path):
    print("[-] Generating default config...")
    os.makedirs(os.path.join(project_root, "configs"), exist_ok=True)
    with open(config_path, "w") as f:
        f.write("""model:
  base_id: "Qwen/Qwen2.5-1.5B-Instruct"
  max_seq_length: 1024
  load_in_4bit: true

lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  bias: "none"
  target_modules: 
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  warmup_steps: 10
  max_steps: 60
  learning_rate: 2.0e-4
  logging_steps: 1
  optim: "paged_adamw_8bit"
  output_dir: "tiny_reasoner_artifacts"

paths:
  adapter_save_path: "tiny_reasoner_artifacts"
""")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train():
    gc.collect()
    torch.cuda.empty_cache()
    
    cfg = load_config(config_path)
    model_id = cfg['model']['base_id']
    output_dir = os.path.join(project_root, cfg['training']['output_dir'])

    print(f">>> Loading Base Model: {model_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> Applying LoRA Adapters...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['lora_alpha'],
        lora_dropout=cfg['lora']['lora_dropout'],
        bias=cfg['lora']['bias'],
        task_type="CAUSAL_LM",
        target_modules=cfg['lora']['target_modules']
    )
    model = get_peft_model(model, peft_config)

    print(">>> Preparing Dataset...")
    raw_dataset = get_gsm8k_dataset(split="train")
    
    def process_data(example):
        formatted_batch = format_qwen_prompt(example, tokenizer)
        encodings = tokenizer(
            formatted_batch['text'],
            truncation=True,
            max_length=cfg['model']['max_seq_length']
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    tokenized_dataset = raw_dataset.map(
        process_data,
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        warmup_steps=cfg['training']['warmup_steps'],
        max_steps=cfg['training']['max_steps'],
        learning_rate=cfg['training']['learning_rate'],
        fp16=True,
        logging_steps=cfg['training']['logging_steps'],
        optim=cfg['training']['optim'],
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    print(">>> Starting Training...")
    trainer.train()

    print(f">>> Saving artifacts to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    meta = {"base_model": model_id}
    with open(os.path.join(output_dir, "adapter_metadata.yaml"), "w") as f:
        yaml.safe_dump(meta, f)
        
    print(">>> Training Complete. Artifacts saved locally.")

if __name__ == "__main__":
    train()