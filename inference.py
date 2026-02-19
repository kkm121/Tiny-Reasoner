"""
Script: Local Inference Engine
Description: 
    Loads locally stored LoRA adapters and the Qwen base model for offline, 
    interactive reasoning tasks. Optimized for consumer hardware.
"""
import os
import torch
import yaml
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftModel

def load_model(adapter_path):
    meta_path = os.path.join(adapter_path, "adapter_metadata.yaml")
    base_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            data = yaml.safe_load(f)
            base_id = data.get("base_model", base_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    return model, tokenizer

def generate(model, tokenizer, prompt_text):
    formatted_prompt = (
        f"<|im_start|>user\n"
        f"{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in decoded:
        return decoded.split("assistant")[-1].strip()
    return decoded

def run_interface():
    adapter_path = "tiny_reasoner_artifacts"
    
    if not os.path.exists(adapter_path):
        print(f"Error: Directory '{adapter_path}' not found.")
        return

    print("Loading model...")
    model, tokenizer = load_model(adapter_path)
    
    print("\n" + "="*60)
    print("Tiny-Reasoner Local Interface")
    print("="*60)
    
    while True:
        try:
            query = input("\nInput: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            response = generate(model, tokenizer, query)
            print("-" * 60)
            print(f"Output:\n{response}")
            print("-" * 60)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_interface()