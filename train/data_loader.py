"""
Script: GSM8K Data Loader
Description: 
    Handles loading of the GSM8K dataset and formats samples into the 
    ChatML structure required by Qwen2.5 models.
"""
from datasets import load_dataset

def format_qwen_prompt(example, tokenizer):
    texts = []
    questions = example['question'] if isinstance(example['question'], list) else [example['question']]
    answers = example['answer'] if isinstance(example['answer'], list) else [example['answer']]

    for q, a in zip(questions, answers):
        text = (
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>"
        )
        texts.append(text + tokenizer.eos_token)
    
    return {"text": texts}

def get_gsm8k_dataset(split="train"):
    dataset = load_dataset("gsm8k", "main", split=split)
    return dataset