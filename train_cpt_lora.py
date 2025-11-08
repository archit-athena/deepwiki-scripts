#!/usr/bin/env python3
"""
LoRA CPT Training Script with on-the-fly <doc> <code> formatting
Supports multi-GPU training on 4x H200
"""

import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
from typing import Dict, Any


def extract_code_blocks(content: str):
    """Extract code blocks from markdown content."""
    code_blocks = []
    pattern = r'```(\w*)\n(.*?)```'

    for match in re.finditer(pattern, content, re.DOTALL):
        language = match.group(1) or 'text'
        code = match.group(2).strip()
        if code:
            code_blocks.append({'language': language, 'code': code})

    return code_blocks


def remove_code_blocks(content: str) -> str:
    """Remove code blocks from content."""
    content = re.sub(r'```[\s\S]*?```', '', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def format_as_cpt(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw dataset example to CPT format with <doc> and <code> tags.
    """
    content = example['content']
    source_code_snippets = example.get('source_code_snippets', [])

    # Extract inline code blocks from documentation
    inline_code_blocks = extract_code_blocks(content)
    clean_doc = remove_code_blocks(content)

    # Build CPT formatted text
    parts = []

    # Add documentation
    if clean_doc:
        parts.append(f"<doc>\n{clean_doc}\n</doc>")

    # Add inline code blocks
    for code_block in inline_code_blocks:
        lang = code_block['language']
        code = code_block['code']
        parts.append(f"<code language=\"{lang}\">\n{code}\n</code>")

    # Add source code snippets
    for snippet in source_code_snippets:
        file_path = snippet['file_path']
        code = snippet['code']
        ext = Path(file_path).suffix[1:] if Path(file_path).suffix else 'text'

        lang_map = {
            'rs': 'rust', 'toml': 'toml', 'py': 'python',
            'js': 'javascript', 'ts': 'typescript'
        }
        lang = lang_map.get(ext, ext)

        parts.append(f"<code language=\"{lang}\" source=\"{file_path}\">\n{code}\n</code>")

    # Join all parts
    formatted_text = "\n\n".join(parts)

    return {
        'text': formatted_text,
        'id': example['id'],
        'source_file': example['source_file']
    }


def tokenize_function(examples, tokenizer, max_length=16384):
    """Tokenize examples for training."""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )


def main():
    # Configuration
    MODEL_NAME = "Kwaipilot/KAT-Dev"
    DATASET_NAME = "archit11/deepwiki-16k"
    OUTPUT_DIR = "./deepwiki-lora-output"
    MAX_SEQ_LENGTH = 16384  # 16k context

    # LoRA Configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print("=" * 60)
    print("DeepWiki CPT LoRA Training")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Add special tokens if needed
    special_tokens = {'additional_special_tokens': ['<doc>', '</doc>', '<code>', '</code>']}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"  Added {num_added} special tokens")

    # Load dataset
    print("\n[2/6] Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    print(f"  Loaded {len(dataset['train'])} examples")

    # Format dataset with CPT tags
    print("\n[3/6] Formatting with <doc> <code> tags...")
    formatted_dataset = dataset.map(
        format_as_cpt,
        remove_columns=dataset['train'].column_names,
        desc="Formatting examples"
    )

    # Tokenize dataset
    print("\n[4/6] Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=['text', 'id', 'source_file'],
        desc="Tokenizing"
    )

    # Load model
    print("\n[5/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Resize token embeddings if we added tokens
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Configure LoRA
    print("\n[6/6] Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",  # Change to "wandb" if you want logging
        max_grad_norm=1.0,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    print("\n✅ Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
