#!/usr/bin/env python3
"""
Combined CPT Training: Documentation + Code datasets
Mixes archit11/deepwiki-16k (docs) and archit11/hyperswitch-code (code) for comprehensive training.
"""

import re
import torch
from datasets import load_dataset, concatenate_datasets, interleave_datasets
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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def format_doc_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format documentation sample with <doc> and <code> tags."""
    content = example['content']
    source_code_snippets = example.get('source_code_snippets', [])

    inline_code_blocks = extract_code_blocks(content)
    clean_doc = remove_code_blocks(content)

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
        lang_map = {'rs': 'rust', 'toml': 'toml', 'py': 'python', 'js': 'javascript', 'ts': 'typescript'}
        lang = lang_map.get(ext, ext)
        parts.append(f"<code language=\"{lang}\" source=\"{file_path}\">\n{code}\n</code>")

    return {
        'text': "\n\n".join(parts),
        'source': 'documentation',
        'id': example.get('id', 'doc_unknown')
    }


def format_code_sample(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format code-only sample with <code> tags."""
    text = example['text']
    file_path = example.get('file_path', 'unknown')

    # Wrap in <code> tags
    formatted = f"<code language=\"rust\" source=\"{file_path}\">\n{text}\n</code>"

    return {
        'text': formatted,
        'source': 'code',
        'id': example.get('file_path', 'code_unknown')
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
    DOC_DATASET = "archit11/deepwiki-16k"
    CODE_DATASET = "archit11/hyperswitch-code"
    MODEL_NAME = "Kwaipilot/KAT-Dev"
    OUTPUT_DIR = "./combined-cpt-output"
    MAX_SEQ_LENGTH = 16384

    # Dataset mixing ratio (docs:code)
    DOC_WEIGHT = 0.5  # 50% documentation
    CODE_WEIGHT = 0.5  # 50% code

    # LoRA Configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    logger.info("=" * 70)
    logger.info("Combined CPT Training: Documentation + Code")
    logger.info("=" * 70)

    # Load tokenizer
    logger.info("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Add special tokens
    special_tokens = {'additional_special_tokens': ['<doc>', '</doc>', '<code>', '</code>']}
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"  Added {num_added} special tokens")

    # Load datasets
    logger.info("\n[2/7] Loading datasets...")
    logger.info(f"  Loading {DOC_DATASET}...")
    doc_dataset = load_dataset(DOC_DATASET)
    logger.info(f"    ✓ {len(doc_dataset['train'])} documentation samples")

    logger.info(f"  Loading {CODE_DATASET}...")
    code_dataset = load_dataset(CODE_DATASET)
    logger.info(f"    ✓ {len(code_dataset['train'])} code samples")

    # Format datasets
    logger.info("\n[3/7] Formatting datasets with CPT tags...")
    logger.info("  Formatting documentation samples...")
    formatted_docs = doc_dataset['train'].map(
        format_doc_sample,
        remove_columns=doc_dataset['train'].column_names,
        desc="Formatting docs"
    )

    logger.info("  Formatting code samples...")
    formatted_code = code_dataset['train'].map(
        format_code_sample,
        remove_columns=code_dataset['train'].column_names,
        desc="Formatting code"
    )

    # Combine datasets with interleaving (maintains diversity)
    logger.info("\n[4/7] Combining datasets...")
    logger.info(f"  Mixing ratio - Docs: {DOC_WEIGHT*100:.0f}%, Code: {CODE_WEIGHT*100:.0f}%")

    combined_dataset = interleave_datasets(
        [formatted_docs, formatted_code],
        probabilities=[DOC_WEIGHT, CODE_WEIGHT],
        seed=42
    )

    logger.info(f"  ✓ Combined dataset: {len(combined_dataset)} samples")

    # Tokenize
    logger.info("\n[5/7] Tokenizing combined dataset...")
    tokenized_dataset = combined_dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=['text', 'source', 'id'],
        desc="Tokenizing"
    )

    # Load model
    logger.info("\n[6/7] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Configure LoRA
    logger.info("\n[7/7] Configuring LoRA...")
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
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
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("\n" + "=" * 70)
    logger.info("Starting combined CPT training...")
    logger.info(f"  Documentation samples: ~{int(len(combined_dataset) * DOC_WEIGHT):,}")
    logger.info(f"  Code samples: ~{int(len(combined_dataset) * CODE_WEIGHT):,}")
    logger.info("=" * 70)

    trainer.train()

    # Save
    logger.info("\nSaving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    logger.info("\n✅ Training complete!")
    logger.info(f"   Model saved to: {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
