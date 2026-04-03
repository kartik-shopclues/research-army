"""
scripts/finetune.py — QLoRA fine-tuning for domain specialist LLMs
Uses HuggingFace PEFT + bitsandbytes for 24GB GPU

Usage:
    python scripts/finetune.py --domain space --data ./data/space/ --base mistralai/Mistral-7B-Instruct-v0.3
    python scripts/finetune.py --domain defence --data ./data/defence/ --base meta-llama/Llama-3.2-8B-Instruct
    python scripts/finetune.py --domain quantum --data ./data/quantum/ --base Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

from rich.console import Console

console = Console()

# ── Import guard: these are only needed during training ───────────────────
def _import_training_libs():
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            TrainingArguments, BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
        from datasets import Dataset
        return torch, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, \
               BitsAndBytesConfig, LoraConfig, get_peft_model, TaskType, \
               prepare_model_for_kbit_training, SFTTrainer, \
               DataCollatorForCompletionOnlyLM, Dataset
    except ImportError as e:
        console.print(f"[red]Missing training library: {e}[/red]")
        console.print("Install with: pip install transformers peft trl datasets bitsandbytes")
        raise


# ── Domain system prompts for training ────────────────────────────────────
DOMAIN_PROMPTS = {
    "space": "You are an expert space science researcher with deep knowledge of astrophysics, orbital mechanics, satellite systems, and space missions.",
    "defence": "You are an expert defence and security researcher with deep knowledge of military strategy, threat intelligence, and defence technology.",
    "quantum": "You are an expert quantum computing researcher with deep knowledge of quantum algorithms, QKD, and quantum hardware.",
}


def load_raw_texts(data_dir: str) -> List[str]:
    """Load all .txt, .md, .pdf files from data directory."""
    texts = []
    data_path = Path(data_dir)

    for ext in ["*.txt", "*.md"]:
        for f in data_path.rglob(ext):
            try:
                texts.append(f.read_text(encoding="utf-8", errors="replace"))
                console.print(f"  [green]Loaded:[/green] {f.name}")
            except Exception as e:
                console.print(f"  [red]Skip {f.name}: {e}[/red]")

    for f in data_path.rglob("*.pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(f))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            if text.strip():
                texts.append(text)
                console.print(f"  [green]Loaded PDF:[/green] {f.name}")
        except Exception as e:
            console.print(f"  [red]Skip {f.name}: {e}[/red]")

    return texts


def texts_to_qa_dataset(texts: List[str], domain: str, tokenizer) -> List[Dict]:
    """
    Convert raw texts into instruction-tuning examples.
    We chunk each text and wrap it in a Q&A template.
    """
    system_prompt = DOMAIN_PROMPTS[domain]
    examples = []
    chunk_size = 800  # words per chunk

    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - 100):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) < 100:
                continue

            # Format as chat-style instruction
            formatted = (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                f"Summarise and explain the following {domain} research content:\n\n"
                f"{chunk[:2000]} [/INST]\n"
                f"Based on this content, here is a detailed expert analysis:\n\n"
                f"{chunk[:1500]}</s>"
            )
            examples.append({"text": formatted})

    console.print(f"[cyan]Created {len(examples)} training examples[/cyan]")
    return examples


def finetune(
    domain: str,
    data_dir: str,
    base_model: str,
    output_dir: str = None,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 2,
):
    (torch, AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
     BitsAndBytesConfig, LoraConfig, get_peft_model, TaskType,
     prepare_model_for_kbit_training, SFTTrainer,
     DataCollatorForCompletionOnlyLM, Dataset) = _import_training_libs()

    output_dir = output_dir or f"./models/{domain}_lora"
    os.makedirs(output_dir, exist_ok=True)

    console.rule(f"[bold]Fine-tuning {domain.upper()} LLM[/bold]")
    console.print(f"Base model : {base_model}")
    console.print(f"Data dir   : {data_dir}")
    console.print(f"Output dir : {output_dir}")
    console.print(f"Epochs     : {epochs}")

    # ── 4-bit quantisation config ─────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ── Load tokenizer ────────────────────────────────────────────────────
    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model in 4-bit ───────────────────────────────────────────────
    console.print("[cyan]Loading base model in 4-bit...[/cyan]")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load and format data ──────────────────────────────────────────────
    console.print("[cyan]Loading training data...[/cyan]")
    raw_texts = load_raw_texts(data_dir)
    if not raw_texts:
        console.print("[red]No training data found! Add .txt/.pdf files to the data directory.[/red]")
        return

    examples = texts_to_qa_dataset(raw_texts, domain, tokenizer)
    dataset = Dataset.from_list(examples)
    dataset = dataset.shuffle(seed=42)

    # Train/val split
    split = dataset.train_test_split(test_size=0.05)
    train_ds = split["train"]
    eval_ds  = split["test"]
    console.print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Training args ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        optim="paged_adamw_8bit",
        dataloader_pin_memory=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
    )

    console.print("[green]Starting training...[/green]")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────
    console.print(f"[green]Saving LoRA adapter to {output_dir}[/green]")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    console.print(f"""
[bold green]Training complete![/bold green]

Next steps:
  1. Merge adapter into base model:
     python scripts/merge_lora.py --domain {domain} --adapter {output_dir} --base {base_model}

  2. Convert to GGUF for Ollama:
     python scripts/convert_to_gguf.py --domain {domain}

  3. Create Ollama modelfile and load:
     ollama create {domain}-specialist -f ./models/{domain}.Modelfile
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tune a domain specialist LLM")
    parser.add_argument("--domain",  required=True, choices=["space", "defence", "quantum"])
    parser.add_argument("--data",    required=True, help="Path to training data directory")
    parser.add_argument("--base",    required=True, help="HuggingFace base model ID")
    parser.add_argument("--output",  default=None, help="Output directory for adapter")
    parser.add_argument("--epochs",  type=int, default=3)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--batch",   type=int, default=2)
    args = parser.parse_args()

    finetune(
        domain=args.domain,
        data_dir=args.data,
        base_model=args.base,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
    )
