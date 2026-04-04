import sys
import os
import argparse
import subprocess
from rich.console import Console

console = Console()

# Mapping domain to a quantized base model suitable for Unsloth
DOMAIN_BASE_MODELS = {
    "space": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "defence": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "quantum": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", # Qwen equivalent
}

def train_adapter(domain: str, dataset_path: str):
    """
    Trains a LoRA adapter on the selected domain dataset.
    Saves adapter to ./adapters/{domain}_lora
    """
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template, standardize_sharegpt
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch
    except ImportError:
        console.print("[red]Unsloth/Transformers not found. Ensure fine-tuning dependencies are installed.[/red]")
        sys.exit(1)

    max_seq_length = 2048
    base_model = DOMAIN_BASE_MODELS.get(domain)
    
    if not base_model:
        console.print(f"[red]Unknown domain {domain}[/red]")
        sys.exit(1)

    console.print(f"[green]Loading base model: {base_model}[/green]")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,
        max_seq_length = max_seq_length,
        dtype = None, # Auto detect
        load_in_4bit = True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
        mapping = {"role" : "from", "content" : "value", "user" : "user", "assistant" : "assistant"},
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
        
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Keep short for demonstration
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    console.print("[magenta]Starting training...[/magenta]")
    trainer_stats = trainer.train()
    
    adapter_path = f"./adapters/{domain}_lora"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    console.print(f"[green]Adapter saved to {adapter_path}[/green]")

def merge_and_export(domain: str):
    """
    Loads base model + adapter, merges them, exports to GGUF, 
    and adds it to Ollama.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        console.print("[red]Unsloth/Transformers not found.[/red]")
        sys.exit(1)

    adapter_path = f"./adapters/{domain}_lora"
    if not os.path.exists(adapter_path):
        console.print(f"[red]Adapter not found at {adapter_path}. Did you train it?[/red]")
        sys.exit(1)

    base_model = DOMAIN_BASE_MODELS.get(domain)
    console.print(f"[green]Loading base model for merging: {base_model} with adapter {adapter_path}[/green]")
    
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False, # Load in 16bit for merging
    )
    
    # Load adapter
    model.load_adapter(adapter_path)
    
    export_dir = f"./exports/{domain}_gguf"
    os.makedirs(export_dir, exist_ok=True)
    
    console.print("[magenta]Exporting to GGUF (q4_k_m)... This may take a few minutes.[/magenta]")
    # Unsloth merges and saves natively
    model.save_pretrained_gguf(export_dir, tokenizer, quantization_method = "q4_k_m")
    
    # The GGUF file is usually saved as something like unsloth.Q4_K_M.gguf
    # Let's find it
    gguf_files = [f for f in os.listdir(export_dir) if f.endswith(".gguf")]
    if not gguf_files:
        console.print("[red]GGUF export failed.[/red]")
        sys.exit(1)
        
    gguf_path = os.path.join(export_dir, gguf_files[0])
    
    modelfile_path = f"./exports/Modelfile_{domain}"
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {gguf_path}\n")
        f.write("PARAMETER temperature 0.7\n")
        f.write("PARAMETER keep_alive -1\n")
        f.write("PARAMETER num_ctx 4096\n")
    
    model_tag = f"{domain}-specialist:v1"
    console.print(f"[magenta]Creating Ollama model: {model_tag}[/magenta]")
    subprocess.run(["ollama", "create", model_tag, "-f", modelfile_path], check=True)
    
    console.print(f"[bold green]Success! Model {model_tag} is now available in Ollama.[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "merge"])
    parser.add_argument("--domain", required=True, choices=["space", "defence", "quantum"])
    parser.add_argument("--dataset", required=False, default="")
    args = parser.parse_args()

    if args.action == "train":
        if not args.dataset:
            console.print("[red]--dataset required for train[/red]")
            sys.exit(1)
        train_adapter(args.domain, args.dataset)
    elif args.action == "merge":
        merge_and_export(args.domain)
