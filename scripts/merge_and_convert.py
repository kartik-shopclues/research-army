"""
scripts/merge_and_convert.py — Merge LoRA adapter into base model, then convert to GGUF for Ollama

Usage:
    python scripts/merge_and_convert.py --domain space \
        --base mistralai/Mistral-7B-Instruct-v0.3 \
        --adapter ./models/space_lora \
        --quantize q4_K_M
"""
import argparse
import os
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

OLLAMA_MODELFILES = {
    "space": """\
FROM ./space-merged-gguf/space-q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
SYSTEM \"\"\"You are an expert space science researcher with deep knowledge of astrophysics, orbital mechanics, satellite systems, space missions (NASA, ISRO, ESA, SpaceX), and space-based technology. Always ground your answers in evidence and cite sources.\"\"\"
""",
    "defence": """\
FROM ./defence-merged-gguf/defence-q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
SYSTEM \"\"\"You are an expert defence and security researcher with deep knowledge of military strategy, geopolitics, threat intelligence, weapons systems, and defence technology. Always ground your answers in evidence and cite sources.\"\"\"
""",
    "quantum": """\
FROM ./quantum-merged-gguf/quantum-q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
SYSTEM \"\"\"You are an expert quantum computing researcher with deep knowledge of quantum algorithms, QKD, quantum hardware (superconducting, photonic, ion-trap), and quantum cryptography. Always ground your answers in evidence and cite sources.\"\"\"
""",
}


def merge_lora(base_model: str, adapter_path: str, output_path: str, domain: str):
    """Merge LoRA weights into the base model."""
    console.print(f"[cyan]Merging LoRA adapter for {domain}...[/cyan]")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        console.print("[red]Install: pip install transformers peft torch[/red]")
        raise

    os.makedirs(output_path, exist_ok=True)

    console.print("  Loading base model (this may take several minutes)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # merge on CPU to save GPU VRAM
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    console.print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, adapter_path)

    console.print("  Merging weights...")
    model = model.merge_and_unload()

    console.print(f"  Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    console.print(f"[green]Merged model saved: {output_path}[/green]")
    return output_path


def convert_to_gguf(merged_path: str, domain: str, quantize: str = "q4_K_M") -> str:
    """Convert merged HF model to GGUF using llama.cpp."""
    console.print(f"[cyan]Converting to GGUF ({quantize})...[/cyan]")

    # Check llama.cpp is installed
    llamacpp_convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    llamacpp_quantize = Path.home() / "llama.cpp" / "llama-quantize"

    if not llamacpp_convert.exists():
        console.print("[yellow]llama.cpp not found. Installing...[/yellow]")
        subprocess.run([
            "bash", "-c",
            "cd ~ && git clone https://github.com/ggerganov/llama.cpp && "
            "cd llama.cpp && make -j$(nproc) LLAMA_CUDA=1"
        ], check=True)

    gguf_dir  = f"./{domain}-merged-gguf"
    gguf_f16  = f"{gguf_dir}/{domain}-f16.gguf"
    gguf_q4   = f"{gguf_dir}/{domain}-{quantize}.gguf"
    os.makedirs(gguf_dir, exist_ok=True)

    # Step 1: Convert to F16 GGUF
    console.print("  Step 1: Converting to F16 GGUF...")
    subprocess.run([
        "python3", str(llamacpp_convert),
        merged_path,
        "--outfile", gguf_f16,
        "--outtype", "f16",
    ], check=True)

    # Step 2: Quantize to Q4_K_M
    console.print(f"  Step 2: Quantizing to {quantize}...")
    subprocess.run([
        str(llamacpp_quantize),
        gguf_f16,
        gguf_q4,
        quantize.upper(),
    ], check=True)

    # Clean up F16 (large)
    os.remove(gguf_f16)
    console.print(f"[green]GGUF saved: {gguf_q4}[/green]")
    return gguf_q4


def create_ollama_model(domain: str, gguf_path: str):
    """Register the GGUF model with Ollama."""
    console.print(f"[cyan]Creating Ollama model: {domain}-specialist[/cyan]")

    modelfile_path = f"./models/{domain}.Modelfile"
    os.makedirs("./models", exist_ok=True)

    modelfile_content = OLLAMA_MODELFILES[domain].replace(
        f"./{domain}-merged-gguf/{domain}-q4_K_M.gguf",
        gguf_path,
    )

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    subprocess.run(
        ["ollama", "create", f"{domain}-specialist", "-f", modelfile_path],
        check=True,
    )

    console.print(f"[green]Model registered: {domain}-specialist[/green]")
    console.print(f"Test with: ollama run {domain}-specialist")


def update_config(domain: str):
    """Print config update instructions."""
    console.print(f"""
[yellow]Update config/settings.py to use your fine-tuned model:[/yellow]

  {domain}_model = "{domain}-specialist"

Then restart: python main.py
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",   required=True, choices=["space", "defence", "quantum"])
    parser.add_argument("--base",     required=True, help="HuggingFace base model ID")
    parser.add_argument("--adapter",  required=True, help="Path to LoRA adapter")
    parser.add_argument("--quantize", default="q4_K_M", help="Quantization level")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge if already done")
    args = parser.parse_args()

    merged_path = f"./models/{args.domain}-merged"

    if not args.skip_merge:
        merge_lora(args.base, args.adapter, merged_path, args.domain)

    gguf_path = convert_to_gguf(merged_path, args.domain, args.quantize)
    create_ollama_model(args.domain, gguf_path)
    update_config(args.domain)
