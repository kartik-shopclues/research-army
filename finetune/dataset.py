import json
import os
from typing import Dict, List

def build_dataset_for_domain(cache_dict: Dict[str, Dict], domain: str, out_path: str) -> int:
    """
    Extracts successful domain responses from the memory cache and formats 
    them as a ShareGPT JSONL dataset for fine-tuning.
    
    Args:
        cache_dict: The MemoryStore._cache dictionary
        domain: "space", "defence", or "quantum"
        out_path: Filepath to write the dataset (e.g. data/finetune_space.jsonl)
        
    Returns:
        number of examples extracted
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    examples_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for key, result in cache_dict.items():
            query = result.get("query", "")
            if not query:
                continue
                
            # Mode A single-domain hit
            if result.get("mode") == "mode_a" and result.get("domain") == domain:
                answer = result.get("synthesis")
                if answer:
                    record = {
                        "conversations": [
                            {"from": "user", "value": query},
                            {"from": "assistant", "value": answer}
                        ]
                    }
                    f.write(json.dumps(record) + "\n")
                    examples_count += 1
            
            # Mode B or Mode B+ (Debate/Broadcast)
            elif "domain_outputs" in result and domain in result["domain_outputs"]:
                answer = result["domain_outputs"][domain].get("response")
                # sometimes sub_tasks are defined in plan
                plan = result.get("plan", {})
                sub_task = plan.get("sub_tasks", {}).get(domain, query)
                
                if answer:
                    record = {
                        "conversations": [
                            {"from": "user", "value": sub_task or query},
                            {"from": "assistant", "value": answer}
                        ]
                    }
                    f.write(json.dumps(record) + "\n")
                    examples_count += 1
                    
            # Fallback for old cache formats
            elif domain in result and isinstance(result[domain], dict) and "response" in result[domain]:
                answer = result[domain]["response"]
                if answer:
                    record = {
                        "conversations": [
                            {"from": "user", "value": query},
                            {"from": "assistant", "value": answer}
                        ]
                    }
                    f.write(json.dumps(record) + "\n")
                    examples_count += 1

    return examples_count
