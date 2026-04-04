import asyncio
import os
import sys
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Form, UploadFile, File
from pydantic import BaseModel
from collections import deque
import api.server # Import server globals
from finetune.dataset import build_dataset_for_domain

router = APIRouter(prefix="/finetune", tags=["finetune"])

# In-memory buffer to hold logs of the current training/merging job
logs_buffer = deque(maxlen=500)
active_job: Optional[str] = None
connected_websockets = set()

async def broadcast_log(line: str):
    logs_buffer.append(line)
    for ws in list(connected_websockets):
        try:
            await ws.send_text(line)
        except Exception:
            connected_websockets.remove(ws)

async def run_subprocess(cmd: list, job_name: str):
    global active_job
    active_job = job_name
    await broadcast_log(f"--- Starting {job_name} ---\n")
    await broadcast_log(f"Command: {' '.join(cmd)}\n")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )
    
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        await broadcast_log(line.decode('utf-8'))
        
    await process.wait()
    await broadcast_log(f"--- {job_name} Finished with exit code {process.returncode} ---\n")
    active_job = None

# FinetuneRequest is removed as we use Form data to support file uploads
@router.get("/stats")
async def get_stats():
    """Returns number of training examples available per domain."""
    stats = {}
    if api.server.memory:
        cache = api.server.memory._cache
        for domain in ["space", "defence", "quantum"]:
            count = 0
            for key, res in cache.items():
                if res.get("mode") == "mode_a" and res.get("domain") == domain:
                    if res.get("synthesis"): count += 1
                elif "domain_outputs" in res and domain in res["domain_outputs"]:
                    if res["domain_outputs"][domain].get("response"): count += 1
            stats[domain] = count
    
    # Check if there are adapters present
    adapters = {}
    for domain in ["space", "defence", "quantum"]:
        adapters[domain] = os.path.exists(f"./adapters/{domain}_lora")

    return {
        "stats": stats, 
        "adapters": adapters,
        "active_job": active_job
    }

@router.post("/start")
async def start_training(
    background_tasks: BackgroundTasks,
    domain: str = Form(...),
    dataset: Optional[UploadFile] = File(None)
):
    global active_job
    if active_job:
        raise HTTPException(400, "A job is already running")
        
    if domain not in ["space", "defence", "quantum"]:
        raise HTTPException(400, "Invalid domain")

    # 1. Extract or save data
    if dataset:
        dataset_path = f"data/finetune_{domain}_custom.jsonl"
        os.makedirs("data", exist_ok=True)
        content = await dataset.read()
        with open(dataset_path, "wb") as f:
            f.write(content)
        count = "Custom File"
        await broadcast_log(f"Using uploaded custom dataset: {dataset.filename}\n")
    else:
        if not api.server.memory:
            raise HTTPException(500, "MemoryStore not initialized")
            
        dataset_path = f"data/finetune_{domain}.jsonl"
        count = build_dataset_for_domain(api.server.memory._cache, domain, dataset_path)
        
        if count == 0:
            raise HTTPException(400, "No data available to train this domain.")

        await broadcast_log(f"Extracted {count} examples to {dataset_path}\n")

    # 2. Run trainer in background
    cmd = [sys.executable, "-m", "finetune.trainer", "train", "--domain", domain, "--dataset", dataset_path]
    background_tasks.add_task(run_subprocess, cmd, f"Fine-Tuning {domain}")
    
    return {"status": "started", "domain": domain, "examples": count}

@router.post("/merge")
async def start_merging(
    background_tasks: BackgroundTasks,
    domain: str = Form(...)
):
    global active_job
    if active_job:
        raise HTTPException(400, "A job is already running")
        
    if domain not in ["space", "defence", "quantum"]:
        raise HTTPException(400, "Invalid domain")
        
    if not os.path.exists(f"./adapters/{domain}_lora"):
        raise HTTPException(400, f"No LoRA adapter found for {domain}")

    cmd = [sys.executable, "-m", "finetune.trainer", "merge", "--domain", domain]
    background_tasks.add_task(run_subprocess, cmd, f"Merging {domain}")
    
    return {"status": "started", "domain": domain}

@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    
    # Send history
    for line in logs_buffer:
        await websocket.send_text(line)
        
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
