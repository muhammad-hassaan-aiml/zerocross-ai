import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

# Ensure Python discovers the compiled C++ engine and network modules
sys.path.extend(['.', 'build', 'python', '../build', os.path.join(os.getcwd(), 'build')])

import zerocross_engine
from network import ZeroCrossNet

app = FastAPI(title="ZeroCross AI Engine")

device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
net = None
current_loaded_model = None
models_dir = os.path.join(os.path.dirname(__file__), '../models')
frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')

def load_specific_model(filename):
    global net, current_loaded_model
    model_path = os.path.join(models_dir, filename)
    
    print(f"Loading {filename} onto {device}...")
    new_net = ZeroCrossNet()
    if os.path.exists(model_path):
        new_net.load_checkpoint(model_path)
        new_net.to(device)
        new_net.eval()
        net = new_net
        current_loaded_model = filename
        print(f"Model {filename} loaded successfully!")
        return True
    else:
        print(f"WARNING: Model {filename} not found!")
        return False

@app.on_event("startup")
def startup_event():
    global net
    net = ZeroCrossNet()
    net.to(device)
    net.eval()

    # Auto-load a sensible default checkpoint so the engine isn't serving
    # moves from random init weights before the frontend gets a chance to
    # call /load. Prefer a file literally named "best_model.pth"; otherwise
    # fall back to the first checkpoint alphabetically.
    if os.path.exists(models_dir):
        available = sorted(f for f in os.listdir(models_dir) if f.endswith('.pth'))
        if available:
            preferred = 'best_model.pth' if 'best_model.pth' in available else available[0]
            load_specific_model(preferred)

    print("ZeroCross Engine Server Started.")

# Serve frontend static assets if directory exists
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path, 
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "ZeroCross AI Engine API Running"}

@app.get("/models")
def get_available_models():
    """Scans the models directory and returns all available .pth files."""
    if not os.path.exists(models_dir):
        return {"models": []}
    
    pth_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    pth_files.sort() 
    return {"models": pth_files, "current": current_loaded_model}

class LoadModelRequest(BaseModel):
    filename: str

@app.post("/load")
def load_model_endpoint(req: LoadModelRequest):
    """Hot-swaps the active neural network in memory."""
    success = load_specific_model(req.filename)
    if success:
        return {"message": f"Successfully loaded {req.filename}"}
    else:
        raise HTTPException(status_code=404, detail="Model file not found")

class MoveRequest(BaseModel):
    board: List[int]
    active_grid: int
    simulations: int = 100

@app.post("/move")
def get_best_move(req: MoveRequest):
    if len(req.board) != 81:
        raise HTTPException(status_code=400, detail="Board must be exactly 81 elements")
    
    try:
        state = zerocross_engine.GameState.from_array(req.board, req.active_grid)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid board state: {str(e)}")

    if state.is_terminal():
        return {"move": -1, "message": "Game is already over"}

    tree = zerocross_engine.MCTSTree(state, False)
    
    while not tree.is_done(req.simulations):
        leaf = tree.request_leaf()
        if leaf is not None:
            leaf_tensor = torch.tensor(leaf, dtype=torch.float32).view(6, 81)
            c0 = leaf_tensor[0].bool()
            c1 = leaf_tensor[1].bool()
            c2 = leaf_tensor[2].bool()
            legal_mask = c2 & ~c0 & ~c1
            
            prob_list, val_scalar = net.predict(leaf, legal_mask)
            tree.submit_result(prob_list, val_scalar)
            
    policy = tree.root_policy(0.0)
    best_move = int(np.argmax(policy))
    
    # Get current root value prediction for UI win probability
    _, win_prob = net.predict(state.encode(), state.legal_mask())
    
    return {
        "move": best_move,
        "policy": np.asarray(policy).tolist(),
        "value": float(win_prob)
    }