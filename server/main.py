import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Ensure Python can discover the compiled C++ engine and neural network modules
sys.path.extend(['.', 'build', 'python', '../build', os.path.join(os.getcwd(), 'build')])

import zerocross_engine
from network import ZeroCrossNet

app = FastAPI(title="ZeroCross AI Engine")

device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
net = None
current_loaded_model = None
models_dir = os.path.join(os.path.dirname(__file__), '../models')

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
    # Initialize a baseline network on startup
    net = ZeroCrossNet()
    net.to(device)
    net.eval()
    print("ZeroCross Engine Server Started. Awaiting model selection from UI.")

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
    
    # 1. Reconstruct Game State
    try:
        state = zerocross_engine.GameState.from_array(req.board, req.active_grid)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid board state: {str(e)}")

    if state.is_terminal():
        return {"move": -1, "message": "Game is already over"}

    # 2. Run MCTS (No noise for deployment)
    tree = zerocross_engine.MCTSTree(state, False)
    
    while not tree.is_done(req.simulations):
        leaf = tree.request_leaf()
        if leaf is not None:
            # Derive legal mask from the 486-element tensor safely
            leaf_tensor = torch.tensor(leaf, dtype=torch.float32).view(6, 81)
            c0 = leaf_tensor[0].bool()
            c1 = leaf_tensor[1].bool()
            c2 = leaf_tensor[2].bool()
            legal_mask = c2 & ~c0 & ~c1
            
            # Predict using your dual-head network
            prob_list, val_scalar = net.predict(leaf, legal_mask)
            tree.submit_result(prob_list, val_scalar)
            
    # 3. Return best move (temperature 0.0 for deterministic competitive play)
    policy = tree.root_policy(0.0)
    best_move = int(np.argmax(policy))
    
    return {
        "move": best_move,
        "policy": policy
    }