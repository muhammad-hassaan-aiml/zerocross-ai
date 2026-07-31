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

@app.on_event("startup")
def load_model():
    global net
    # Path to the latest trained model weights
    model_path = os.path.join(os.path.dirname(__file__), '../models/latest_checkpoint.pth')
    
    print(f"Loading ZeroCross AI Champion onto {device}...")
    net = ZeroCrossNet()
    if os.path.exists(model_path):
        net.load_checkpoint(model_path)
        print("Model loaded successfully!")
    else:
        print("WARNING: Model not found! Using random initialization.")
    net.to(device)
    net.eval()

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