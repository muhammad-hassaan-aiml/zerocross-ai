import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import List

# Ensure Python discovers the compiled C++ engine and network modules
sys.path.extend(['.', 'build', 'python', '../build', os.path.join(os.getcwd(), 'build')])

import zerocross_engine
from network import ZeroCrossNet

app = FastAPI(title="ZeroCross AI Engine")

device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
net = None
models_dir = os.path.join(os.path.dirname(__file__), '../models')
frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')

# Hardcode the model path to prevent path traversal attacks
MODEL_PATH = os.path.join(models_dir, 'best_model.pth')

@app.on_event("startup")
def startup_event():
    global net
    
    # Fail loudly if the model is missing or 0 bytes
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        raise RuntimeError(f"FATAL: Valid model not found or is empty at {MODEL_PATH}")

    net = ZeroCrossNet()
    net.load_checkpoint(MODEL_PATH)
    net.to(device)
    net.eval()
    print(f"ZeroCross Engine Server Started. Loaded best_model.pth on {device}.")

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

# Input validation and compute capping
class MoveRequest(BaseModel):
    board: List[int]
    active_grid: int = Field(ge=-1, le=8)
    simulations: int = Field(default=200, le=800)

    @field_validator('board')
    @classmethod
    def validate_board(cls, v):
        if len(v) != 81:
            raise ValueError("Board must be exactly 81 elements")
        if any(cell not in [-1, 0, 1] for cell in v):
            raise ValueError("Board cells must only contain -1, 0, or 1")
        return v

@app.post("/move")
def get_best_move(req: MoveRequest):
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