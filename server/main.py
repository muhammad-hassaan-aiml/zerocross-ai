import os
import sys
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import List

sys.path.extend(['.', 'build', 'python', '../build', os.path.join(os.getcwd(), 'build')])

import zerocross_engine

app = FastAPI(title="ZeroCross AI Engine")

ort_session = None
models_dir = os.path.join(os.path.dirname(__file__), '../models')
frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')

MODEL_PATH = os.path.join(models_dir, 'best_model.onnx')

@app.on_event("startup")
def startup_event():
    global ort_session
    
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        raise RuntimeError(f"FATAL: Valid ONNX model not found or is empty at {MODEL_PATH}")

    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("ZeroCross Engine Server Started. Loaded best_model.onnx on CPU via ONNX Runtime.")

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
            leaf_np = np.array(leaf, dtype=np.float32).reshape(1, 6, 9, 9)
            
            ort_inputs = {ort_session.get_inputs()[0].name: leaf_np}
            logits, values = ort_session.run(None, ort_inputs)
            
            raw_logits = logits.flatten().tolist()
            raw_value = float(values[0][0])
            
            tree.submit_result(raw_logits, raw_value)
            
    policy = tree.root_policy(0.0)
    best_move = int(np.argmax(policy))
    
    root_state_np = np.array(state.encode(), dtype=np.float32).reshape(1, 6, 9, 9)
    ort_inputs = {ort_session.get_inputs()[0].name: root_state_np}
    _, root_val = ort_session.run(None, ort_inputs)
    
    return {
        "move": best_move,
        "policy": np.asarray(policy).tolist(),
        "value": float(root_val[0][0])
    }