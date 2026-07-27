import os
import torch
from collections import deque
from network import ZeroCrossNet
from self_play import SelfPlayWorker
from train import train_network
from arena import Arena

def run_pipeline(iterations=100, max_buffer_size=500000):
    # Smart hardware selector (Local CPU vs Cloud GPU)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"--- Starting ZeroCross Training Pipeline on {device} ---")
    
    best_net = ZeroCrossNet()
    
    # Path resolution across Kaggle, Colab, and Local
    if os.path.exists("/kaggle/working"):
        drive_dir = "/kaggle/working/models"
    elif os.path.exists("/content/drive"):
        drive_dir = "/content/drive/MyDrive/zerocross_models"
    else:
        drive_dir = "models"
        
    os.makedirs(drive_dir, exist_ok=True)
    model_path = os.path.join(drive_dir, "best_model.pth")
    
    if os.path.exists(model_path):
        print(f"Loading existing champion from {model_path}...")
        best_net.load_state_dict(torch.load(model_path, map_location=device))
    
    best_net.to(device)
    
    # Sliding Window Replay Buffer for Kaggle (500k max samples)
    replay_buffer = deque(maxlen=max_buffer_size)
    
    for i in range(iterations):
        print(f"\n{'='*50}")
        print(f" ALPHAZERO ITERATION {i+1} / {iterations}")
        print(f"{'='*50}")
        
        # 1. Generate Data (Self-Play)
        print("\n[1/4] Generating Batched Self-Play Data...")
        worker = SelfPlayWorker(best_net, num_concurrent_games=200, mcts_simulations=400)
        new_samples = worker.generate_data(total_games_to_play=100) 
        
        # Add new games to the replay buffer
        replay_buffer.extend(new_samples)
        print(f"Replay Buffer Capacity: {len(replay_buffer)} / {max_buffer_size} samples")
        
        # 2. Train Candidate on Replay Buffer
        print(f"\n[2/4] Training Candidate Network on Replay Buffer...")
        candidate_net = ZeroCrossNet().to(device)
        candidate_net.load_state_dict(best_net.state_dict())
        
        candidate_net = train_network(
            candidate_net, 
            list(replay_buffer), 
            batch_size=256, 
            epochs=5, 
            lr=0.001, 
            device=device
        )
        
        # 3. Arena Evaluation
        print("\n[3/4] Arena Evaluation (Candidate vs Champion)...")
        arena = Arena(candidate_net, best_net, mcts_simulations=100, device=device)
        promoted = arena.evaluate(num_games=40)
        
        # 4. Model Gating & Persistence
        print("\n[4/4] Model Gating...")
        if promoted:
            print(f">>> UPGRADE! Candidate promoted! Saving to {model_path} <<<")
            best_net.load_state_dict(candidate_net.state_dict())
            torch.save(best_net.state_dict(), model_path)
        else:
            print(">>> REJECTED! Candidate failed to hit 55%. Keeping old champion. <<<")

if __name__ == "__main__":
    run_pipeline(iterations=100)