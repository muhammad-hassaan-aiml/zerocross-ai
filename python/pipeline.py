import os
import torch
from network import ZeroCrossNet
from self_play import SelfPlayWorker
from train import train_network
from arena import Arena

def run_pipeline(iterations=100):
    # CORRECTED: Smart hardware selector to prevent local 940MX crashes while allowing Colab T4 GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"--- Starting ZeroCross Training Pipeline on {device} ---")
    
    best_net = ZeroCrossNet()
    
    # Cloud-agnostic checkpoint routing
    drive_dir = "/content/drive/MyDrive/zerocross_models"
    if not os.path.exists("/content/drive"):
        drive_dir = "models" # Fallback for local execution
        
    os.makedirs(drive_dir, exist_ok=True)
    model_path = os.path.join(drive_dir, "best_model.pth")
    
    if os.path.exists(model_path):
        print(f"Loading existing champion from {model_path}...")
        best_net.load_state_dict(torch.load(model_path, map_location=device))
    
    best_net.to(device)
    best_net.eval()
    
    for i in range(iterations):
        print(f"\n{'='*50}")
        print(f" ALPHAZERO ITERATION {i+1} / {iterations}")
        print(f"{'='*50}")
        
        # 1. Generate Data (Self-Play)
        print("\n[1/3] Generating Batched Self-Play Data...")
        worker = SelfPlayWorker(best_net, num_concurrent_games=200, mcts_simulations=400)
        training_data = worker.generate_data(total_games_to_play=100) 
        
        # 2. Train Candidate
        print(f"\n[2/3] Training Candidate Network on {len(training_data)} samples...")
        candidate_net = ZeroCrossNet().to(device)
        candidate_net.load_state_dict(best_net.state_dict()) # Copy champion's brain
        
        candidate_net = train_network(
            candidate_net, 
            training_data, 
            batch_size=256, 
            epochs=5, 
            lr=0.001, 
            device=device
        )
        
        # 3. The Arena
        print("\n[3/3] Arena Evaluation (Candidate vs Champion)...")
        arena = Arena(candidate_net, best_net, mcts_simulations=100, device=device)
        promoted = arena.evaluate(num_games=40)
        
        # 4. Model Gating & Persistence
        if promoted:
            print(f"\n>>> UPGRADE! Candidate promoted! Saving to {model_path} <<<")
            best_net.load_state_dict(candidate_net.state_dict())
            torch.save(best_net.state_dict(), model_path)
        else:
            print("\n>>> REJECTED! Candidate failed to hit 55%. Keeping old champion. <<<")

if __name__ == "__main__":
    run_pipeline(iterations=100)