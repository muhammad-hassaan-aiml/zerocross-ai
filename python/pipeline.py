import os
import csv
import time
import torch
from collections import deque
from network import ZeroCrossNet
from self_play import SelfPlayWorker
from train import train_network
from arena import Arena

def run_pipeline(iterations=100, max_buffer_size=500000):
    # Smart hardware selector Local CPU vs Cloud GPU
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Starting ZeroCross Training Pipeline on {device}")
    
    # Path resolution across Kaggle, Colab, and Local
    if os.path.exists("/kaggle/working"):
        drive_dir = "/kaggle/working/models"
    elif os.path.exists("/content/drive"):
        drive_dir = "/content/drive/MyDrive/zerocross_models"
    else:
        drive_dir = "models"
        
    os.makedirs(drive_dir, exist_ok=True)
    model_path = os.path.join(drive_dir, "best_model.pth")
    csv_path = os.path.join(drive_dir, "training_log.csv")
    buffer_path = os.path.join(drive_dir, "replay_buffer.pt")
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "LR", "PI_Loss", "V_Loss", "Entropy", "WinRate_vs_Random", "Promoted"])
    
    best_net = ZeroCrossNet()
    optimizer_state = None
    start_iteration = 0
    
    # Resume Campaign State
    if os.path.exists(model_path):
        print(f"Loading existing champion from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Safely handle both standard weights and our new dual save format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            best_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            
            # Read CSV to auto resume the exact iteration count across cloud resets
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    start_iteration = sum(1 for row in f) - 1
        else:
            best_net.load_state_dict(checkpoint)
            
    best_net.to(device)
    
    # Sliding Window Replay Buffer
    replay_buffer = deque(maxlen=max_buffer_size)
    
    # Load historical buffer if resuming campaign
    if os.path.exists(buffer_path):
        print(f"Loading historical replay buffer from {buffer_path}")
        checkpoint = torch.load(buffer_path)
        
        # Handle the new dictionary format or fallback to the old list format
        if isinstance(checkpoint, dict) and 'data' in checkpoint:
            replay_buffer.extend(checkpoint['data'])
        else:
            replay_buffer.extend(checkpoint)
            
        print(f"Restored {len(replay_buffer)} historical samples to memory")
    
    for i in range(start_iteration, start_iteration + iterations):
        current_iter = i + 1
        print(f"\nALPHAZERO ITERATION {current_iter} of {start_iteration + iterations}")
        
        # DYNAMIC LEARNING RATE SCHEDULE
        if current_iter <= 20:
            current_lr = 0.001
        elif current_iter <= 50:
            current_lr = 0.0001
        else:
            current_lr = 0.00001
            
        print(f"Current Learning Rate: {current_lr}")
        
        # 1. Generate Data Self Play
        print("\n[1/4] Generating Batched Self Play Data")
        worker = SelfPlayWorker(best_net, num_concurrent_games=200, mcts_simulations=400)
        new_samples = worker.generate_data(total_games_to_play=100) 
        
        replay_buffer.extend(new_samples)
        print(f"Replay Buffer Capacity: {len(replay_buffer)} of {max_buffer_size} samples")
        
        # 2. Train Candidate on Replay Buffer
        print("\n[2/4] Training Candidate Network")
        candidate_net = ZeroCrossNet().to(device)
        candidate_net.load_state_dict(best_net.state_dict())
        
        candidate_net, opt_state, metrics = train_network(
            candidate_net, 
            list(replay_buffer), 
            batch_size=256, 
            epochs=5, 
            lr=current_lr,
            device=device,
            optimizer_state=optimizer_state
        )
        
        # 3. Arena Evaluation
        print("\n[3/4] Arena Evaluation Candidate vs Champion")
        arena = Arena(candidate_net, best_net, mcts_simulations=100, device=device)
        promoted = arena.evaluate(num_games=40)
        
        print("\nBENCHMARK: Candidate vs Random 20 Games")
        random_win_rate = arena.benchmark_baseline(num_games=20)
        
        # 4. Model Gating and Persistence
        print("\n[4/4] Model Gating")
        if promoted:
            print(f"UPGRADE Candidate promoted Saving to {model_path}")
            best_net.load_state_dict(candidate_net.state_dict())
            optimizer_state = opt_state
            
            torch.save({
                'iteration': current_iter,
                'model_state_dict': best_net.state_dict(),
                'optimizer_state_dict': optimizer_state,
                'learning_rate': current_lr,
                'timestamp': time.time()
            }, model_path)
        else:
            print("REJECTED Candidate failed to hit 55 percent Keeping old champion")
            
        # Log Iteration Metrics to CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_iter, 
                current_lr, 
                metrics.get('pi_loss', 0.0), 
                metrics.get('v_loss', 0.0), 
                metrics.get('entropy', 0.0), 
                round(random_win_rate, 4),
                promoted
            ])
        print(f"Metrics successfully appended to {csv_path}")
        
        # Save the Replay Buffer to disk with versioning metadata
        print("Syncing replay buffer to disk")
        buffer_data = {
            'iteration': current_iter,
            'sample_count': len(replay_buffer),
            'date_saved': time.strftime("%Y %m %d %H %M %S"),
            'data': list(replay_buffer)
        }
        
        # Overwrite the latest operational buffer
        torch.save(buffer_data, buffer_path)
        
        # Save a persistent archive copy for this specific iteration
        archive_path = os.path.join(drive_dir, f"replay_buffer_iter_{current_iter}.pt")
        torch.save(buffer_data, archive_path)
        print(f"Archived chunk saved to {archive_path}")

if __name__ == "__main__":
    run_pipeline(iterations=100)