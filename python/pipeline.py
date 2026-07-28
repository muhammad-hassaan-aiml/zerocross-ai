import os
import csv
import time
import sys
import torch
from collections import deque
from network import ZeroCrossNet
from self_play import SelfPlayWorker
from train import train_network
from evaluate import Evaluator

def estimate_buffer_memory_mb(buffer):
    """Calculates the exact byte footprint of the Python replay buffer and converts to MB."""
    if not buffer:
        return 0.0
    
    sample = buffer[0]
    # Calculate bytes for state list, policy list, and reward float, including pointers
    state_size = sys.getsizeof(sample[0]) + sum(sys.getsizeof(x) for x in sample[0])
    policy_size = sys.getsizeof(sample[1]) + sum(sys.getsizeof(x) for x in sample[1])
    reward_size = sys.getsizeof(sample[2])
    tuple_size = sys.getsizeof(sample)
    
    bytes_per_sample = state_size + policy_size + reward_size + tuple_size
    return (bytes_per_sample * len(buffer)) / (1024 ** 2)

def run_pipeline(iterations=100, max_buffer_size=500000):
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
            
    print(f"Starting ZeroCross Training Pipeline on {device}")
    
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
    
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "LR", "PI_Loss", "V_Loss", "Entropy", "WinRate_vs_Random", "Elo_Diff_vs_Champ", "Promoted"])

    best_net = ZeroCrossNet()
    optimizer_state = None
    start_iteration = 0
    
    if os.path.exists(model_path):
        print(f"Loading existing champion from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            best_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    start_iteration = sum(1 for row in f) - 1
        else:
            best_net.load_state_dict(checkpoint)
            
    best_net.to(device)
    replay_buffer = deque(maxlen=max_buffer_size)
    
    if os.path.exists(buffer_path):
        print(f"Loading historical replay buffer from {buffer_path}")
        checkpoint = torch.load(buffer_path, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'data' in checkpoint:
            replay_buffer.extend(checkpoint['data'])
        else:
            replay_buffer.extend(checkpoint)
            
        print(f"Restored {len(replay_buffer)} historical samples to memory")
        
    for i in range(start_iteration, start_iteration + iterations):
        iter_start_time = time.time()
        current_iter = i + 1
        print(f"\nALPHAZERO ITERATION {current_iter} of {start_iteration + iterations}")
        
        if current_iter <= 20:
            current_lr = 0.001
        elif current_iter <= 50:
            current_lr = 0.0001
        else:
            current_lr = 0.00001
            
        print(f"Current Learning Rate: {current_lr}")
        
        print("\n[1/4] Generating Batched Self Play Data")
        gen_start = time.time()
        worker = SelfPlayWorker(best_net, num_concurrent_games=10, mcts_simulations=50)
        new_samples = worker.generate_data(total_games_to_play=2) 
        gen_duration = time.time() - gen_start
        
        replay_buffer.extend(new_samples)
        buffer_mb = estimate_buffer_memory_mb(replay_buffer)
        print(f"Replay Buffer Capacity: {len(replay_buffer)} of {max_buffer_size} samples (Approx {buffer_mb:.2f} MB RAM)")
        
        print("\n[2/4] Training Candidate Network")
        train_start = time.time()
        candidate_net = ZeroCrossNet().to(device)
        candidate_net.load_state_dict(best_net.state_dict())
        
        candidate_net, opt_state, metrics = train_network(
            candidate_net, 
            list(replay_buffer), 
            batch_size=32, 
            epochs=2, 
            lr=current_lr,
            device=device,
            optimizer_state=optimizer_state
        )
        train_duration = time.time() - train_start
        
        print("\n[3/4] Comprehensive Evaluation")
        eval_start = time.time()
        evaluator = Evaluator(device=device)
        promoted, rand_wr, champ_wr, elo_diff = evaluator.run_full_evaluation(
            candidate_net=candidate_net,
            champion_net=best_net,
            sims=20,
            games_per_match=2
        )
        eval_duration = time.time() - eval_start
        
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
            
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_iter, 
                current_lr, 
                metrics.get('pi_loss', 0.0), 
                metrics.get('v_loss', 0.0), 
                metrics.get('entropy', 0.0), 
                round(rand_wr, 4),
                round(elo_diff, 1),
                promoted
            ])
        print(f"Metrics successfully appended to {csv_path}")
        
        print("Syncing replay buffer to disk")
        buffer_data = {
            'iteration': current_iter,
            'sample_count': len(replay_buffer),
            'date_saved': time.strftime("%Y %m %d %H %M %S"),
            'data': list(replay_buffer)
        }
        
        torch.save(buffer_data, buffer_path)
        
        archive_path = os.path.join(drive_dir, f"replay_buffer_iter_{current_iter}.pt")
        torch.save(buffer_data, archive_path)
        print(f"Archived chunk saved to {archive_path}")
        
        max_archives = 3
        old_archive_path = os.path.join(drive_dir, f"replay_buffer_iter_{current_iter - max_archives}.pt")
        if os.path.exists(old_archive_path):
            os.remove(old_archive_path)
            print(f"Deleted old archive {old_archive_path} to save space")
        
        total_iter_duration = time.time() - iter_start_time
        
        print("\nITERATION BENCHMARK REPORT")
        print(f"Average MCTS Batch Size: {worker.avg_batch_size:.2f}")
        print(f"Data Generation Time:    {gen_duration:.2f} sec")
        print(f"Network Training Time:   {train_duration:.2f} sec")
        print(f"Evaluation Time:         {eval_duration:.2f} sec")
        print(f"Total Iteration Time:    {total_iter_duration:.2f} sec")

if __name__ == "__main__":
    run_pipeline(iterations=1)