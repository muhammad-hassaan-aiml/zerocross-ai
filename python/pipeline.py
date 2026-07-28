import os
import csv
import time
import sys
import argparse
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
    state_size = sys.getsizeof(sample[0]) + sum(sys.getsizeof(x) for x in sample[0])
    policy_size = sys.getsizeof(sample[1]) + sum(sys.getsizeof(x) for x in sample[1])
    reward_size = sys.getsizeof(sample[2])
    tuple_size = sys.getsizeof(sample)
    
    bytes_per_sample = state_size + policy_size + reward_size + tuple_size
    return (bytes_per_sample * len(buffer)) / (1024 ** 2)

def run_pipeline(iterations=100, max_buffer_size=500000, do_generate=True, do_train=True, do_evaluate=True,
                 concurrent_games=10, mcts_sims=50, eval_games=2, eval_sims=20):
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
            start_iteration = checkpoint.get('iteration', 0)
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
        
        metrics = {'pi_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0}
        rand_wr, champ_wr, elo_diff = 0.0, 0.0, 0.0
        promoted = False
        candidate_net = best_net
        opt_state = optimizer_state
        
        gen_duration = train_duration = eval_duration = aug_duration = avg_batch_size = 0.0
        
        if do_generate:
            print("\n[1/4] Generating Batched Self Play Data")
            gen_start = time.time()
            worker = SelfPlayWorker(best_net, num_concurrent_games=concurrent_games, mcts_simulations=mcts_sims, temperature_moves=30)
            new_samples = worker.generate_data(total_games_to_play=concurrent_games)
            gen_duration = time.time() - gen_start
            aug_duration = worker.total_augmentation_time
            avg_batch_size = worker.avg_batch_size
            
            replay_buffer.extend(new_samples)
            buffer_mb = estimate_buffer_memory_mb(replay_buffer)
            print(f"Replay Buffer Capacity: {len(replay_buffer)} of {max_buffer_size} samples (Approx {buffer_mb:.2f} MB RAM)")
            
            print("Syncing replay buffer to disk...")
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
        
        if do_train:
            print("\n[2/4] Training Candidate Network")
            if len(replay_buffer) == 0:
                print("Skipping training: Replay buffer is empty.")
                do_train = False
            else:
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
        
        if do_evaluate:
            print(f"\n[3/4] Comprehensive Evaluation ({eval_games} games/match)")
            eval_start = time.time()
            evaluator = Evaluator(device=device)
            promoted, rand_wr, champ_wr, elo_diff = evaluator.run_full_evaluation(
                candidate_net=candidate_net,
                champion_net=best_net,
                sims=eval_sims,
                games_per_match=eval_games
            )
            eval_duration = time.time() - eval_start
        
        if do_train:
            print("\n[4/4] Model Gating")
            if not do_evaluate:
                promoted = True
                print("Evaluation skipped. Force promoting candidate.")
                
            if promoted:
                print(f"UPGRADE Candidate promoted! Saving to {model_path}")
                best_net.load_state_dict(candidate_net.state_dict())
                optimizer_state = opt_state
                
                checkpoint_data = {
                    'iteration': current_iter,
                    'model_state_dict': best_net.state_dict(),
                    'optimizer_state_dict': optimizer_state,
                    'learning_rate': current_lr,
                    'timestamp': time.time()
                }
                
                torch.save(checkpoint_data, model_path)
                
                history_path = os.path.join(drive_dir, f"champion_gen_{current_iter}.pth")
                torch.save(checkpoint_data, history_path)
                print(f"Historical champion archived to {history_path}")
                
            else:
                print("REJECTED Candidate failed to hit 55 percent. Keeping old champion.")
        
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
        
        total_iter_duration = time.time() - iter_start_time
        
        print("\nITERATION BENCHMARK REPORT")
        if do_generate:
            print(f"Average MCTS Batch Size: {avg_batch_size:.2f}")
            print(f"Data Generation Time:    {gen_duration:.2f} sec")
            print(f"Augmentation Time:  {aug_duration:.2f} sec")
        if do_train:
            print(f"Network Training Time:   {train_duration:.2f} sec")
        if do_evaluate:
            print(f"Evaluation Time:         {eval_duration:.2f} sec")
        print(f"Total Iteration Time:    {total_iter_duration:.2f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroCross Training Pipeline")
    parser.add_argument("--iterations", type=int, default=1, help="Number of pipeline iterations")
    parser.add_argument("--generate-only", action="store_true", help="Only generate self-play data and update replay buffer")
    parser.add_argument("--train-only", action="store_true", help="Only train the network on existing buffer and force promote")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate the current best model")
    
    parser.add_argument("--concurrent-games", type=int, default=10, help="Parallel games for self-play")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move during self-play")
    parser.add_argument("--eval-games", type=int, default=2, help="Games per matchup in evaluation (e.g. 40 on Kaggle)")
    parser.add_argument("--eval-sims", type=int, default=20, help="MCTS simulations per move during evaluation")
    
    args = parser.parse_args()
    
    do_generate = True
    do_train = True
    do_evaluate = True
    
    if args.generate_only:
        do_train = False
        do_evaluate = False
    elif args.train_only:
        do_generate = False
        do_evaluate = False
    elif args.evaluate_only:
        do_generate = False
        do_train = False
        
    run_pipeline(
        iterations=args.iterations,
        do_generate=do_generate,
        do_train=do_train,
        do_evaluate=do_evaluate,
        concurrent_games=args.concurrent_games,
        mcts_sims=args.mcts_sims,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims
    )