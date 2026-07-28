import time
import torch
import os
from network import ZeroCrossNet
from self_play import SelfPlayWorker

def run_profiling():
    # Setup device
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    print("ZERO-CROSS PROFILING RUN")
    print(f"Device: {device}")
    
    net = ZeroCrossNet()
    net.to(device)
    net.eval()
    
    # Auto-scale based on hardware to prevent local CPU lockups
    if device.type == 'cpu':
        print("\nNotice: CPU detected. Scaling down profile run for local testing.")
        num_games = 2
        sims = 20
    else:
        print("\nNotice: CUDA GPU detected. Running full Kaggle-scale stress test.")
        num_games = 100
        sims = 100
    
    print("\nConfiguration:")
    print(f"Concurrent Games: {num_games}")
    print(f"MCTS Simulations per move: {sims}")
    
    worker = SelfPlayWorker(net, num_concurrent_games=num_games, mcts_simulations=sims)
    
    print("\nStarting generation... (this will take a moment)")
    start_time = time.time()
    
    dataset = worker.generate_data(total_games_to_play=num_games)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate metrics
    # D4 symmetry produces 8 augmented samples per actual board position played
    unique_positions = len(dataset) // 8  
    pos_per_sec = unique_positions / elapsed
    games_per_hour = (num_games / elapsed) * 3600
    
    print("\nPROFILING RESULTS")
    print(f"Time Elapsed:      {elapsed:.2f} seconds")
    print(f"Total Games:       {num_games}")
    print(f"Unique Positions:  {unique_positions}")
    print(f"Augmented Samples: {len(dataset)}")
    print(f"Throughput:        {pos_per_sec:.2f} positions/second")
    print(f"Estimated Pace:    {games_per_hour:.0f} games/hour")
    
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU Memory:   {max_mem:.2f} MB")
        
    print("\nProfiling complete! You are ready to launch on Kaggle.")

if __name__ == "__main__":
    run_profiling()