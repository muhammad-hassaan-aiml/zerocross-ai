import time
import torch
import numpy as np
import os
import sys

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])
import zerocross_engine
from network import ZeroCrossNet
from augment import get_symmetries

class SelfPlayWorker:
    def __init__(self, net, num_concurrent_games=100, mcts_simulations=400, temperature_moves=30):
        self.net = net
        self.num_games = num_concurrent_games
        self.simulations = mcts_simulations
        self.temp_moves = temperature_moves
        
        self.states = [zerocross_engine.GameState() for _ in range(self.num_games)]
        self.parallel_mcts = zerocross_engine.ParallelMCTS(self.num_games, True)
        
        self.game_histories = [[] for _ in range(self.num_games)]
        self.completed_games_data = []
        self.batch_sizes = []
        self.total_augmentation_time = 0.0

        self.device = next(self.net.parameters()).device
        
        self.is_cuda = "cuda" in str(self.device)
        if self.is_cuda:
            torch.backends.cudnn.benchmark = True

    def generate_data(self, total_games_to_play):
        games_completed = 0
        self.batch_sizes = []
        
        while games_completed < total_games_to_play:
            
            # C++ handles the multithreaded tree traversal and returns a zero-copy NumPy array and mapping
            leaves, _ = self.parallel_mcts.request_batch(self.simulations, 8)
            
            if leaves.shape[0] > 0:
                self.batch_sizes.append(leaves.shape[0])
                batch_states = torch.from_numpy(leaves).to(self.device)
                
                with torch.no_grad():
                    if self.is_cuda:
                        with torch.autocast('cuda'):
                            logits, values = self.net(batch_states)
                    else:
                        logits, values = self.net(batch_states)
                        
                    policies_cpu = logits.cpu().numpy()
                    values_cpu = values.cpu().numpy()
                
                # Submit contiguous NumPy arrays directly back to the C++ multithreaded manager
                p_arr = np.ascontiguousarray(policies_cpu, dtype=np.float32)
                v_arr = np.ascontiguousarray(values_cpu.flatten(), dtype=np.float32)
                self.parallel_mcts.submit_batch(p_arr, v_arr)

            for i in range(self.num_games):
                if self.parallel_mcts.is_done(i, self.simulations):
                    temp = 1.0 if len(self.game_histories[i]) < self.temp_moves else 0.0
                    raw_policy = self.parallel_mcts.root_policy(i, temp)
                    
                    mcts_policy = np.array(raw_policy, dtype=np.float64)
                    policy_sum = np.sum(mcts_policy)
                    if policy_sum > 0:
                        mcts_policy /= policy_sum
                    
                    self.game_histories[i].append({
                        'state': self.states[i].encode(),
                        'policy': mcts_policy.tolist(),
                        'player': self.states[i].get_current_player()
                    })
                    
                    action = np.argmax(mcts_policy) if temp == 0.0 else np.random.choice(81, p=mcts_policy)
                    self.states[i].play(action)
                    self.parallel_mcts.advance(i, action)
                    
                    if self.states[i].is_terminal():
                        self._process_completed_game(i)
                        games_completed += 1
                        
                        self.states[i] = zerocross_engine.GameState()
                        self.parallel_mcts.set_state(i, self.states[i])
                        
        self.avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        return self.completed_games_data

    def _process_completed_game(self, game_idx):
        winner = self.states[game_idx].get_winner()
        
        aug_start = time.time()
        for step in self.game_histories[game_idx]:
            reward = 1.0 if step['player'] == winner else (-1.0 if winner != 0 else 0.0)
            
            augmented_steps = get_symmetries(step['state'], step['policy'], reward)
            self.completed_games_data.extend(augmented_steps)
            
        self.total_augmentation_time += (time.time() - aug_start)
        self.game_histories[game_idx] = []

if __name__ == "__main__":
    print("Initializing Neural Network...")
    net = ZeroCrossNet()
    net.eval()
    
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    net.to(device)
    
    print(f"Starting Batched Self-Play on {device}...")
    worker = SelfPlayWorker(net, num_concurrent_games=10, mcts_simulations=50)
    dataset = worker.generate_data(total_games_to_play=2)
    print(f"Successfully generated {len(dataset)} training samples from 2 games!")
    print(f"Average MCTS Batch Size: {worker.avg_batch_size:.2f}")