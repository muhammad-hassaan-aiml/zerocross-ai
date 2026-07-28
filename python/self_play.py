import time
import torch
import numpy as np
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
        self.trees = [zerocross_engine.MCTSTree(self.states[i], True) for i in range(self.num_games)]
        
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
            leaf_states, active_indices = [], []
            
            for i in range(self.num_games):
                if not self.trees[i].is_done(self.simulations):
                    leaf = self.trees[i].request_leaf()
                    if leaf is not None:
                        leaf_states.append(leaf)
                        active_indices.append(i)
            
            if len(leaf_states) > 0:
                self.batch_sizes.append(len(leaf_states))
                batch_states = torch.tensor(np.array(leaf_states), dtype=torch.float32).view(-1, 6, 9, 9).to(self.device)
                
                c0 = batch_states[:, 0].flatten(start_dim=1).bool()
                c1 = batch_states[:, 1].flatten(start_dim=1).bool()
                c2 = batch_states[:, 2].flatten(start_dim=1).bool()
                batch_masks = c2 & ~c0 & ~c1
                
                with torch.no_grad():
                    if self.is_cuda:
                        with torch.autocast('cuda'):
                            logits, values = self.net(batch_states)
                    else:
                        logits, values = self.net(batch_states)
                        
                    masked_logits = logits.masked_fill(~batch_masks, -1e9)
                    
                    policies_cpu = masked_logits.cpu().numpy()
                    values_cpu = values.cpu().numpy()
                
                for idx, tree_idx in enumerate(active_indices):
                    self.trees[tree_idx].submit_result(policies_cpu[idx].tolist(), float(values_cpu[idx][0]))

            for i in range(self.num_games):
                if self.trees[i].is_done(self.simulations):
                    temp = 1.0 if len(self.game_histories[i]) < self.temp_moves else 0.0
                    raw_policy = self.trees[i].root_policy(temp)
                    
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
                    self.trees[i].advance(action)
                    
                    if self.states[i].is_terminal():
                        self._process_completed_game(i)
                        games_completed += 1
                        
                        self.states[i] = zerocross_engine.GameState()
                        self.trees[i] = zerocross_engine.MCTSTree(self.states[i], True)
                        
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