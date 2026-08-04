import torch
import math
import random
import numpy as np
import os
import sys

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])

import zerocross_engine
from network import ZeroCrossNet

class Evaluator:
    def __init__(self, device='cpu'):
        self.device = device

    def calculate_elo_diff(self, win_rate):
        win_rate = max(0.01, min(0.99, win_rate))
        return -400.0 * math.log10((1.0 / win_rate) - 1.0)

    def get_lower_confidence_bound(self, win_rate, total_games, z=1.28):
        if total_games == 0:
            return 0.0
        variance = (win_rate * (1.0 - win_rate)) / total_games
        return win_rate - (z * math.sqrt(variance))

    def play_match_batched(self, net1, net2, num_games, sims):
        states = [zerocross_engine.GameState() for _ in range(num_games)]
        p1_color = [1 if g % 2 == 0 else -1 for g in range(num_games)]
        trees = [None] * num_games
        ply_counts = [0] * num_games
        
        p1_wins, p2_wins, draws = 0, 0, 0
        active_games = set(range(num_games))
        
        is_cuda = "cuda" in str(self.device)
        if net1: net1.eval().to(self.device)
        if net2: net2.eval().to(self.device)

        while active_games:
            net1_reqs = []
            net2_reqs = []
            
            for i in list(active_games):
                if states[i].is_terminal():
                    winner = states[i].get_winner()
                    if winner == p1_color[i]: 
                        p1_wins += 1
                    elif winner == -p1_color[i]: 
                        p2_wins += 1
                    else: 
                        draws += 1
                    active_games.remove(i)
                    continue
                    
                curr_player = states[i].get_current_player()
                is_p1_turn = (curr_player == p1_color[i])
                active_net = net1 if is_p1_turn else net2
                
                if active_net is None:
                    mask = states[i].legal_mask()
                    valid = [idx for idx, legal in enumerate(mask) if legal]
                    states[i].play(random.choice(valid))
                    ply_counts[i] += 1
                else:
                    if trees[i] is None:
                        trees[i] = zerocross_engine.MCTSTree(states[i], False)
                        
                    if not trees[i].is_done(sims):
                        leaves = trees[i].request_leaves(8)
                        if leaves.shape[0] > 0:
                            if is_p1_turn:
                                net1_reqs.append((i, leaves))
                            else:
                                net2_reqs.append((i, leaves))
                    else:
                        temp = 1.0 if ply_counts[i] < 6 else 0.0
                        raw_policy = trees[i].root_policy(temp)
                        mcts_policy = np.array(raw_policy, dtype=np.float64)
                        policy_sum = np.sum(mcts_policy)
                        
                        if policy_sum > 0: 
                            mcts_policy /= policy_sum
                        else:
                            mcts_policy = np.ones(81) / 81.0
                            
                        if temp > 0.0:
                            action = np.random.choice(81, p=mcts_policy)
                        else:
                            action = np.argmax(mcts_policy)
                            
                        states[i].play(action)
                        ply_counts[i] += 1
                        trees[i] = None
                        
            for net, reqs in [(net1, net1_reqs), (net2, net2_reqs)]:
                if net is not None and len(reqs) > 0:
                    indices = []
                    leaves_list = []
                    for req in reqs:
                        indices.extend([req[0]] * req[1].shape[0])
                        leaves_list.append(req[1])
                        
                    batch_states_np = np.concatenate(leaves_list, axis=0)
                    batch_states = torch.from_numpy(batch_states_np).to(self.device)
                    
                    with torch.no_grad():
                        if is_cuda:
                            with torch.autocast('cuda'):
                                logits, values = net(batch_states)
                        else:
                            logits, values = net(batch_states)
                            
                        policies_cpu = logits.cpu().numpy()
                        values_cpu = values.cpu().numpy()
                        
                    tree_results = {idx: {'policies': [], 'values': []} for idx in set(indices)}
                    for idx, game_idx in enumerate(indices):
                        tree_results[game_idx]['policies'].append(policies_cpu[idx])
                        tree_results[game_idx]['values'].append(values_cpu[idx][0])
                        
                    for game_idx, res in tree_results.items():
                        p_arr = np.ascontiguousarray(res['policies'], dtype=np.float32)
                        v_arr = np.ascontiguousarray(res['values'], dtype=np.float32)
                        trees[game_idx].submit_results(p_arr, v_arr)
                        
        p1_score = p1_wins + 0.5 * draws
        win_rate = p1_score / num_games
        elo_diff = self.calculate_elo_diff(win_rate)
        
        return win_rate, elo_diff, p1_wins, p2_wins, draws

    def run_full_evaluation(self, candidate_net, champion_nets, sims=50, games_per_match=20):
        print(f"\nStarting Comprehensive Evaluation ({games_per_match} games per matchup)")
        
        print("\nMatchup 1: Candidate vs Random Baseline")
        cand_vs_rand_wr, cand_vs_rand_elo, w, l, d = self.play_match_batched(candidate_net, None, games_per_match, sims)
        print(f"Candidate vs Random | WR: {cand_vs_rand_wr:.2%} | Elo Diff: {cand_vs_rand_elo:+.0f} | W: {w} L: {l} D: {d}")
        
        if not champion_nets or champion_nets[0] is None:
            print("\nNo champion provided. Candidate becomes first champion by default.")
            return True, cand_vs_rand_wr, 1.0, 0.0
            
        promoted = True
        primary_champ_wr = 0.0
        primary_champ_elo = 0.0
        
        for idx, champ in enumerate(champion_nets):
            if champ is None:
                continue
                
            champ_type = "Latest Champion" if idx == 0 else f"Historical Champion {idx}"
            print(f"\nMatchup {idx + 2}: Candidate vs {champ_type}")
            
            wr, elo, w, l, d = self.play_match_batched(candidate_net, champ, games_per_match, sims)
            lcb = self.get_lower_confidence_bound(wr, games_per_match)
            
            print(f"Candidate vs {champ_type} | WR: {wr:.2%} (LCB: {lcb:.2%}) | Elo Diff: {elo:+.0f} | W: {w} L: {l} D: {d}")
            
            if idx == 0:
                primary_champ_wr = wr
                primary_champ_elo = elo
                
            if lcb <= 0.50:
                promoted = False
                
        return promoted, cand_vs_rand_wr, primary_champ_wr, primary_champ_elo

if __name__ == "__main__":
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    print(f"Testing Batched Evaluator Harness on {device}")
    
    cand = ZeroCrossNet()
    champ = ZeroCrossNet()
    
    evaluator = Evaluator(device=device)
    
    promoted, rand_wr, champ_wr, elo_diff = evaluator.run_full_evaluation(
        candidate_net=cand, 
        champion_nets=[champ], 
        sims=10, 
        games_per_match=2
    )
    
    print(f"\nEvaluator Test Complete. Promoted: {promoted}")