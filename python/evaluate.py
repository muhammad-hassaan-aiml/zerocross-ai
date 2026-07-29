import torch
import math
import random
import numpy as np
import os
import sys

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])

import zerocross_engine
from network import ZeroCrossNet

class RandomAgent:
    """Baseline agent that selects uniformly random legal moves."""
    def select_move(self, state):
        mask = state.legal_mask()
        valid_moves = [i for i, is_legal in enumerate(mask) if is_legal]
        return random.choice(valid_moves)

class NeuralMCTSAgent:
    """Agent that uses the C++ MCTSTree guided by the Neural Network."""
    def __init__(self, net, simulations, device):
        self.net = net
        self.sims = simulations
        self.device = device
        if self.net:
            self.net.eval().to(self.device)

    def select_move(self, state):
        tree = zerocross_engine.MCTSTree(state, False)
        
        while not tree.is_done(self.sims):
            leaf = tree.request_leaf()
            if leaf is not None:
                state_tensor = torch.tensor(leaf, dtype=torch.float32).view(1, 6, 9, 9).to(self.device)
                
                with torch.no_grad():
                    logits, values = self.net(state_tensor)
                
                # Submit raw logits. The C++ engine handles the legal masking and exp() internally.
                tree.submit_result(logits.cpu().numpy()[0].tolist(), float(values.cpu().numpy()[0][0]))
        
        # Temperature 0.0 triggers the safe argmax branch in C++ to prevent float overflow
        raw_policy = tree.root_policy(0.0)
        mcts_policy = np.array(raw_policy, dtype=np.float64)
        
        # Safe normalization fallback
        policy_sum = np.sum(mcts_policy)
        if policy_sum > 0:
            mcts_policy /= policy_sum
            
        return np.argmax(mcts_policy)

class Evaluator:
    def __init__(self, device='cpu'):
        self.device = device

    def calculate_elo_diff(self, win_rate):
        """Calculates Elo difference based on win rate. Clamped to prevent math domain errors."""
        win_rate = max(0.01, min(0.99, win_rate))
        return -400.0 * math.log10((1.0 / win_rate) - 1.0)

    def play_match(self, p1, p2, num_games):
        p1_wins, p2_wins, draws = 0, 0, 0
        
        for g in range(num_games):
            # Swap colors to eliminate first-player bias
            if g % 2 == 0:
                x_agent, o_agent = p1, p2
            else:
                x_agent, o_agent = p2, p1
                
            state = zerocross_engine.GameState()
            while not state.is_terminal():
                current_player = state.get_current_player()
                active_agent = x_agent if current_player == 1 else o_agent
                
                action = active_agent.select_move(state)
                state.play(action)
                
            winner = state.get_winner()
            
            # Tally score from Player 1's perspective
            if winner == 1:
                if g % 2 == 0: p1_wins += 1
                else: p2_wins += 1
            elif winner == -1:
                if g % 2 == 0: p2_wins += 1
                else: p1_wins += 1
            else:
                draws += 1
                
        p1_score = p1_wins + 0.5 * draws
        win_rate = p1_score / num_games
        elo_diff = self.calculate_elo_diff(win_rate)
        
        return win_rate, elo_diff, p1_wins, p2_wins, draws

    def run_full_evaluation(self, candidate_net, champion_net, sims=50, games_per_match=20):
        print(f"\nStarting Comprehensive Evaluation ({games_per_match} games per matchup)")
        
        random_agent = RandomAgent()
        cand_agent = NeuralMCTSAgent(candidate_net, sims, self.device)
        champ_agent = NeuralMCTSAgent(champion_net, sims, self.device) if champion_net else None
        
        print("\nMatchup 1: Candidate vs Random Baseline")
        cand_vs_rand_wr, cand_vs_rand_elo, w, l, d = self.play_match(cand_agent, random_agent, games_per_match)
        print(f"Candidate vs Random | WR: {cand_vs_rand_wr:.2%} | Elo Diff: {cand_vs_rand_elo:+.0f} | W: {w} L: {l} D: {d}")
        
        if champ_agent:
            print("\nMatchup 2: Champion vs Random Baseline")
            champ_vs_rand_wr, champ_vs_rand_elo, w, l, d = self.play_match(champ_agent, random_agent, games_per_match)
            print(f"Champion vs Random  | WR: {champ_vs_rand_wr:.2%} | Elo Diff: {champ_vs_rand_elo:+.0f} | W: {w} L: {l} D: {d}")
            
            print("\nMatchup 3: Candidate vs Champion")
            cand_vs_champ_wr, cand_vs_champ_elo, w, l, d = self.play_match(cand_agent, champ_agent, games_per_match)
            print(f"Candidate vs Champ  | WR: {cand_vs_champ_wr:.2%} | Elo Diff: {cand_vs_champ_elo:+.0f} | W: {w} L: {l} D: {d}")
            
            promoted = cand_vs_champ_wr >= 0.55
        else:
            print("\nNo champion provided. Candidate becomes first champion by default.")
            cand_vs_champ_wr = 1.0
            cand_vs_champ_elo = 0.0
            promoted = True
            
        return promoted, cand_vs_rand_wr, cand_vs_champ_wr, cand_vs_champ_elo

if __name__ == "__main__":
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    print(f"Testing Evaluator Harness on {device}")
    
    cand = ZeroCrossNet()
    champ = ZeroCrossNet()
    
    evaluator = Evaluator(device=device)
    
    # Tiny test run to verify logic flows end-to-end
    promoted, rand_wr, champ_wr, elo_diff = evaluator.run_full_evaluation(
        candidate_net=cand, 
        champion_net=champ, 
        sims=10, 
        games_per_match=2
    )
    
    print(f"\nEvaluator Test Complete. Promoted: {promoted}")