import torch
import torch.nn.functional as F
import numpy as np
import random
import zerocross_engine
from network import ZeroCrossNet

class Arena:
    def __init__(self, net1, net2=None, mcts_simulations=100, device='cpu'):
        self.net1 = net1  # Candidate Model
        self.net2 = net2  # Best Model (can be None if only benchmarking)
        self.sims = mcts_simulations
        self.device = device
        
        if self.net1: self.net1.eval().to(self.device)
        if self.net2: self.net2.eval().to(self.device)

    def _evaluate_leaf(self, net, leaf, legal_mask):
        state_tensor = torch.tensor(leaf, dtype=torch.float32).view(1, 6, 9, 9).to(self.device)
        mask_tensor = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, values = net(state_tensor)
            masked_logits = logits.masked_fill(~mask_tensor, -1e9)
            probs = F.softmax(masked_logits, dim=-1)
            
        return probs.cpu().numpy()[0].tolist(), float(values.cpu().numpy()[0][0])

    def play_game(self, p1_net, p2_net):
        """Plays a single evaluation game. If a player is 'random', they play random legal moves."""
        state = zerocross_engine.GameState()
        tree = zerocross_engine.MCTSTree(state, False) # Disable Dirichlet noise for arena
        
        ply = 0
        while not state.is_terminal():
            current_player = state.get_current_player()
            active_net = p1_net if current_player == 1 else p2_net
            
            # --- Random Baseline Logic ---
            if active_net == 'random':
                mask = state.legal_mask()
                valid_moves = [i for i, is_legal in enumerate(mask) if is_legal]
                action = random.choice(valid_moves)
                
            # --- MCTS Network Logic ---
            else:
                while not tree.is_done(self.sims):
                    leaf = tree.request_leaf()
                    if leaf is not None:
                        policy, val = self._evaluate_leaf(active_net, leaf, state.legal_mask())
                        tree.submit_result(policy, val)
                
                # Always fetch safe, proportional probabilities from C++
                raw_policy = tree.root_policy(1.0)
                mcts_policy = np.array(raw_policy, dtype=np.float64)
                mcts_policy /= np.sum(mcts_policy)

                # Handle Temperature scaling safely in Python
                if ply < 4:
                    action = np.random.choice(81, p=mcts_policy) # Explore
                else:
                    action = np.argmax(mcts_policy) # Exploit
            
            state.play(action)
            tree.advance(action)
            ply += 1
            
        return state.get_winner()

    def evaluate(self, num_games=20):
        """Plays a tournament swapping starting sides to eliminate first-player bias."""
        if not self.net2:
            raise ValueError("Champion network required for evaluation.")
            
        p1_wins, p2_wins, draws = 0, 0, 0
        half_games = num_games // 2
        
        print(f"--- Starting Arena Evaluation ({num_games} Games) ---")
        
        # Round 1: Candidate = Player 1 (X), Best = Player 2 (O)
        for g in range(half_games):
            winner = self.play_game(self.net1, self.net2)
            if winner == 1: p1_wins += 1
            elif winner == -1: p2_wins += 1
            else: draws += 1
            print(f"Game {g+1}/{num_games} complete | Candidate(X): {p1_wins}, Best(O): {p2_wins}, Draws: {draws}")

        # Round 2: Best = Player 1 (X), Candidate = Player 2 (O)
        for g in range(half_games):
            winner = self.play_game(self.net2, self.net1)
            if winner == -1: p1_wins += 1 # Candidate won as O
            elif winner == 1: p2_wins += 1 # Best won as X
            else: draws += 1
            print(f"Game {half_games + g + 1}/{num_games} complete | Candidate: {p1_wins}, Best: {p2_wins}, Draws: {draws}")

        total_points = p1_wins + (0.5 * draws)
        win_rate = total_points / num_games
        
        print(f"\nTournament Final Score: Candidate Win Rate = {win_rate:.2%}")
        return win_rate >= 0.55

    def benchmark_baseline(self, num_games=20):
        """New benchmarking evaluation: Candidate vs Random Baseline."""
        p1_wins, p2_wins, draws = 0, 0, 0
        half_games = num_games // 2
        
        print(f"\n--- Benchmarking: Candidate vs Random Agent ({num_games} Games) ---")
        
        for g in range(half_games):
            winner = self.play_game(self.net1, 'random')
            if winner == 1: p1_wins += 1
            elif winner == -1: p2_wins += 1
            else: draws += 1

        for g in range(half_games):
            winner = self.play_game('random', self.net1)
            if winner == -1: p1_wins += 1 
            elif winner == 1: p2_wins += 1
            else: draws += 1

        total_points = p1_wins + (0.5 * draws)
        win_rate = total_points / num_games
        print(f">>> Absolute Win Rate vs Random: {win_rate:.2%}\n")
        return win_rate

if __name__ == "__main__":
    # Smart device selection
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    print(f"Initializing Arena on {device}...")
    
    cand = ZeroCrossNet()
    best = ZeroCrossNet()
    
    # Test run: 2 evaluation games, 2 benchmark games
    arena = Arena(cand, best, mcts_simulations=20, device=device)
    
    print("Testing Gating Evaluation...")
    promoted = arena.evaluate(num_games=2)
    
    print("Testing Random Benchmark...")
    baseline = arena.benchmark_baseline(num_games=2)
    
    print(f"\nTests Complete! Promoted: {promoted}, Random Win Rate: {baseline}")