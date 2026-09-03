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
        ply_counts = [0] * num_games
        
        p1_wins, p2_wins, draws = 0, 0, 0
        active_games = set(range(num_games))
        
        is_cuda = "cuda" in str(self.device)
        if net1: net1.eval().to(self.device)
        if net2: net2.eval().to(self.device)

        parallel_mcts = zerocross_engine.ParallelMCTS(num_games, False)

        while active_games:
            progressed = True
            while progressed:
                progressed = False
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
                        progressed = True
                        continue
                        
                    curr_player = states[i].get_current_player()
                    is_p1_turn = (curr_player == p1_color[i])
                    active_net = net1 if is_p1_turn else net2
                    
                    if active_net is None:
                        mask = states[i].legal_mask()
                        valid = [idx for idx, legal in enumerate(mask) if legal]
                        action = random.choice(valid)
                        states[i].play(action)
                        parallel_mcts.advance(i, action)
                        ply_counts[i] += 1
                        progressed = True

            if not active_games:
                break

            mcts_acted = False
            for i in list(active_games):
                if parallel_mcts.is_done(i, sims):
                    temp = 1.0 if ply_counts[i] < 6 else 0.0
                    raw_policy = parallel_mcts.root_policy(i, temp)
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
                    parallel_mcts.advance(i, action)
                    ply_counts[i] += 1
                    mcts_acted = True

            if mcts_acted:
                continue

            leaves, mapping = parallel_mcts.request_batch(sims, 8)
            
            if leaves.shape[0] > 0:
                all_policies = np.zeros((leaves.shape[0], 81), dtype=np.float32)
                all_values = np.zeros(leaves.shape[0], dtype=np.float32)

                net1_k = []
                net1_leaves = []
                net2_k = []
                net2_leaves = []

                for k in range(leaves.shape[0]):
                    game_idx = mapping[k]
                    curr_player = states[game_idx].get_current_player()
                    is_p1_turn = (curr_player == p1_color[game_idx])
                    if is_p1_turn:
                        net1_k.append(k)
                        net1_leaves.append(leaves[k])
                    else:
                        net2_k.append(k)
                        net2_leaves.append(leaves[k])

                if net1_leaves and net1 is not None:
                    b1_t = torch.from_numpy(np.array(net1_leaves, dtype=np.float32)).to(self.device)
                    with torch.no_grad():
                        if is_cuda:
                            with torch.autocast('cuda'):
                                l1, v1 = net1(b1_t)
                        else:
                            l1, v1 = net1(b1_t)
                        p1_cpu = l1.cpu().numpy()
                        v1_cpu = v1.cpu().numpy()
                    for idx, k in enumerate(net1_k):
                        all_policies[k] = p1_cpu[idx]
                        all_values[k] = v1_cpu[idx][0]

                if net2_leaves and net2 is not None:
                    b2_t = torch.from_numpy(np.array(net2_leaves, dtype=np.float32)).to(self.device)
                    with torch.no_grad():
                        if is_cuda:
                            with torch.autocast('cuda'):
                                l2, v2 = net2(b2_t)
                        else:
                            l2, v2 = net2(b2_t)
                        p2_cpu = l2.cpu().numpy()
                        v2_cpu = v2.cpu().numpy()
                    for idx, k in enumerate(net2_k):
                        all_policies[k] = p2_cpu[idx]
                        all_values[k] = v2_cpu[idx][0]

                parallel_mcts.submit_batch(all_policies, all_values)
                        
        p1_score = p1_wins + 0.5 * draws
        win_rate = p1_score / num_games
        elo_diff = self.calculate_elo_diff(win_rate)
        
        return win_rate, elo_diff, p1_wins, p2_wins, draws

    def run_full_evaluation(self, candidate_net, champion_nets, sims=50, games_per_match=20,
                             check_random_baseline=True, random_baseline_games=None,
                             random_baseline_sims=None, last_random_baseline_wr=0.0):
        """
        check_random_baseline, random_baseline_games, random_baseline_sims:
        the "Candidate vs Random Baseline" matchup below is a sanity metric
        only -- notice its result (cand_vs_rand_wr) never appears in the
        promotion decision further down, only WR/LCB against champion_nets
        does. Once the network is any good at all, "does it still beat
        random moves" stops being informative on any single iteration; it's
        useful as a slow trend line (did something break?), not as a thing
        worth re-measuring at full precision every iteration.

        So this match is controllable independently of the real gating
        matches: check_random_baseline=False skips it entirely for this
        call (reusing last_random_baseline_wr for the return value), and
        random_baseline_games/sims let it run cheaper than the
        games_per_match/sims used for the matches that actually decide
        promotion. Both default to games_per_match/sims (i.e. the old
        behavior) when left as None, so existing callers are unaffected.
        The calling pipeline decides the actual policy (e.g. "only check
        every 10 iterations") -- this method just exposes the knobs.
        """
        print(f"\nStarting Comprehensive Evaluation ({games_per_match} games per matchup)")

        if check_random_baseline:
            rb_games = random_baseline_games if random_baseline_games is not None else games_per_match
            rb_sims = random_baseline_sims if random_baseline_sims is not None else sims
            print(f"\nMatchup 1: Candidate vs Random Baseline ({rb_games} games, {rb_sims} sims)")
            cand_vs_rand_wr, cand_vs_rand_elo, w, l, d = self.play_match_batched(candidate_net, None, rb_games, rb_sims)
            print(f"Candidate vs Random | WR: {cand_vs_rand_wr:.2%} | Elo Diff: {cand_vs_rand_elo:+.0f} | W: {w} L: {l} D: {d}")
        else:
            cand_vs_rand_wr = last_random_baseline_wr
            print(f"\nMatchup 1: Candidate vs Random Baseline -- SKIPPED this iteration (reusing last "
                  f"measured win rate {cand_vs_rand_wr:.2%}). This match never affects promotion, so it "
                  f"doesn't need to run at full cost every single iteration.")

        if not champion_nets or champion_nets[0] is None:
            print("\nNo champion provided. Candidate becomes first champion by default.")
            return True, cand_vs_rand_wr, 1.0, 0.0, 1.0
            
        promoted = True
        primary_champ_wr = 0.0
        primary_champ_elo = 0.0
        # Worst win rate across EVERY opponent faced this call (latest champion,
        # rotating historical pick, and the sentinel if the caller passed one).
        # Callers that need to gate on "did the candidate hold up against
        # everything, not just the most recent champion" (e.g. pipeline.py's
        # forced-promotion check) should use this instead of primary_champ_wr,
        # which only ever reflects champion_nets[0].
        min_champ_wr = 1.0
        
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

            min_champ_wr = min(min_champ_wr, wr)

            if lcb <= 0.50:
                promoted = False
                
        return promoted, cand_vs_rand_wr, primary_champ_wr, primary_champ_elo, min_champ_wr

if __name__ == "__main__":
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    print(f"Testing Batched Evaluator Harness on {device}")
    
    cand = ZeroCrossNet()
    champ = ZeroCrossNet()
    
    evaluator = Evaluator(device=device)
    
    promoted, rand_wr, champ_wr, elo_diff, min_champ_wr = evaluator.run_full_evaluation(
        candidate_net=cand, 
        champion_nets=[champ], 
        sims=10, 
        games_per_match=2
    )
    
    print(f"\nEvaluator Test Complete. Promoted: {promoted}")