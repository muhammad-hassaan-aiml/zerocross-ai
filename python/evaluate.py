import torch
import torch.multiprocessing as torch_mp
import math
import random
import numpy as np
import os
import sys

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])

import zerocross_engine
from network import ZeroCrossNet

def _match_worker(gpu_id, net_a_state, net_b_state, net_kwargs, games, sims, label, out_queue):
    """
    Plays one matchup on its own CUDA device in its own subprocess. Mirrors
    self_play.py's _selfplay_worker_process pattern: nets are rebuilt from
    CPU state dicts inside the subprocess (a CUDA tensor can't cross a
    process boundary), and only the small scalar result is sent back
    through the queue.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        net_a = None
        if net_a_state is not None:
            net_a = ZeroCrossNet(**net_kwargs).to(device)
            net_a.load_state_dict(net_a_state)
            net_a.eval()
        net_b = None
        if net_b_state is not None:
            net_b = ZeroCrossNet(**net_kwargs).to(device)
            net_b.load_state_dict(net_b_state)
            net_b.eval()

        evaluator = Evaluator(device=device)
        wr, elo, w, l, d = evaluator.play_match_batched(net_a, net_b, games, sims)
        lcb = evaluator.get_lower_confidence_bound(wr, games)
        out_queue.put({'ok': True, 'label': label, 'wr': wr, 'lcb': lcb, 'elo': elo, 'w': w, 'l': l, 'd': d})
    except Exception as e:
        out_queue.put({'ok': False, 'label': label, 'error': repr(e)})


def run_matches_across_gpus(net_a, matches, gpu_ids, net_kwargs):
    """
    Plays every (label, net_b, games, sims) matchup in `matches`, splitting
    them round-robin across gpu_ids so more than one GPU stays busy during
    evaluation/milestone gauntlets. Previously every matchup ran back to
    back on a single device (gpu_ids[0]) while any other GPU sat idle for
    the entire evaluation phase -- the same problem multi-GPU self-play in
    pipeline.py already solves for data generation, just not for matches.

    net_a is the fixed side of every matchup (the candidate during routine
    evaluation, the current champion during the milestone gauntlet).
    net_b may be None for a matchup against the random-move baseline.

    Falls back to running matches one at a time on a single device
    (gpu_ids[0], or CPU if gpu_ids is empty) when fewer than 2 GPUs are
    usable -- multiprocessing has no benefit there and would only add
    overhead.

    Returns a dict: label -> {'wr', 'lcb', 'elo', 'w', 'l', 'd'}.
    """
    if len(gpu_ids) < 2:
        device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
        evaluator = Evaluator(device=device)
        results = {}
        for label, net_b, games, sims in matches:
            wr, elo, w, l, d = evaluator.play_match_batched(net_a, net_b, games, sims)
            lcb = evaluator.get_lower_confidence_bound(wr, games)
            lcb_str = f" (LCB: {lcb:.2%})" if net_b is not None else ""
            print(f"{label} | WR: {wr:.2%}{lcb_str} | Elo Diff: {elo:+.0f} | W: {w} L: {l} D: {d}")
            results[label] = {'wr': wr, 'lcb': lcb, 'elo': elo, 'w': w, 'l': l, 'd': d}
        return results

    print(f"Running {len(matches)} matchup(s) across {len(gpu_ids)} GPUs {gpu_ids}...")

    ctx = torch_mp.get_context('spawn')
    net_a_state_cpu = {k: v.cpu() for k, v in net_a.state_dict().items()} if net_a is not None else None

    procs, queues, labels, assigned_gpu, has_net_b = [], [], [], [], []
    for idx, (label, net_b, games, sims) in enumerate(matches):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        net_b_state_cpu = {k: v.cpu() for k, v in net_b.state_dict().items()} if net_b is not None else None
        q = ctx.Queue()
        p = ctx.Process(
            target=_match_worker,
            args=(gpu_id, net_a_state_cpu, net_b_state_cpu, net_kwargs, games, sims, label, q)
        )
        p.start()
        procs.append(p)
        queues.append(q)
        labels.append(label)
        assigned_gpu.append(gpu_id)
        has_net_b.append(net_b is not None)

    results = {}
    for p, q, label, gpu_id, net_b_present in zip(procs, queues, labels, assigned_gpu, has_net_b):
        status = q.get()   # blocks until that matchup's worker signals done
        p.join()
        if not status.get('ok'):
            raise RuntimeError(f"Evaluation matchup '{label}' on GPU {gpu_id} failed: {status.get('error')}")
        lcb_str = f" (LCB: {status['lcb']:.2%})" if net_b_present else ""
        print(f"{label} [GPU {gpu_id}] | WR: {status['wr']:.2%}{lcb_str} | "
              f"Elo Diff: {status['elo']:+.0f} | W: {status['w']} L: {status['l']} D: {status['d']}")
        results[label] = {k: status[k] for k in ('wr', 'lcb', 'elo', 'w', 'l', 'd')}
    return results


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
                             random_baseline_sims=None, last_random_baseline_wr=0.0,
                             champion_names=None, gpu_ids=None, net_kwargs=None):
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

        champion_names: optional list, same length/order as champion_nets,
        giving a human-readable identifier for each opponent (e.g. "Latest
        Champion (iter 227)", "Historical Champion (champion_gen_173)",
        "Sentinel (iter 226)"), printed in brackets after "Candidate vs" so
        it's obvious at a glance which specific checkpoint the candidate
        played -- not just "Historical Champion 1". Any index this doesn't
        cover (or if the whole argument is omitted) falls back to the old
        generic "Latest Champion" / "Historical Champion N" label.

        gpu_ids, net_kwargs: when gpu_ids has more than one entry, every
        matchup this call plays (random-baseline check included) is spread
        across those GPUs via run_matches_across_gpus instead of running
        back to back on self.device -- so a second (third, ...) GPU isn't
        left idle for the whole evaluation phase the way it previously was.
        net_kwargs (the ZeroCrossNet constructor kwargs, e.g. num_res_blocks/
        num_channels) is required in that case so each matchup's subprocess
        can rebuild the nets from CPU state dicts. Omit both, or pass a
        single-GPU list, to keep the original sequential behavior on
        self.device.
        """
        print(f"\nStarting Comprehensive Evaluation ({games_per_match} games per matchup)")

        def label_for(idx):
            if champion_names and idx < len(champion_names) and champion_names[idx]:
                return f"Candidate vs {champion_names[idx]}"
            return f"Candidate vs {'Latest Champion' if idx == 0 else f'Historical Champion {idx}'}"

        has_champions = bool(champion_nets) and champion_nets[0] is not None

        # Every matchup this call needs to play, built up front so parallel
        # mode can dispatch all of them (random baseline included) across
        # every usable GPU at once instead of one matchup at a time.
        matches = []
        if check_random_baseline:
            rb_games = random_baseline_games if random_baseline_games is not None else games_per_match
            rb_sims = random_baseline_sims if random_baseline_sims is not None else sims
            matches.append(("Candidate vs Random Baseline", None, rb_games, rb_sims))

        champ_indices = []
        if has_champions:
            for idx, champ in enumerate(champion_nets):
                if champ is None:
                    continue
                matches.append((label_for(idx), champ, games_per_match, sims))
                champ_indices.append(idx)

        parallel = gpu_ids is not None and len(gpu_ids) > 1 and net_kwargs is not None
        if parallel:
            results = run_matches_across_gpus(candidate_net, matches, gpu_ids, net_kwargs)
        else:
            results = {}
            for match_idx, (label, net_b, games, s) in enumerate(matches, start=1):
                print(f"\nMatchup {match_idx}: {label} ({games} games, {s} sims)")
                wr, elo, w, l, d = self.play_match_batched(candidate_net, net_b, games, s)
                lcb = self.get_lower_confidence_bound(wr, games)
                lcb_str = f" (LCB: {lcb:.2%})" if net_b is not None else ""
                print(f"{label} | WR: {wr:.2%}{lcb_str} | Elo Diff: {elo:+.0f} | W: {w} L: {l} D: {d}")
                results[label] = {'wr': wr, 'lcb': lcb, 'elo': elo, 'w': w, 'l': l, 'd': d}

        if check_random_baseline:
            cand_vs_rand_wr = results["Candidate vs Random Baseline"]['wr']
        else:
            cand_vs_rand_wr = last_random_baseline_wr
            print(f"\nCandidate vs Random Baseline -- SKIPPED this iteration (reusing last measured win "
                  f"rate {cand_vs_rand_wr:.2%}). This match never affects promotion, so it doesn't need to "
                  f"run at full cost every single iteration.")

        if not has_champions:
            print("\nNo champion provided. Candidate becomes first champion by default.")
            return True, cand_vs_rand_wr, 1.0, 0.0, 1.0, 1.0

        promoted = True
        primary_champ_wr = 0.0
        primary_champ_elo = 0.0
        # Worst RAW win rate across EVERY opponent faced this call. Kept for
        # logging/trend purposes only -- raw win rate says nothing about how
        # many games it's based on, so it's not safe to gate any decision on
        # this alone (a 55% raw win rate over 10 games and over 500 games are
        # very different levels of evidence). See min_champ_lcb below for the
        # value that should actually be used for gating decisions.
        min_champ_wr = 1.0
        # Worst LOWER-CONFIDENCE-BOUND across every opponent faced this call.
        # This is the statistically honest version of min_champ_wr: it already
        # accounts for sample size, so "0.51 LCB over 300 games" and "0.51 LCB
        # over 20 games" both mean the same thing (true win rate is very
        # likely >= 51%), unlike raw win rate where those two cases carry
        # wildly different amounts of evidence. Callers gating a decision
        # (e.g. pipeline.py's forced-promotion check) should use this, not
        # min_champ_wr, and should compare it against the SAME threshold
        # (0.50) used by the normal per-opponent gate just below -- so an
        # "escape hatch" for stalled candidates never gets to apply a looser
        # statistical bar than ordinary promotion does.
        min_champ_lcb = 1.0

        for idx in champ_indices:
            r = results[label_for(idx)]

            if idx == 0:
                primary_champ_wr = r['wr']
                primary_champ_elo = r['elo']

            min_champ_wr = min(min_champ_wr, r['wr'])
            min_champ_lcb = min(min_champ_lcb, r['lcb'])

            if r['lcb'] <= 0.50:
                promoted = False

        return promoted, cand_vs_rand_wr, primary_champ_wr, primary_champ_elo, min_champ_wr, min_champ_lcb

    def round_robin(self, nets, sims=50, games_per_match=20):
        """
        Play every named net in `nets` (dict: name -> nn.Module) against every
        other one, games_per_match games per pairing (split evenly as each
        side playing first), and return a results table.

        This is the tool for "which of these checkpoints is actually best"
        questions that a single candidate-vs-champion gate can't answer --
        e.g. comparing the current champion against several old milestones
        and the pinned sentinel all at once (see arena.py). It does not feed
        into any promotion decision by itself; it's for manual/periodic
        inspection.

        Returns a list of dicts, one per unordered pair:
          {
            "a": name_a, "b": name_b,
            "wr_a": <A's win rate over both halves>,
            "lcb_a": <lower-confidence-bound on wr_a>,
            "elo_diff": <A's Elo diff vs B>,
            "wins_a", "wins_b", "draws"
          }
        Printed as a running log as it goes, since a full round robin over
        many checkpoints can take a while.
        """
        names = list(nets.keys())
        results = []

        half = max(1, games_per_match // 2)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a, name_b = names[i], names[j]
                net_a, net_b = nets[name_a], nets[name_b]

                print(f"\n{name_a} vs {name_b} ({games_per_match} games, {sims} sims)...")

                wr1, _, w1, l1, d1 = self.play_match_batched(net_a, net_b, half, sims)
                # Play the second half with sides swapped so neither checkpoint
                # gets a first-move-advantage artifact baked into the result.
                wr2, _, l2, w2, d2 = self.play_match_batched(net_b, net_a, half, sims)

                total_games = half * 2
                wins_a = w1 + w2
                wins_b = l1 + l2
                draws = d1 + d2
                wr_a = (wins_a + 0.5 * draws) / total_games
                lcb_a = self.get_lower_confidence_bound(wr_a, total_games)
                elo_diff = self.calculate_elo_diff(wr_a)

                print(f"  {name_a} vs {name_b} | WR({name_a}): {wr_a:.2%} (LCB: {lcb_a:.2%}) | "
                      f"Elo Diff: {elo_diff:+.0f} | W:{wins_a} L:{wins_b} D:{draws}")

                results.append({
                    "a": name_a, "b": name_b,
                    "wr_a": wr_a, "lcb_a": lcb_a, "elo_diff": elo_diff,
                    "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
                })

        return results

if __name__ == "__main__":
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    print(f"Testing Batched Evaluator Harness on {device}")
    
    cand = ZeroCrossNet()
    champ = ZeroCrossNet()
    
    evaluator = Evaluator(device=device)
    
    promoted, rand_wr, champ_wr, elo_diff, min_champ_wr, min_champ_lcb = evaluator.run_full_evaluation(
        candidate_net=cand, 
        champion_nets=[champ], 
        sims=10, 
        games_per_match=2
    )
    
    print(f"\nEvaluator Test Complete. Promoted: {promoted} (min_champ_lcb={min_champ_lcb:.2%})")