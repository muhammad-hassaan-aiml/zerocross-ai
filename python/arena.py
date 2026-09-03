"""
arena.py -- standalone round-robin tournament for comparing ZeroCross
checkpoints on demand (best_model, sentinel, old milestones, champion_gen_*
history, or any .pth files you point it at).

This is deliberately separate from pipeline.py: it never writes to
best_model.pth, pipeline_state.json, or any promotion state -- it only
plays games and reports results. Use it whenever you want an answer to
"is X actually stronger than Y" without running a training iteration,
e.g. sanity-checking a milestone_log.csv warning, or comparing gen_226
against a batch of post-226 checkpoints you rolled back from.

Usage examples:

  # Compare the current champion, the pinned sentinel, and a couple of
  # named checkpoints, 100 games/pairing:
  python arena.py \
      --checkpoint models/best_model.pth \
      --checkpoint models/sentinel_model.pth \
      --checkpoint models/champion_gen_180.pth \
      --games 100 --sims 200

  # Compare every champion_gen_*.pth + milestone_iter_*.pth in a directory:
  python arena.py --glob "models/champion_gen_*.pth" --glob "models/milestone_iter_*.pth" --games 100

Results print as a running log during play, then a final ranked summary
table (by average win rate across all pairings played) and a CSV.
"""
import os
import sys
import glob as globmod
import argparse
import csv

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])

import torch
from network import ZeroCrossNet
from evaluate import Evaluator


def strip_module_prefix(state_dict):
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


def load_checkpoint(path, device, net_kwargs):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
    state_dict = strip_module_prefix(state_dict)
    net = ZeroCrossNet(**net_kwargs).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    iteration = checkpoint.get('iteration') if isinstance(checkpoint, dict) else None
    return net, iteration


def name_for(path, iteration):
    base = os.path.splitext(os.path.basename(path))[0]
    return f"{base} (iter {iteration})" if iteration is not None else base


def main():
    parser = argparse.ArgumentParser(description="ZeroCross Arena -- round-robin checkpoint comparison")
    parser.add_argument("--checkpoint", action="append", default=[], help="Path to a .pth checkpoint to include. Repeat for multiple")
    parser.add_argument("--glob", action="append", default=[], dest="globs", help="Glob pattern matching multiple .pth checkpoints to include (e.g. 'models/champion_gen_*.pth'). Repeat for multiple patterns")
    parser.add_argument("--games", type=int, default=100, help="Games per pairing (split evenly across both sides playing first)")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--num-res-blocks", type=int, default=None, help="Override ZeroCrossNet residual block count (must match how the checkpoints were trained)")
    parser.add_argument("--num-channels", type=int, default=None, help="Override ZeroCrossNet channel width (must match how the checkpoints were trained)")
    parser.add_argument("--output-csv", type=str, default="arena_results.csv", help="Where to write the pairwise results table")
    args = parser.parse_args()

    paths = list(args.checkpoint)
    for pattern in args.globs:
        paths.extend(sorted(globmod.glob(pattern)))
    # De-duplicate while preserving order (a checkpoint could get pulled in
    # by both an explicit --checkpoint and an overlapping --glob).
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    if len(paths) < 2:
        print("Need at least 2 checkpoints to run a tournament (got "
              f"{len(paths)}). Pass more --checkpoint / --glob arguments.")
        sys.exit(1)

    net_kwargs = {}
    if args.num_res_blocks is not None:
        net_kwargs['num_res_blocks'] = args.num_res_blocks
    if args.num_channels is not None:
        net_kwargs['num_channels'] = args.num_channels

    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
    print(f"Arena running on {device}. Loading {len(paths)} checkpoint(s)...")

    nets = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"  SKIPPING {path}: file not found")
            continue
        try:
            net, iteration = load_checkpoint(path, device, net_kwargs)
        except Exception as e:
            print(f"  SKIPPING {path}: failed to load ({e!r})")
            continue
        display_name = name_for(path, iteration)
        nets[display_name] = net
        print(f"  Loaded {display_name}")

    if len(nets) < 2:
        print("Fewer than 2 checkpoints loaded successfully -- nothing to compare.")
        sys.exit(1)

    evaluator = Evaluator(device=device)
    results = evaluator.round_robin(nets, sims=args.sims, games_per_match=args.games)

    with open(args.output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B", "WinRate_A", "LCB_A", "EloDiff_A", "Wins_A", "Wins_B", "Draws"])
        for r in results:
            writer.writerow([r["a"], r["b"], round(r["wr_a"], 4), round(r["lcb_a"], 4),
                              round(r["elo_diff"], 1), r["wins_a"], r["wins_b"], r["draws"]])
    print(f"\nPairwise results written to {args.output_csv}")

    # Simple ranking: average win rate across every pairing each checkpoint
    # played (not a full Elo solve -- good enough for "who's clearly ahead"
    # at a glance; use the pairwise CSV for anything more rigorous).
    scores = {name: [] for name in nets}
    for r in results:
        scores[r["a"]].append(r["wr_a"])
        scores[r["b"]].append(1.0 - r["wr_a"])

    print("\nRANKING (avg win rate across all pairings played):")
    ranked = sorted(scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1]) if kv[1] else 0.0, reverse=True)
    for rank, (name, wrs) in enumerate(ranked, start=1):
        avg = sum(wrs) / len(wrs) if wrs else 0.0
        print(f"  {rank}. {name:<40s} avg WR: {avg:.2%}  ({len(wrs)} pairing(s))")


if __name__ == "__main__":
    main()