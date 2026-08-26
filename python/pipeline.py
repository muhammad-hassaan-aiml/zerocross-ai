import os
import csv
import json
import math
import time
import sys
import argparse
import random
import shutil

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])

import torch
from collections import deque
from network import ZeroCrossNet
from self_play import SelfPlayWorker
from train import train_network
from evaluate import Evaluator


def estimate_buffer_memory_mb(buffer):
    if not buffer:
        return 0.0

    sample = buffer[0]
    state_size = sample[0].nbytes if hasattr(sample[0], 'nbytes') else sys.getsizeof(sample[0]) + sum(sys.getsizeof(x) for x in sample[0])
    policy_size = sample[1].nbytes if hasattr(sample[1], 'nbytes') else sys.getsizeof(sample[1]) + sum(sys.getsizeof(x) for x in sample[1])
    reward_size = sample[2].nbytes if hasattr(sample[2], 'nbytes') else sys.getsizeof(sample[2])
    tuple_size = sys.getsizeof(sample)

    bytes_per_sample = state_size + policy_size + reward_size + tuple_size
    return (bytes_per_sample * len(buffer)) / (1024 ** 2)


def get_usable_gpus():
    """
    Returns the indices of visible CUDA devices with compute capability >= 6
    (Pascal or newer). Older GPUs are excluded because autocast/GradScaler
    mixed-precision assumes reasonably modern tensor-core-friendly hardware,
    same rule the notebook's own GPU-verification cell uses for device 0 --
    this just applies it consistently across every visible device instead of
    only the first one, which matters once DataParallel is in play.
    """
    if not torch.cuda.is_available():
        return []
    usable = []
    for idx in range(torch.cuda.device_count()):
        major, _ = torch.cuda.get_device_capability(idx)
        if major >= 6:
            usable.append(idx)
        else:
            print(f"Skipping GPU {idx} ({torch.cuda.get_device_name(idx)}): "
                  f"compute capability < 6, not used for training.")
    return usable


def safe_torch_load(path, map_location=None):
    """
    Load a checkpoint/buffer file, tolerating files a Kaggle session left
    half-written (killed mid-save, disk quota hit mid-write, etc). A plain
    torch.load() on a truncated file raises and kills the whole run -- which
    then can't even resume, because the very file it needs to resume from is
    the one that's broken.

    On failure this quarantines the bad file (renames it with a
    .corrupt-<timestamp> suffix so it stops being picked up) and returns
    None. Callers treat None exactly like "file does not exist" and fall
    back to a fresh start for that piece of state, rather than crashing.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        quarantine_path = f"{path}.corrupt-{int(time.time())}"
        print(f"WARNING: could not load {path} ({e!r}). "
              f"Quarantining it to {quarantine_path} and continuing as if it were missing.")
        try:
            os.rename(path, quarantine_path)
        except OSError as rename_err:
            print(f"  (also failed to quarantine it: {rename_err!r})")
        return None


def strip_module_prefix(state_dict):
    """Undo nn.DataParallel's 'module.' key prefix so a checkpoint saved
    from a multi-GPU run can be loaded into a plain (or differently-sized)
    GPU setup on a later run."""
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


def prune_numbered_files(drive_dir, prefix, suffix, keep_last_n):
    """
    Keep only the keep_last_n highest-numbered '<prefix><N><suffix>' files
    in drive_dir, deleting the rest.

    This replaces a fragile arithmetic version of the same idea (delete
    "current_iter - keep_last_n*interval") which only worked if a file were
    guaranteed to exist at that exact number. That assumption silently broke
    for champion_gen_*.pth files, since those are only written on iterations
    that actually got promoted -- with gaps (especially long ones during a
    rejection streak), the computed filename usually doesn't correspond to
    any file that was ever written, so nothing gets deleted and champions
    accumulate on disk indefinitely. Listing and sorting the files that
    actually exist works regardless of how sparse or irregular they are.
    """
    try:
        files = [f for f in os.listdir(drive_dir) if f.startswith(prefix) and f.endswith(suffix)]
    except OSError:
        return

    def _num(fname):
        try:
            return int(fname[len(prefix):-len(suffix)])
        except ValueError:
            return -1

    files.sort(key=_num)
    excess = len(files) - keep_last_n
    for f in files[:max(0, excess)]:
        try:
            os.remove(os.path.join(drive_dir, f))
            print(f"Deleted old archive {f} to save space")
        except OSError as e:
            print(f"Could not delete {f}: {e}")


def run_pipeline(iterations=100, max_buffer_size=1000000, do_generate=True, do_train=True, do_evaluate=True,
                  concurrent_games=10, games_per_iteration=None, mcts_sims=50, eval_games=2, eval_sims=20,
                  batch_size=512, num_res_blocks=None, num_channels=None, max_consecutive_rejections=5,
                  min_force_promote_winrate=0.52, stall_eval_multiplier=3, buffer_archive_interval=5,
                  champion_archive_keep=25, buffer_archive_keep=3, random_baseline_interval=10,
                  random_baseline_games=20, random_baseline_sims=50):

    if games_per_iteration is None:
        games_per_iteration = concurrent_games

    if eval_games < 10:
        print(f"NOTE: --eval-games is {eval_games}. That's fine for a quick smoke test, but gating decisions "
              f"at this sample size are mostly noise (see the stall-guard notes below) -- use something like "
              f"30-100+ for a real training run.")
    if random_baseline_interval <= 1:
        print("NOTE: --random-baseline-interval is 1, so the vs-random sanity match runs every iteration at "
              "full cost even though it never affects promotion. Raise it (e.g. 10) once you've confirmed the "
              "network is comfortably beating random, to stop spending games/sims on a foregone conclusion.")

    net_kwargs = {}
    if num_res_blocks is not None:
        net_kwargs['num_res_blocks'] = num_res_blocks
    if num_channels is not None:
        net_kwargs['num_channels'] = num_channels

    # MULTI-GPU DETECTION (capability-gated, see get_usable_gpus())
    gpu_ids = get_usable_gpus()
    num_gpus = len(gpu_ids)
    device = torch.device(f"cuda:{gpu_ids[0]}" if num_gpus > 0 else "cpu")

    print(f"Starting ZeroCross Training Pipeline on {device} ({num_gpus} usable GPU(s): {gpu_ids})")

    if os.path.exists("/kaggle/working"):
        drive_dir = "/kaggle/working/models"
    elif os.path.exists("/content/drive"):
        drive_dir = "/content/drive/MyDrive/zerocross_models"
    else:
        drive_dir = "models"

    os.makedirs(drive_dir, exist_ok=True)
    model_path = os.path.join(drive_dir, "best_model.pth")
    last_candidate_path = os.path.join(drive_dir, "last_candidate.pth")
    csv_path = os.path.join(drive_dir, "training_log.csv")
    buffer_path = os.path.join(drive_dir, "replay_buffer.pt")
    state_path = os.path.join(drive_dir, "pipeline_state.json")

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "LR", "PI_Loss", "V_Loss", "Entropy",
                              "WinRate_vs_Random", "WinRate_vs_Champ", "Elo_Diff_vs_Champ",
                              "Promoted", "ForcedPromotion", "ConsecutiveRejections"])

    consecutive_rejections = 0
    start_iteration = 0
    if os.path.exists(state_path):
        with open(state_path) as f:
            state_data = json.load(f)
            consecutive_rejections = state_data.get("consecutive_rejections", 0)
            start_iteration = state_data.get("total_iterations", 0)

    best_net = ZeroCrossNet(**net_kwargs)
    optimizer_state = None

    if os.path.exists(model_path):
        checkpoint = safe_torch_load(model_path, map_location=device)
        if checkpoint is not None:
            print(f"Loading existing champion from {model_path}")
            state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint

            # STRIP 'module.' PREFIX IF LOADED FROM A PREVIOUS MULTI-GPU RUN
            state_dict = strip_module_prefix(state_dict)
            best_net.load_state_dict(state_dict)

            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer_state = checkpoint.get('optimizer_state_dict', None)
                if start_iteration == 0:
                    start_iteration = checkpoint.get('iteration', 0)
        else:
            print("Starting a fresh champion network (no valid checkpoint found).")

    best_net.to(device)
    replay_buffer = deque(maxlen=max_buffer_size)

    if os.path.exists(buffer_path):
        checkpoint = safe_torch_load(buffer_path)
        if checkpoint is not None:
            print(f"Loading historical replay buffer from {buffer_path}")
            if isinstance(checkpoint, dict) and 'data' in checkpoint:
                replay_buffer.extend(checkpoint['data'])
            else:
                replay_buffer.extend(checkpoint)
            print(f"Restored {len(replay_buffer)} historical samples to memory")
        else:
            print("Starting with an empty replay buffer (no valid buffer file found).")

    # Once consecutive_rejections has climbed to half of max_consecutive_rejections,
    # widen the evaluation for every subsequent iteration so the win-rate estimate
    # that decides "force or hold back" is much less noisy. See the gating block
    # near the bottom of the loop for how this is actually used.
    stall_boost_at = max(1, math.ceil(max_consecutive_rejections / 2))

    # The "vs Random Baseline" matchup inside evaluate.py's run_full_evaluation
    # is a sanity/trend metric only -- it never factors into the promotion
    # decision (only the matches against champion_nets do). Once the network
    # is competent it's a foregone conclusion every time, so re-measuring it
    # at full games/sims cost on every single iteration is pure waste. It's
    # instead checked at reduced cost every random_baseline_interval
    # iterations (always including the first iteration of this run, so a
    # fresh baseline is available immediately); other iterations just log
    # the last measured value. Set --random-baseline-interval 1 to go back
    # to checking it every iteration at full games_per_match/eval_sims.
    last_rand_wr = 0.0

    for i in range(start_iteration, start_iteration + iterations):
        iter_start_time = time.time()
        current_iter = i + 1
        print(f"\nALPHAZERO ITERATION {current_iter} of {start_iteration + iterations}")

        # LR SCHEDULE TUNED FOR LARGER BATCH SIZES (e.g. 2048)
        if current_iter <= 100:
            current_lr = 0.001
        elif current_iter <= 250:
            current_lr = 0.0005
        elif current_iter <= 700:
            current_lr = 0.0001
        else:
            current_lr = 0.00003

        print(f"Current Learning Rate: {current_lr}")

        metrics = {'pi_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0}
        rand_wr, champ_wr, elo_diff = 0.0, 0.0, 0.0
        promoted = False
        forced_promotion = False
        candidate_net = best_net
        opt_state = optimizer_state

        gen_duration = train_duration = eval_duration = aug_duration = avg_batch_size = 0.0

        if do_generate:
            print("\n[1/4] Generating Batched Self Play Data")
            gen_start = time.time()

            # --- FIX: put the champion in eval() mode before self-play ---
            # ZeroCrossNet uses BatchNorm throughout (stem/res-blocks/heads).
            # SelfPlayWorker calls net.forward() directly on raw MCTS-leaf
            # batches (it has to -- it needs pre-softmax logits, see
            # network.py's predict() docstring), and a freshly-constructed
            # nn.Module defaults to train() mode. Without this, self-play
            # was evaluating positions using per-batch BatchNorm statistics
            # (noisy, and dependent on whatever other leaves happened to be
            # in the same MCTS request batch) instead of the network's
            # learned running statistics -- while evaluate.py's gating
            # matches DOES correctly call .eval(). That mismatch meant the
            # policy/value the tree search actually searched with during
            # self-play was never quite the same function that got
            # evaluated and gated, which is a real source of instability
            # and wasted self-play games. Re-asserted every iteration so it
            # can't silently drift back to train() from anywhere else.
            best_net.eval()

            worker = SelfPlayWorker(best_net, num_concurrent_games=concurrent_games, mcts_simulations=mcts_sims, temperature_moves=45)
            new_samples = worker.generate_data(total_games_to_play=games_per_iteration)
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

            # --- Buffer archiving is throttled instead of every iteration ---
            # Each archive is a FULL copy of the current buffer (can be
            # hundreds of MB to ~1GB+ at max_buffer_size). Writing that to
            # disk on every single iteration, on top of the buffer_path
            # save above, doubles the I/O for no real benefit once the
            # buffer has filled up (buffer_path already holds the latest
            # state for resuming). Archiving every buffer_archive_interval
            # iterations keeps the rollback safety net while cutting that
            # redundant write volume dramatically -- worth watching on
            # Kaggle where /kaggle/working has a limited quota.
            if current_iter % buffer_archive_interval == 0:
                archive_path = os.path.join(drive_dir, f"replay_buffer_iter_{current_iter}.pt")
                torch.save(buffer_data, archive_path)
                print(f"Archived chunk saved to {archive_path}")
                prune_numbered_files(drive_dir, "replay_buffer_iter_", ".pt", buffer_archive_keep)

        if do_train:
            print(f"\n[2/4] Training Candidate Network (Batch Size: {batch_size})")
            if len(replay_buffer) == 0:
                print("Skipping training: Replay buffer is empty.")
                do_train = False
            else:
                train_start = time.time()
                candidate_net = ZeroCrossNet(**net_kwargs).to(device)
                candidate_net.load_state_dict(best_net.state_dict())

                # DATAPARALLEL ACROSS CAPABILITY-GATED GPUS, IF MORE THAN ONE
                if num_gpus > 1:
                    print(f"Utilizing {num_gpus} GPUs {gpu_ids} with DataParallel for Training")
                    candidate_net = torch.nn.DataParallel(candidate_net, device_ids=gpu_ids)

                candidate_net, opt_state, metrics = train_network(
                    candidate_net,
                    list(replay_buffer),
                    batch_size=batch_size,
                    epochs=2,
                    lr=current_lr,
                    device=device,
                    optimizer_state=optimizer_state
                )
                train_duration = time.time() - train_start

        # UNWRAP THE MODEL FOR CLEAN EVALUATION AND SAVING
        raw_candidate = candidate_net.module if hasattr(candidate_net, 'module') else candidate_net

        if do_evaluate:
            # STALL-ADAPTIVE EVALUATION SIZE:
            # A short losing streak against the gate is expected and healthy --
            # it means the gate is doing its job. But at small --eval-games,
            # a 90%-confidence lower bound (see evaluate.py's z=1.28) is easy
            # to fail on pure sampling noise even when the candidate is
            # genuinely a bit better. Rather than let noise be the reason we
            # eventually force a promotion, spend more games narrowing the
            # estimate once we're already halfway to the rejection limit, so
            # by the time --max-rejections is reached the "champ_wr" figure
            # used below is trustworthy rather than a 2-game coin flip.
            effective_eval_games = eval_games
            if consecutive_rejections >= stall_boost_at:
                effective_eval_games = eval_games * stall_eval_multiplier
                print(f"\n(Stalled for {consecutive_rejections} consecutive iteration(s) -- widening this "
                      f"evaluation to {effective_eval_games} games/match, x{stall_eval_multiplier}, to cut "
                      f"through noise before any forced-promotion decision.)")

            print(f"\n[3/4] Comprehensive Evaluation ({effective_eval_games} games/match)")
            eval_start = time.time()
            evaluator = Evaluator(device=device)

            run_random_check = (
                i == start_iteration
                or random_baseline_interval <= 1
                or current_iter % random_baseline_interval == 0
            )

            champion_nets = [best_net]
            history_files = [f for f in os.listdir(drive_dir) if f.startswith("champion_gen_") and f.endswith(".pth")]
            if history_files:
                selected_history = random.choice(history_files)
                hist_path = os.path.join(drive_dir, selected_history)
                checkpoint = safe_torch_load(hist_path, map_location=device)
                if checkpoint is not None:
                    print(f"Loading historical opponent: {selected_history}")
                    hist_net = ZeroCrossNet(**net_kwargs).to(device)
                    state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
                    state_dict = strip_module_prefix(state_dict)
                    hist_net.load_state_dict(state_dict)
                    hist_net.eval()
                    champion_nets.append(hist_net)

            # evaluate.py's play_match_batched() already calls .eval() on
            # both nets internally, so gating always compares two networks
            # in eval() mode -- this is the behavior self-play now matches.
            promoted, rand_wr, champ_wr, elo_diff = evaluator.run_full_evaluation(
                candidate_net=raw_candidate,
                champion_nets=champion_nets,
                sims=eval_sims,
                games_per_match=effective_eval_games,
                check_random_baseline=run_random_check,
                random_baseline_games=random_baseline_games,
                random_baseline_sims=random_baseline_sims,
                last_random_baseline_wr=last_rand_wr
            )
            last_rand_wr = rand_wr
            eval_duration = time.time() - eval_start

        if do_train:
            print("\n[4/4] Model Gating")
            torch.save({
                'iteration': current_iter,
                'model_state_dict': raw_candidate.state_dict(),
                'optimizer_state_dict': opt_state,
                'learning_rate': current_lr,
                'timestamp': time.time()
            }, last_candidate_path)

            if not do_evaluate:
                promoted = True
                print("Evaluation skipped. Force promoting candidate.")

            # STALL-GUARDED FORCED PROMOTION.
            #
            # The old version of this logic force-promoted unconditionally
            # once consecutive_rejections hit the limit. The problem: the
            # gate's whole job is to stop a worse network from ever becoming
            # the thing self-play plays against and future candidates train
            # from. An unconditional override throws that away exactly when
            # it fires -- if a run is stuck because the candidate is a real
            # regression (bad LR, a bug, an overfit batch) rather than noise,
            # "forced promotion" quietly ratchets the real skill level down
            # and lets it keep sliding, since the now-weaker champion also
            # generates the self-play data candidates keep training on.
            #
            # Two changes fix that without giving up the original goal
            # (never stall forever on eval noise):
            #  1. The evaluation itself gets wider as rejections mount (see
            #     effective_eval_games above), so "stuck at the limit" is a
            #     much more reliable signal by the time we get here.
            #  2. Forcing only fires if champ_wr clears a safety floor
            #     (--min-force-promote-winrate, default 0.45). That still
            #     lets through a candidate that's genuinely close/marginal
            #     (the "probably noise, not regression" case the original
            #     logic was meant for), but refuses to force through a
            #     candidate that's clearly losing outright.
            if not promoted:
                consecutive_rejections += 1
                if consecutive_rejections >= max_consecutive_rejections:
                    if champ_wr >= min_force_promote_winrate:
                        print(f"STALL BREAK: {consecutive_rejections} consecutive rejections, but win rate vs "
                              f"champion is {champ_wr:.2%} (>= {min_force_promote_winrate:.0%} floor) -- treating "
                              f"this as noise-limited gating rather than a real regression. Forcing promotion.")
                        promoted = True
                        forced_promotion = True
                    else:
                        print(f"HELD BACK: {consecutive_rejections} consecutive rejections and win rate vs "
                              f"champion is only {champ_wr:.2%} (< {min_force_promote_winrate:.0%} floor) -- this "
                              f"looks like a genuine regression, not noise, so NOT forcing promotion. If this "
                              f"keeps happening, check the LR schedule / loss curves rather than raising "
                              f"--max-rejections.")
                else:
                    print(f"REJECTED Candidate failed to clear the confidence bounds. ({consecutive_rejections}/{max_consecutive_rejections} rejections, win rate vs champion {champ_wr:.2%})")

            if promoted:
                tag = "FORCED " if forced_promotion else ""
                print(f"UPGRADE {tag}Candidate promoted! Saving to {model_path}")
                best_net.load_state_dict(raw_candidate.state_dict())
                optimizer_state = opt_state
                consecutive_rejections = 0

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
                prune_numbered_files(drive_dir, "champion_gen_", ".pth", champion_archive_keep)

        # Iteration bookkeeping is now saved every iteration regardless of
        # do_train, not just when do_train is True. Previously, running
        # --generate-only sessions (which don't touch do_train's block at
        # all) never persisted total_iterations, so csv rows written during
        # those runs could end up reusing "current_iter" numbers a later
        # full run would also use -- confusing logs and colliding archive
        # filenames. consecutive_rejections is left untouched here when
        # gating didn't run this iteration (do_train False), since it only
        # has meaning relative to promotion decisions.
        with open(state_path, "w") as f:
            json.dump({
                "consecutive_rejections": consecutive_rejections,
                "total_iterations": current_iter
            }, f)

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_iter,
                current_lr,
                metrics.get('pi_loss', 0.0),
                metrics.get('v_loss', 0.0),
                metrics.get('entropy', 0.0),
                round(rand_wr, 4),
                round(champ_wr, 4),
                round(elo_diff, 1),
                promoted,
                forced_promotion,
                consecutive_rejections
            ])
        print(f"Metrics successfully appended to {csv_path}")

        total_iter_duration = time.time() - iter_start_time

        print("\nITERATION BENCHMARK REPORT")
        if do_generate:
            print(f"Average MCTS Batch Size: {avg_batch_size:.2f}")
            print(f"Data Generation Time:    {gen_duration:.2f} sec")
            print(f"Augmentation Time:       {aug_duration:.2f} sec")
        if do_train:
            print(f"Network Training Time:   {train_duration:.2f} sec")
        if do_evaluate:
            print(f"Evaluation Time:         {eval_duration:.2f} sec")
        print(f"Total Iteration Time:    {total_iter_duration:.2f} sec")

        # SAVE A PERMANENT MILESTONE EVERY 50 ITERATIONS
        if current_iter % 50 == 0:
            milestone_path = os.path.join(drive_dir, f"milestone_iter_{current_iter}.pth")
            shutil.copy(model_path, milestone_path)
            print(f"Permanent milestone saved: {milestone_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroCross Training Pipeline")
    parser.add_argument("--iterations", type=int, default=1, help="Number of pipeline iterations")
    parser.add_argument("--generate-only", action="store_true", help="Only generate self-play data and update replay buffer")
    parser.add_argument("--train-only", action="store_true", help="Only train the network on existing buffer and force promote")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate the current best model")

    parser.add_argument("--concurrent-games", type=int, default=10, help="Parallel games in-flight during self-play (bounded by GPU memory)")
    parser.add_argument("--games-per-iteration", type=int, default=None, help="Total self-play games generated per iteration before training (default: same as --concurrent-games)")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move during self-play")
    parser.add_argument("--eval-games", type=int, default=2, help="Games per matchup in evaluation (e.g. 40 on Kaggle)")
    parser.add_argument("--eval-sims", type=int, default=20, help="MCTS simulations per move during evaluation")

    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")

    parser.add_argument("--num-res-blocks", type=int, default=None, help="Override ZeroCrossNet residual block count (default: network.py's default)")
    parser.add_argument("--num-channels", type=int, default=None, help="Override ZeroCrossNet channel width (default: network.py's default)")
    parser.add_argument("--max-rejections", type=int, default=5, help="After this many consecutive gating rejections in a row, consider a stall-guarded forced promotion (see --min-force-promote-winrate) instead of stalling forever on eval noise")
    parser.add_argument("--min-force-promote-winrate", type=float, default=0.52, help="Safety floor for forced promotion: only force through a candidate stuck at --max-rejections if its win rate vs the champion is at least this high. Below this, it's treated as a genuine regression and NOT promoted, no matter how many rejections have piled up")
    parser.add_argument("--stall-eval-multiplier", type=int, default=3, help="Once consecutive rejections reach half of --max-rejections, multiply --eval-games by this factor for subsequent evaluations, to shrink confidence-interval noise before a forced-promotion decision is made")
    parser.add_argument("--max-buffer-size", type=int, default=1000000, help="Max samples kept in the replay buffer (deque maxlen); oldest samples are dropped first")
    parser.add_argument("--buffer-archive-interval", type=int, default=5, help="Write a full numbered replay-buffer archive every N iterations, instead of every iteration, to cut redundant disk I/O")
    parser.add_argument("--buffer-archive-keep", type=int, default=3, help="How many numbered replay-buffer archives to retain on disk (oldest deleted first)")
    parser.add_argument("--champion-archive-keep", type=int, default=25, help="How many historical champion_gen_*.pth checkpoints to retain on disk (oldest deleted first)")

    parser.add_argument("--random-baseline-interval", type=int, default=10, help="Only re-measure win rate vs a random-move baseline every N iterations (it never affects promotion, it's a sanity/trend metric, so it doesn't need full precision every iteration). Set to 1 to check every iteration like before")
    parser.add_argument("--random-baseline-games", type=int, default=20, help="Games used for the vs-random sanity check when it does run -- can be much smaller than --eval-games since beating random is a low bar")
    parser.add_argument("--random-baseline-sims", type=int, default=50, help="MCTS sims used for the vs-random sanity check when it does run -- can be much smaller than --eval-sims for the same reason")

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
        max_buffer_size=args.max_buffer_size,
        do_generate=do_generate,
        do_train=do_train,
        do_evaluate=do_evaluate,
        concurrent_games=args.concurrent_games,
        games_per_iteration=args.games_per_iteration,
        mcts_sims=args.mcts_sims,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        batch_size=args.batch_size,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        max_consecutive_rejections=args.max_rejections,
        min_force_promote_winrate=args.min_force_promote_winrate,
        stall_eval_multiplier=args.stall_eval_multiplier,
        buffer_archive_interval=args.buffer_archive_interval,
        buffer_archive_keep=args.buffer_archive_keep,
        champion_archive_keep=args.champion_archive_keep,
        random_baseline_interval=args.random_baseline_interval,
        random_baseline_games=args.random_baseline_games,
        random_baseline_sims=args.random_baseline_sims,
    )