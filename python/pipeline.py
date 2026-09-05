import os
import csv
import json
import math
import time
import sys
import argparse
import random
import shutil
import tempfile

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])

import torch
import torch.multiprocessing as torch_mp
from collections import deque
from network import ZeroCrossNet
from self_play import SelfPlayWorker
from train import train_network
from evaluate import Evaluator, run_matches_across_gpus


def _selfplay_worker_process(gpu_id, state_dict_cpu, net_kwargs, num_games, concurrent_games,
                              mcts_sims, temp_moves, out_path, status_queue):
    """
    Runs one GPU's share of self-play in its own process.

    CUDA needs its own context per device for this workload -- a single
    process can't drive two GPUs in true parallel the way DataParallel does
    for the training step, so each GPU gets a subprocess with its own copy
    of the champion's weights. Results are handed back via a temp file
    (out_path), not the multiprocessing Queue itself: a plain Queue's pipe
    can deadlock on payloads this large (hundreds of MB of self-play
    samples) if the parent doesn't drain it fast enough. The queue is only
    used for the small completion signal + timing stats.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        net = ZeroCrossNet(**net_kwargs).to(device)
        net.load_state_dict(state_dict_cpu)
        net.eval()

        worker = SelfPlayWorker(net, num_concurrent_games=concurrent_games,
                                 mcts_simulations=mcts_sims, temperature_moves=temp_moves)
        samples = worker.generate_data(total_games_to_play=num_games)
        torch.save(samples, out_path)

        status_queue.put({
            'ok': True,
            'aug_time': worker.total_augmentation_time,
            'avg_batch_size': worker.avg_batch_size,
            'num_samples': len(samples),
        })
    except Exception as e:
        status_queue.put({'ok': False, 'error': repr(e)})


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


def ensure_csv_header(csv_path, expected_header):
    """
    Make sure csv_path's header matches expected_header before anything
    appends to it this run. An older training_log.csv (e.g. one carried
    forward from before min_champ_lcb / milestone tracking existed) has
    fewer columns -- appending new-schema rows straight onto that file
    would silently misalign every column from that point on rather than
    raising an error, which is worse than either starting clean or erroring
    loudly. Instead: if the file doesn't exist, create it with the new
    header (unchanged behavior). If it exists with a DIFFERENT header, the
    old file is preserved untouched under a .pre_migration-<timestamp>
    suffix and a fresh file with the new header is started -- so nothing is
    lost, but every row from here on is unambiguously the new schema.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            csv.writer(f).writerow(expected_header)
        return

    with open(csv_path, newline='') as f:
        existing_header = next(csv.reader(f), [])

    if existing_header == expected_header:
        return

    backup_path = f"{csv_path}.pre_migration-{int(time.time())}"
    os.rename(csv_path, backup_path)
    print(f"NOTE: {csv_path} was on an older schema (header didn't match). "
          f"Preserved the old file as {backup_path} and started a fresh log "
          f"with the current header.")
    with open(csv_path, mode='w', newline='') as f:
        csv.writer(f).writerow(expected_header)


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


def compute_learning_rate(iteration, lr_reheat_interval=40, lr_reheat_duration=5, lr_reheat_multiplier=3.0):
    """
    Base schedule is the original one-way step-decay (unchanged): LR only
    ever goes DOWN as iteration count rises. That works fine while training
    is reliably improving, but it also means once training plateaus there's
    no mechanism to ever perturb the optimizer out of a local minimum --
    it just sits at whatever the current step's LR is, indefinitely, for
    however many more iterations you throw at it.

    lr_reheat_* layers a lightweight SGDR-style periodic warm restart on top
    of that base schedule: every `lr_reheat_interval` iterations (once past
    the initial iteration-100 ramp-up phase), LR is temporarily multiplied
    by `lr_reheat_multiplier` for the next `lr_reheat_duration` iterations,
    then drops straight back to the normal base-schedule value. This is a
    PURE function of (iteration, these three knobs) -- no extra state to
    track or persist across restarts -- so it's safe to preview from a
    resume/state-check cell and safe across however many separate Kaggle
    sessions this ends up spanning.

    Set lr_reheat_interval=0 (or None) to disable entirely and fall back to
    the plain step-decay schedule.

    Returns (lr, is_reheat_iteration).
    """
    if iteration <= 100:
        base_lr = 0.001
    elif iteration <= 250:
        base_lr = 0.0005
    elif iteration <= 700:
        base_lr = 0.0001
    else:
        base_lr = 0.00003

    is_reheat = False
    if lr_reheat_interval and lr_reheat_interval > 0 and iteration > 100:
        cycle_position = (iteration - 101) % lr_reheat_interval
        if cycle_position < lr_reheat_duration:
            is_reheat = True

    if is_reheat:
        return base_lr * lr_reheat_multiplier, True
    return base_lr, False


def run_pipeline(iterations=100, max_buffer_size=1000000, do_generate=True, do_train=True, do_evaluate=True,
                  concurrent_games=10, games_per_iteration=None, mcts_sims=50, eval_games=2, eval_sims=20,
                  batch_size=512, num_res_blocks=None, num_channels=None, max_consecutive_rejections=5,
                  min_force_promote_lcb=0.50, stall_eval_multiplier=3, buffer_archive_interval=5,
                  champion_archive_keep=25, buffer_archive_keep=3, random_baseline_interval=10,
                  random_baseline_games=20, random_baseline_sims=50, sentinel_checkpoint=None,
                  milestone_interval=25, milestone_games=100, milestone_sims=200, milestone_max_refs=5,
                  temp_moves=35, lr_reheat_interval=40, lr_reheat_duration=5, lr_reheat_multiplier=3.0):

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
    milestone_csv_path = os.path.join(drive_dir, "milestone_log.csv")
    buffer_path = os.path.join(drive_dir, "replay_buffer.pt")
    state_path = os.path.join(drive_dir, "pipeline_state.json")

    CSV_HEADER = ["Iteration", "LR", "PI_Loss", "V_Loss", "Entropy",
                  "WinRate_vs_Random", "WinRate_vs_Champ", "Elo_Diff_vs_Champ",
                  "MinChampLCB", "Promoted", "ForcedPromotion", "ConsecutiveRejections"]
    ensure_csv_header(csv_path, CSV_HEADER)

    MILESTONE_CSV_HEADER = ["Iteration", "RefName", "WinRate", "LCB", "EloDiff", "Wins", "Losses", "Draws"]
    ensure_csv_header(milestone_csv_path, MILESTONE_CSV_HEADER)

    consecutive_rejections = 0
    start_iteration = 0
    if os.path.exists(state_path):
        with open(state_path) as f:
            state_data = json.load(f)
            consecutive_rejections = state_data.get("consecutive_rejections", 0)
            start_iteration = state_data.get("total_iterations", 0)

    # Tracks the iteration number embedded in whatever best_net CURRENTLY is --
    # i.e. the last iteration that actually got promoted (or the resume point,
    # if nothing has been promoted yet this run). Used below to detect when
    # the champion and the pinned sentinel are literally the same weights, so
    # the milestone gauntlet doesn't run a network against a mirror of itself
    # and misread the resulting ~50% coin-flip as "drift".
    champion_iteration = start_iteration

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

    # FIXED SENTINEL OPPONENT (loaded once, unaffected by champion_gen_ pruning).
    #
    # The rotating "historical opponent" picked below (see history_files) is
    # always drawn from whatever champion_gen_*.pth files currently exist,
    # capped at the most recent --champion-archive-keep promotions. That
    # pool can drift downward as a whole over many iterations without ever
    # showing up as a rejection, because every comparison is always against
    # a similarly-drifted recent peer -- there's no permanent, independently
    # verified reference point in the loop. sentinel_net is that reference
    # point: loaded once here, never pruned, never replaced automatically.
    # It should only ever be swapped out deliberately, after a properly
    # powered check (e.g. a round-robin) confirms a new checkpoint is
    # genuinely and robustly stronger -- not by any automated process.
    sentinel_net = None
    sentinel_iteration = None
    if sentinel_checkpoint:
        if os.path.exists(sentinel_checkpoint):
            checkpoint = safe_torch_load(sentinel_checkpoint, map_location=device)
            if checkpoint is not None:
                sentinel_net = ZeroCrossNet(**net_kwargs).to(device)
                state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
                state_dict = strip_module_prefix(state_dict)
                sentinel_net.load_state_dict(state_dict)
                sentinel_net.eval()
                sentinel_iteration = checkpoint.get('iteration') if isinstance(checkpoint, dict) else None
                print(f"Loaded fixed sentinel from {sentinel_checkpoint} (iteration {sentinel_iteration}) -- "
                      f"included in every evaluation, immune to champion_gen_ pruning and the rotating-pool blind spot.")
            else:
                print(f"WARNING: --sentinel-checkpoint {sentinel_checkpoint} failed to load; continuing without one.")
        else:
            print(f"WARNING: --sentinel-checkpoint {sentinel_checkpoint} not found; continuing without one.")

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
        # LR SCHEDULE TUNED FOR LARGER BATCH SIZES (e.g. 2048), with an
        # optional periodic warm restart layered on top -- see
        # compute_learning_rate() for why.
        current_lr, lr_is_reheat = compute_learning_rate(
            current_iter, lr_reheat_interval, lr_reheat_duration, lr_reheat_multiplier
        )

        if lr_is_reheat:
            print(f"Current Learning Rate: {current_lr}  <-- LR REHEAT active this iteration "
                  f"(x{lr_reheat_multiplier:g} boost, every {lr_reheat_interval} iters for "
                  f"{lr_reheat_duration} iter(s))")
        else:
            print(f"Current Learning Rate: {current_lr}")

        metrics = {'pi_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0}
        rand_wr, champ_wr, elo_diff, min_champ_wr, min_champ_lcb = 0.0, 0.0, 0.0, 0.0, 0.0
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

            if num_gpus > 1:
                # MULTI-GPU SELF-PLAY.
                #
                # Previously this phase only ever ran on gpu_ids[0]: best_net
                # lives on `device` (cuda:{gpu_ids[0]}) and a single
                # SelfPlayWorker was handed that one net, so every other GPU
                # sat idle for the entire self-play phase -- the single
                # biggest time cost of every iteration (only the training
                # step used DataParallel across both GPUs). This splits
                # games_per_iteration across all usable GPUs: each gets its
                # own subprocess with its own copy of the current champion's
                # weights on its own device, plays its share of the games,
                # and the resulting samples are merged back below.
                print(f"Splitting self-play across {num_gpus} GPUs {gpu_ids} "
                      f"({games_per_iteration} games total)")

                n_workers = num_gpus
                base_games = games_per_iteration // n_workers
                games_split = [base_games] * n_workers
                games_split[0] += games_per_iteration - base_games * n_workers  # remainder to GPU0

                # Concurrent-games is a per-GPU memory budget, so it's split
                # too -- running the full --concurrent-games on every GPU
                # simultaneously would multiply the memory footprint by
                # num_gpus, not just parallelize compute.
                base_concurrent = max(1, concurrent_games // n_workers)
                concurrent_split = [base_concurrent] * n_workers

                state_dict_cpu = {k: v.cpu() for k, v in best_net.state_dict().items()}
                tmp_dir = tempfile.mkdtemp(prefix="selfplay_")
                ctx = torch_mp.get_context('spawn')

                procs, out_paths, queues = [], [], []
                for w_idx, gpu_id in enumerate(gpu_ids):
                    out_path = os.path.join(tmp_dir, f"samples_gpu{gpu_id}.pt")
                    q = ctx.Queue()
                    p = ctx.Process(
                        target=_selfplay_worker_process,
                        args=(gpu_id, state_dict_cpu, net_kwargs, games_split[w_idx],
                              concurrent_split[w_idx], mcts_sims, temp_moves, out_path, q)
                    )
                    p.start()
                    procs.append(p)
                    out_paths.append(out_path)
                    queues.append(q)

                new_samples = []
                aug_times, batch_sizes = [], []
                for w_idx, (p, out_path, q, gpu_id) in enumerate(zip(procs, out_paths, queues, gpu_ids)):
                    status = q.get()   # blocks until that GPU's worker signals done
                    p.join()
                    if not status.get('ok'):
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        raise RuntimeError(
                            f"Self-play worker on GPU {gpu_id} failed: {status.get('error')}"
                        )
                    new_samples.extend(torch.load(out_path, weights_only=False))
                    aug_times.append(status['aug_time'])
                    batch_sizes.append(status['avg_batch_size'])
                    print(f"  GPU {gpu_id}: {status['num_samples']} samples "
                          f"from {games_split[w_idx]} games")

                shutil.rmtree(tmp_dir, ignore_errors=True)
                aug_duration = sum(aug_times)
                avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
            else:
                worker = SelfPlayWorker(best_net, num_concurrent_games=concurrent_games, mcts_simulations=mcts_sims, temperature_moves=temp_moves)
                new_samples = worker.generate_data(total_games_to_play=games_per_iteration)
                aug_duration = worker.total_augmentation_time
                avg_batch_size = worker.avg_batch_size

            gen_duration = time.time() - gen_start

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

                sample_size = min(len(replay_buffer), 600000)
                train_sample = random.sample(list(replay_buffer), sample_size)
                
                candidate_net, opt_state, metrics = train_network(
                candidate_net,
                train_sample,
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
            champion_names = [f"Latest Champion (iter {champion_iteration})"]
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
                    champion_names.append(f"Historical Champion ({selected_history.replace('.pth', '')})")

            if sentinel_net is not None:
                champion_nets.append(sentinel_net)
                champion_names.append(f"Sentinel (iter {sentinel_iteration})" if sentinel_iteration is not None else "Sentinel")

            # evaluate.py's play_match_batched() already calls .eval() on
            # both nets internally, so gating always compares two networks
            # in eval() mode -- this is the behavior self-play now matches.
            #
            # gpu_ids/net_kwargs let run_full_evaluation spread every matchup
            # (random baseline, latest champion, historical, sentinel) across
            # all usable GPUs instead of running them back to back on just
            # gpu_ids[0] -- previously the entire evaluation phase used one
            # GPU while any others sat idle, same class of fix as the
            # multi-GPU self-play split above. Single/no-GPU runs are
            # unaffected: run_full_evaluation falls back to the original
            # sequential behavior whenever fewer than 2 GPUs are passed.
            promoted, rand_wr, champ_wr, elo_diff, min_champ_wr, min_champ_lcb = evaluator.run_full_evaluation(
                candidate_net=raw_candidate,
                champion_nets=champion_nets,
                champion_names=champion_names,
                sims=eval_sims,
                games_per_match=effective_eval_games,
                check_random_baseline=run_random_check,
                random_baseline_games=random_baseline_games,
                random_baseline_sims=random_baseline_sims,
                last_random_baseline_wr=last_rand_wr,
                gpu_ids=gpu_ids,
                net_kwargs=net_kwargs,
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
            #  2. Forcing only fires if min_champ_lcb -- the WORST
            #     lower-confidence-bound across every opponent faced this
            #     iteration (latest champion, rotating historical pick, AND
            #     the fixed sentinel if one is configured) -- clears
            #     --min-force-promote-lcb (default 0.50, the SAME bar the
            #     normal per-opponent gate uses just above). Gating on raw
            #     win rate here used to let a candidate through on a number
            #     like "52% over a handful of games" that carried almost no
            #     statistical weight -- a coin flip dressed up as a signal.
            #     Using min_champ_lcb against the identical 0.50 bar means
            #     the escape hatch can never approve something the normal
            #     gate would still be unsure about; the only difference is
            #     effective_eval_games has grown (via stall_eval_multiplier)
            #     enough to make that same bar decisive instead of noisy.
            #     Every opponent has to clear it, not just the most recent
            #     champion -- that's what catches a candidate that beats a
            #     possibly-already-drifted latest champion while still
            #     confidently losing to the sentinel.
            if not promoted:
                consecutive_rejections += 1
                if consecutive_rejections >= max_consecutive_rejections:
                    if min_champ_lcb >= min_force_promote_lcb:
                        print(f"STALL BREAK: {consecutive_rejections} consecutive rejections, but the worst LCB "
                              f"across every opponent faced (including the sentinel) is {min_champ_lcb:.2%} "
                              f"(>= {min_force_promote_lcb:.0%} floor, {effective_eval_games} games/match) -- "
                              f"at this sample size that's a genuine statistical edge, not noise. Forcing promotion.")
                        promoted = True
                        forced_promotion = True
                    else:
                        print(f"HELD BACK: {consecutive_rejections} consecutive rejections. Worst LCB across all "
                              f"opponents faced is only {min_champ_lcb:.2%} (raw win rate {min_champ_wr:.2%}, vs "
                              f"latest champion: {champ_wr:.2%}) (< {min_force_promote_lcb:.0%} floor) -- even at "
                              f"{effective_eval_games} games/match this isn't distinguishable from a real "
                              f"regression, so NOT forcing promotion. If this keeps happening, check the LR "
                              f"schedule / self-play volume rather than lowering --min-force-promote-lcb.")
                else:
                    print(f"REJECTED Candidate failed to clear the confidence bounds. ({consecutive_rejections}/{max_consecutive_rejections} "
                          f"rejections, win rate vs latest champion {champ_wr:.2%}, worst LCB vs any opponent {min_champ_lcb:.2%})")

            if promoted:
                tag = "FORCED " if forced_promotion else ""
                print(f"UPGRADE {tag}Candidate promoted! Saving to {model_path}")
                best_net.load_state_dict(raw_candidate.state_dict())
                optimizer_state = opt_state
                consecutive_rejections = 0
                champion_iteration = current_iter

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
                round(min_champ_lcb, 4) if do_evaluate else "",
                promoted,
                forced_promotion,
                consecutive_rejections
            ])
        print(f"Metrics successfully appended to {csv_path}")

        # PERIODIC MILESTONE GAUNTLET.
        #
        # The rotating champion_gen_ pool used for ordinary gating (above) is
        # capped to the most recent --champion-archive-keep promotions, so it
        # can only ever measure the champion against relatively recent peers.
        # If quality drifts down gradually, every one of those peers has
        # drifted by a similar amount, and the rotating comparison alone
        # never flags it. This block runs a separate, higher-precision check
        # every --milestone-interval iterations: the CURRENT CHAMPION (not
        # the just-trained candidate) against the pinned sentinel plus a
        # bounded set of historical milestone checkpoints, logged to its own
        # milestone_log.csv. --milestone-max-refs caps how many milestones
        # are checked each time so cost stays bounded over a long run instead
        # of growing without limit as milestones accumulate.
        if do_evaluate and current_iter % milestone_interval == 0:
            print(f"\nMILESTONE GAUNTLET (iteration {current_iter}, every {milestone_interval})")
            refs = {}
            skip_sentinel_reason = None
            if sentinel_net is not None:
                if sentinel_iteration is not None and sentinel_iteration == champion_iteration:
                    # The champion hasn't been promoted since the sentinel was pinned, so
                    # best_net and sentinel_net are literally the same weights right now.
                    # Playing a network against a mirror of itself just measures move-order/
                    # temperature noise and reads as a coin-flip ~50% -- that's NOT drift,
                    # it's a mathematical certainty for identical weights, so skip the match
                    # entirely rather than logging a misleading "possible drift" warning.
                    skip_sentinel_reason = (f"sentinel is iteration {sentinel_iteration}, same as the current "
                                             f"champion -- they're identical weights right now, so this comparison "
                                             f"would just measure noise. Skipping until the champion is actually "
                                             f"promoted past the sentinel.")
                else:
                    refs["sentinel"] = sentinel_net

            milestone_files = sorted(
                (f for f in os.listdir(drive_dir) if f.startswith("milestone_iter_") and f.endswith(".pth")),
                key=lambda f: int(f[len("milestone_iter_"):-len(".pth")]) if f[len("milestone_iter_"):-len(".pth")].isdigit() else -1
            )
            # Most recent milestones are the most informative about *recent*
            # drift; oldest ones (already covered by the sentinel, usually)
            # are dropped first once milestone_max_refs is exceeded.
            for f in milestone_files[-milestone_max_refs:]:
                milestone_iter_num = int(f[len("milestone_iter_"):-len(".pth")]) if f[len("milestone_iter_"):-len(".pth")].isdigit() else None
                if milestone_iter_num is not None and milestone_iter_num == champion_iteration:
                    # Same reasoning as the sentinel check above -- a milestone saved at
                    # exactly the champion's current iteration is the champion.
                    continue
                path = os.path.join(drive_dir, f)
                checkpoint = safe_torch_load(path, map_location=device)
                if checkpoint is None:
                    continue
                ref_net = ZeroCrossNet(**net_kwargs).to(device)
                state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
                ref_net.load_state_dict(strip_module_prefix(state_dict))
                ref_net.eval()
                refs[f.replace(".pth", "")] = ref_net

            if skip_sentinel_reason:
                print(f"  (skipping sentinel comparison: {skip_sentinel_reason})")

            if not refs:
                print("  (no other sentinel/milestone checkpoints available yet -- skipping)")
            else:
                # Same fix as the routine evaluation above: each reference
                # match is independent, so with gpu_ids spread across more
                # than one GPU they run concurrently instead of one at a
                # time on gpu_ids[0]. Falls back to the original sequential
                # behavior on single/no-GPU runs.
                # `name` (the refs dict key) is what gets written to
                # milestone_log.csv's RefName column, so it's kept exactly
                # as before ("sentinel", "milestone_iter_N") -- plot_metrics.py
                # groups by that exact string, and changing it mid-run would
                # split an existing trend line into two legend entries. The
                # bracketed iteration is added only to the display label used
                # for the console print / results lookup below.
                ref_names = list(refs.keys())
                display_names = {
                    name: (f"{name} (iter {sentinel_iteration})" if name == "sentinel" and sentinel_iteration is not None else name)
                    for name in ref_names
                }
                matches = [(f"Champion vs {display_names[name]}", refs[name], milestone_games, milestone_sims) for name in ref_names]
                results = run_matches_across_gpus(best_net, matches, gpu_ids, net_kwargs)

                with open(milestone_csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    for name, (label, _, _, _) in zip(ref_names, matches):
                        r = results[label]
                        writer.writerow([current_iter, name, round(r['wr'], 4), round(r['lcb'], 4), round(r['elo'], 1), r['w'], r['l'], r['d']])
                        if r['lcb'] <= 0.50:
                            print(f"  WARNING: current champion's LCB against {name} is at or below 50% -- "
                                  f"possible drift. Worth a manual look (see arena.py) before trusting further "
                                  f"promotions built on top of this champion.")
                print(f"Milestone results appended to {milestone_csv_path}")

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
    parser.add_argument("--temp-moves", type=int, default=20, help="Number of plies at the START of each self-play game sampled stochastically (proportional to MCTS visit counts) before switching to greedy (temperature=0) play. Higher values inject more exploration/diversity into training data but also more near-random outcomes that dilute the training signal; lower values produce cleaner but less diverse games. Was hardcoded to 45 previously -- if candidates are stuck reading as statistical noise around 50% vs the champion for many iterations, try lowering this before raising sims/games further")
    parser.add_argument("--lr-reheat-interval", type=int, default=40, help="Every N iterations (once past the initial iteration-100 ramp-up), briefly multiply LR by --lr-reheat-multiplier for --lr-reheat-duration iterations, then drop back to the normal step-decay schedule. This is a periodic SGDR-style warm restart: it exists so a long stall isn't just the optimizer sitting motionless in a local minimum with no mechanism to ever get perturbed out of it. Set to 0 to disable and use the plain one-way step-decay schedule")
    parser.add_argument("--lr-reheat-duration", type=int, default=5, help="How many consecutive iterations each LR reheat stays boosted before dropping back to the base schedule (see --lr-reheat-interval)")
    parser.add_argument("--lr-reheat-multiplier", type=float, default=3.0, help="Multiplier applied to the base scheduled LR during a reheat window (see --lr-reheat-interval)")
    parser.add_argument("--eval-games", type=int, default=2, help="Games per matchup in evaluation (e.g. 40 on Kaggle)")
    parser.add_argument("--eval-sims", type=int, default=20, help="MCTS simulations per move during evaluation")

    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")

    parser.add_argument("--num-res-blocks", type=int, default=None, help="Override ZeroCrossNet residual block count (default: network.py's default)")
    parser.add_argument("--num-channels", type=int, default=None, help="Override ZeroCrossNet channel width (default: network.py's default)")
    parser.add_argument("--max-rejections", type=int, default=5, help="After this many consecutive gating rejections in a row, consider a stall-guarded forced promotion (see --min-force-promote-lcb) instead of stalling forever on eval noise")
    parser.add_argument("--min-force-promote-lcb", type=float, default=0.50, help="Safety floor for forced promotion, expressed as a lower-confidence-bound (LCB) -- the SAME statistical bar the normal per-opponent gate uses. Only force through a candidate stuck at --max-rejections if the worst LCB across every opponent it faced (at the widened --stall-eval-multiplier sample size) is at least this high. Below this, it's treated as a genuine regression and NOT promoted, no matter how many rejections have piled up. Do not set this below 0.50 -- that reintroduces the exact loophole this replaces raw win-rate gating for")
    parser.add_argument("--stall-eval-multiplier", type=int, default=3, help="Once consecutive rejections reach half of --max-rejections, multiply --eval-games by this factor for subsequent evaluations, to shrink confidence-interval noise before a forced-promotion decision is made")
    parser.add_argument("--max-buffer-size", type=int, default=1000000, help="Max samples kept in the replay buffer (deque maxlen); oldest samples are dropped first")
    parser.add_argument("--buffer-archive-interval", type=int, default=5, help="Write a full numbered replay-buffer archive every N iterations, instead of every iteration, to cut redundant disk I/O")
    parser.add_argument("--buffer-archive-keep", type=int, default=3, help="How many numbered replay-buffer archives to retain on disk (oldest deleted first)")
    parser.add_argument("--champion-archive-keep", type=int, default=25, help="How many historical champion_gen_*.pth checkpoints to retain on disk (oldest deleted first)")
    parser.add_argument("--sentinel-checkpoint", type=str, default=None, help="Path to a fixed checkpoint always included as an extra evaluation opponent, in addition to the current champion and the rotating champion_gen_ pool. Unlike that pool (capped to the most recent --champion-archive-keep promotions), this one never changes and is never pruned -- it's what catches collective drift across many iterations that only-recent-history comparisons can't. Update it manually only after a properly powered check (e.g. a round-robin) confirms a new checkpoint is robustly stronger.")

    parser.add_argument("--random-baseline-interval", type=int, default=10, help="Only re-measure win rate vs a random-move baseline every N iterations (it never affects promotion, it's a sanity/trend metric, so it doesn't need full precision every iteration). Set to 1 to check every iteration like before")
    parser.add_argument("--random-baseline-games", type=int, default=20, help="Games used for the vs-random sanity check when it does run -- can be much smaller than --eval-games since beating random is a low bar")
    parser.add_argument("--random-baseline-sims", type=int, default=50, help="MCTS sims used for the vs-random sanity check when it does run -- can be much smaller than --eval-sims for the same reason")

    parser.add_argument("--milestone-interval", type=int, default=25, help="Every N iterations, run a separate high-precision gauntlet: current CHAMPION (not the candidate) vs the pinned sentinel and a bounded set of recent milestone checkpoints, logged to milestone_log.csv. This is an independent drift check -- it never affects promotion, only visibility")
    parser.add_argument("--milestone-games", type=int, default=100, help="Games per matchup in the milestone gauntlet -- can and should be higher-precision than routine --eval-games since it only runs every --milestone-interval iterations")
    parser.add_argument("--milestone-sims", type=int, default=200, help="MCTS sims per move in the milestone gauntlet")
    parser.add_argument("--milestone-max-refs", type=int, default=5, help="Cap on how many past milestone_iter_*.pth checkpoints are checked each gauntlet run (most recent ones kept), so cost stays bounded as milestones accumulate over a long run. The sentinel, if configured, is always included on top of this cap")

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
        temp_moves=args.temp_moves,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        batch_size=args.batch_size,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels,
        max_consecutive_rejections=args.max_rejections,
        min_force_promote_lcb=args.min_force_promote_lcb,
        stall_eval_multiplier=args.stall_eval_multiplier,
        buffer_archive_interval=args.buffer_archive_interval,
        buffer_archive_keep=args.buffer_archive_keep,
        champion_archive_keep=args.champion_archive_keep,
        random_baseline_interval=args.random_baseline_interval,
        random_baseline_games=args.random_baseline_games,
        random_baseline_sims=args.random_baseline_sims,
        sentinel_checkpoint=args.sentinel_checkpoint,
        milestone_interval=args.milestone_interval,
        milestone_games=args.milestone_games,
        milestone_sims=args.milestone_sims,
        milestone_max_refs=args.milestone_max_refs,
        lr_reheat_interval=args.lr_reheat_interval,
        lr_reheat_duration=args.lr_reheat_duration,
        lr_reheat_multiplier=args.lr_reheat_multiplier,
    )