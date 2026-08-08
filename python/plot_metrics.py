import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_metrics(csv_path=None, save_dir=None):
    base_dir = "/kaggle/working/models" if os.path.exists("/kaggle/working") else "models"

    if csv_path is None:
        csv_path = os.path.join(base_dir, "training_log.csv")
    if save_dir is None:
        save_dir = os.path.join(base_dir, "plots")

    if not os.path.exists(csv_path):
        print(f"Log file not found at {csv_path}. Run the pipeline first.")
        return

    # Read the CSV
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.")
        return

    os.makedirs(save_dir, exist_ok=True)
    iterations = df["Iteration"]

    # Older logs (from before forced-promotion tracking was added) won't have
    # these columns -- treat them as "unknown / not forced" rather than
    # erroring, so this still works on a training_log.csv from an older run.
    has_forced_col = "ForcedPromotion" in df.columns
    forced_iters = iterations[df["ForcedPromotion"]] if has_forced_col else iterations[[]]

    # 1. Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, df["PI_Loss"], label="Policy Loss", marker='o')
    plt.plot(iterations, df["V_Loss"], label="Value Loss", marker='o')
    plt.title("Network Losses over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "losses.png"))
    plt.close()

    # 2. Plot Entropy
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, df["Entropy"], label="Policy Entropy", color="purple", marker='o')
    plt.title("Policy Entropy (Exploration) over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "entropy.png"))
    plt.close()

    # 3. Plot Elo & Win Rates, with forced promotions flagged
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Elo Diff vs Champion', color=color)
    ax1.plot(iterations, df["Elo_Diff_vs_Champ"], color=color, marker='o', label="Elo Diff")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Win Rate vs Random', color=color)
    ax2.plot(iterations, df["WinRate_vs_Random"], color=color, marker='x', linestyle='--', label="WR vs Random")
    ax2.tick_params(axis='y', labelcolor=color)

    # Forced promotions are the iterations most worth double-checking --
    # each one means the candidate only got through because it hit the
    # rejection-streak safety valve, not because it cleanly won the gate.
    # A red line here and there among a generally-rising trend is normal.
    # Several clustered together is a sign the stall is a real regression
    # (bad LR, overfitting, a bug) rather than eval noise -- see pipeline.py's
    # --min-force-promote-winrate for the guard that's meant to catch that.
    for it in forced_iters:
        ax1.axvline(x=it, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    if len(forced_iters) > 0:
        ax1.axvline(x=forced_iters.iloc[0], color='red', linestyle=':', alpha=0.6,
                     linewidth=1.5, label="Forced promotion")
        ax1.legend(loc='upper left')

    plt.title("Model Strength Progression")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "strength_metrics.png"))
    plt.close()

    print(f"Plots successfully generated and saved to {save_dir}/")

    # Quick text summary -- the plots are for trends, this is for "is
    # something wrong right now".
    total_promoted = int(df["Promoted"].sum()) if "Promoted" in df.columns else None
    total_forced = int(df["ForcedPromotion"].sum()) if has_forced_col else None
    current_streak = int(df["ConsecutiveRejections"].iloc[-1]) if "ConsecutiveRejections" in df.columns else None

    print(f"\nSummary over {len(df)} logged iteration(s):")
    if total_promoted is not None:
        print(f"  Promotions:            {total_promoted}")
    if total_forced is not None:
        print(f"  ...of which forced:    {total_forced}"
              f"{'  <-- worth a look, see the red markers above' if total_forced > 0 else ''}")
    if current_streak is not None:
        print(f"  Current reject streak: {current_streak}")

if __name__ == "__main__":
    plot_training_metrics()
