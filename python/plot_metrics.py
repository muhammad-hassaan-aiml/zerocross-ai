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

    # 3. Plot Elo & Win Rates
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

    plt.title("Model Strength Progression")
    fig.tight_layout()  
    plt.savefig(os.path.join(save_dir, "strength_metrics.png"))
    plt.close()

    print(f"Plots successfully generated and saved to {save_dir}/")

if __name__ == "__main__":
    plot_training_metrics()