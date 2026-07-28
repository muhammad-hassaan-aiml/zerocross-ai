import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime
import os
from network import ZeroCrossNet

# 1. Dataset Wrapper with Corrected Legal Masking
class ZeroCrossDataset(Dataset):
    def __init__(self, data_tuples):
        """data_tuples is a list of (state_486, policy_81, reward_scalar)"""
        self.data = data_tuples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, reward = self.data[idx]
        
        # Reconstruct spatial tensor [6, 9, 9] from flat array
        state_tensor = torch.tensor(state, dtype=torch.float32).view(6, 9, 9)
        
        # CORRECTED: True legal mask requires the cell to be in an active grid AND empty
        c0 = state_tensor[0].flatten().bool() # Current player pieces
        c1 = state_tensor[1].flatten().bool() # Opponent pieces
        c2 = state_tensor[2].flatten().bool() # Active macro grids
        
        legal_mask = c2 & ~c0 & ~c1
        
        policy_tensor = torch.tensor(policy, dtype=torch.float32)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        
        return state_tensor, legal_mask, policy_tensor, reward_tensor

# 2. The Custom AlphaZero Loss Function Upgraded with Entropy
def alphazero_loss_and_metrics(pred_logits, pred_values, target_policies, target_rewards, legal_masks):
    # A. Value Loss Mean Squared Error
    value_loss = F.mse_loss(pred_values, target_rewards)
    
    # B. Policy Loss Cross Entropy with Soft Targets
    # Mask illegal moves with -1e9 before taking log_softmax to prevent NaNs
    masked_logits = pred_logits.masked_fill(~legal_masks, -1e9)
    log_preds = F.log_softmax(masked_logits, dim=1)
    
    # Cross entropy: sum of target_prob * log_pred_prob
    policy_loss = -(target_policies * log_preds).sum(dim=1).mean()
    
    # C. Entropy Creativity and Diversity of the network predictions
    probs = F.softmax(masked_logits, dim=1)
    entropy = -torch.sum(probs * log_preds, dim=1).mean()
    
    return value_loss + policy_loss, value_loss.item(), policy_loss.item(), entropy.item()

# 3. The Training Loop Upgraded for Checkpoint Persistence
def train_network(net, dataset_tuples, batch_size=64, epochs=10, lr=0.001, device='cpu', optimizer_state=None):
    dataset = ZeroCrossDataset(dataset_tuples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Adam optimizer with Weight Decay L2 Regularization
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
    # Reload optimizer momentum for long Kaggle campaigns
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        # Force the dynamically scheduled LR overriding the old saved one
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    net.to(device)
    
    total_pi_loss, total_v_loss, total_entropy = 0.0, 0.0, 0.0
    batch_count = 0
    
    for epoch in range(epochs):
        net.train() # Unlock BatchNorm layers
        
        for states, masks, target_policies, target_rewards in dataloader:
            states, masks = states.to(device), masks.to(device)
            target_policies, target_rewards = target_policies.to(device), target_rewards.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, values = net(states)
            
            # Calculate Loss and Metrics
            loss, v_loss, p_loss, entropy = alphazero_loss_and_metrics(logits, values, target_policies, target_rewards, masks)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_v_loss += v_loss
            total_pi_loss += p_loss
            total_entropy += entropy
            batch_count += 1

    # Finalize Averages for logging
    avg_pi_loss = total_pi_loss / max(1, batch_count)
    avg_v_loss = total_v_loss / max(1, batch_count)
    avg_entropy = total_entropy / max(1, batch_count)
    
    metrics = {
        "pi_loss": round(avg_pi_loss, 4),
        "v_loss": round(avg_v_loss, 4),
        "entropy": round(avg_entropy, 4)
    }
    
    return net, optimizer.state_dict(), metrics

if __name__ == "__main__":
    from self_play import SelfPlayWorker
    
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6) else "cpu")
            
    print(f"Initializing on {device}")
    
    net = ZeroCrossNet()
    
    # Pipeline configuration variables
    GAMES_TO_PLAY = 2
    CURRENT_GENERATION = 1
    
    print(f"Generating {GAMES_TO_PLAY} games of self play data for pipeline verification")
    worker = SelfPlayWorker(net, num_concurrent_games=2, mcts_simulations=25)
    training_data = worker.generate_data(total_games_to_play=GAMES_TO_PLAY)
    
    print(f"Training on {len(training_data)} augmented samples")
    trained_net, opt_state, metrics = train_network(net, training_data, batch_size=32, epochs=5, lr=0.001, device=device)
    
    os.makedirs("models", exist_ok=True)
    
    # Rich metadata dictionary payload
    checkpoint_data = {
        'generation': CURRENT_GENERATION,
        'games_played': GAMES_TO_PLAY,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': metrics,
        'model_state_dict': trained_net.state_dict(),
        'optimizer_state_dict': opt_state
    }
    
    save_path = "models/latest_checkpoint.pth"
    torch.save(checkpoint_data, save_path)
    
    print(f"Pipeline Verified! Metrics extracted: {metrics}")
    print(f"Checkpoint with full metadata saved to {save_path}")
    
    # Verify reload logic works
    verify_net = ZeroCrossNet()
    verify_net.load_checkpoint(save_path)
    print("Verification successful: Metadata-rich checkpoint loaded smoothly into ZeroCrossNet.")