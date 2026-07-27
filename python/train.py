import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
        c2 = state_tensor[2].flatten().bool() # Active macro-grids
        
        legal_mask = c2 & ~c0 & ~c1
        
        policy_tensor = torch.tensor(policy, dtype=torch.float32)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        
        return state_tensor, legal_mask, policy_tensor, reward_tensor

# 2. The Custom AlphaZero Loss Function
def alphazero_loss(pred_logits, pred_values, target_policies, target_rewards, legal_masks):
    # A. Value Loss (Mean Squared Error)
    value_loss = F.mse_loss(pred_values, target_rewards)
    
    # B. Policy Loss (Cross Entropy with Soft Targets)
    # Mask illegal moves with -1e9 before taking log_softmax to prevent NaNs
    masked_logits = pred_logits.masked_fill(~legal_masks, -1e9)
    log_preds = F.log_softmax(masked_logits, dim=1)
    
    # Cross entropy: sum of (target_prob * log_pred_prob)
    policy_loss = -(target_policies * log_preds).sum(dim=1).mean()
    
    # Total Loss (L2 weight decay is handled automatically by the Adam optimizer)
    return value_loss + policy_loss, value_loss.item(), policy_loss.item()

# 3. The Training Loop
def train_network(net, dataset_tuples, batch_size=64, epochs=10, lr=0.001, device='cpu'):
    dataset = ZeroCrossDataset(dataset_tuples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Adam optimizer with Weight Decay (L2 Regularization)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    net.to(device)
    
    for epoch in range(epochs):
        net.train() # Unlock BatchNorm layers
        total_loss, total_v_loss, total_p_loss = 0, 0, 0
        
        for states, masks, target_policies, target_rewards in dataloader:
            states, masks = states.to(device), masks.to(device)
            target_policies, target_rewards = target_policies.to(device), target_rewards.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (getting raw logits)
            logits, values = net(states)
            
            # Calculate Custom Loss
            loss, v_loss, p_loss = alphazero_loss(logits, values, target_policies, target_rewards, masks)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_v_loss += v_loss
            total_p_loss += p_loss
            
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Total Loss: {total_loss/len(dataloader):.4f} | "
              f"Policy Loss: {total_p_loss/len(dataloader):.4f} | "
              f"Value Loss: {total_v_loss/len(dataloader):.4f}")
              
    return net

if __name__ == "__main__":
    import os
    from self_play import SelfPlayWorker
    
    # Safe hardware device selection (avoids CC 5.0 GPU architecture mismatch)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Notice: Using CPU because local GPU is below Compute Capability 7.0")
        
    print(f"Initializing on {device}...")
    
    # 1. Initialize Network
    net = ZeroCrossNet()
    
    # 2. Generate Dummy Data via Self-Play
    print("Generating 2 games of self-play data for pipeline verification...")
    worker = SelfPlayWorker(net, num_concurrent_games=2, mcts_simulations=25)
    training_data = worker.generate_data(total_games_to_play=2)
    
    # 3. Train the Network
    print(f"Training on {len(training_data)} augmented samples...")
    trained_net = train_network(net, training_data, batch_size=32, epochs=5, lr=0.001, device=device)
    
    # 4. Save Checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(trained_net.state_dict(), "models/latest_checkpoint.pth")
    print("Pipeline Verified! Checkpoint saved to models/latest_checkpoint.pth")