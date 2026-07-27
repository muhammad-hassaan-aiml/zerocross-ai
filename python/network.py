import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Skip connection safely before the final ReLU
        out += identity
        out = F.relu(out)
        return out

class ZeroCrossNet(nn.Module):
    def __init__(self, num_res_blocks=4):
        super().__init__()
        
        # Input stem: [B, 6, 9, 9] -> [B, 64, 9, 9]
        self.stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Tower
        self.res_blocks = nn.ModuleList([ResBlock(64) for _ in range(num_res_blocks)])
        
        # Policy Head: raw logits for 81 actions
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 81, 81)
        )
        
        # Value Head: scalar outcome prediction in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(81, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """Used directly during the training loop on batches of tensors."""
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
            
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    @torch.no_grad()
    def predict(self, state, legal_mask):
        """Used during self-play inference. Handles flat, unbatched arrays safely."""
        self.eval() 
        
        # 1. Convert to tensors
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(legal_mask, torch.Tensor):
            legal_mask = torch.tensor(legal_mask, dtype=torch.bool)
            
        # 2. Reshape flat arrays from C++ to 4D spatial tensors [B, C, H, W]
        if state.dim() == 1:
            state = state.view(1, 6, 9, 9)
        elif state.dim() == 2 and state.size(1) == 486:
            state = state.view(-1, 6, 9, 9)
            
        if legal_mask.dim() == 1:
            legal_mask = legal_mask.unsqueeze(0)
            
        # Device alignment
        device = next(self.parameters()).device
        state = state.to(device)
        legal_mask = legal_mask.to(device)
        
        # Forward pass
        logits, value = self.forward(state)
        
        # Apply legal mask 
        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        probabilities = F.softmax(masked_logits, dim=-1)
        
        # 3. Pybind11 Safety: Convert back to native Python list and float
        prob_list = probabilities.cpu().flatten().tolist()
        val_scalar = value.cpu().item()
        
        return prob_list, val_scalar

    def load_checkpoint(self, path):
        """Utility for loading weights."""
        self.load_state_dict(torch.load(path, map_location=next(self.parameters()).device))
        self.eval()