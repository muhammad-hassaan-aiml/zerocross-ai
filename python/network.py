import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Standard Residual Block with Batch Normalization.
    Preserves spatial representation while deepening network feature extraction.
    """
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)


class ZeroCrossNet(nn.Module):
    """
    AlphaZero-style Architecture for Ultimate Tic-Tac-Toe.
    
    Specs:
      - Input:  [B, 6, 9, 9] tensor representation
      - Backbone: 128 Filters, 6 Residual Blocks (~2.0M parameters)
      - Policy Head: Outputs raw logits for 81 actions
      - Value Head:  Outputs scalar outcome evaluation in [-1, 1]

    NOTE: sized for Ultimate Tic-Tac-Toe's actual complexity, not Chess/Go scale.
    A 12-block/256-channel net (~14.4M params) is oversized for this game given
    realistic self-play volumes on a single Kaggle GPU; it just burns your
    games/hour budget on capacity the game doesn't need. Override via the
    constructor args (or pipeline.py's --num-res-blocks/--num-channels) if you
    want to scale it back up once the pipeline is proven out.
    """
    def __init__(self, num_res_blocks=6, num_channels=128):
        super().__init__()
        
        # Stem: [B, 6, 9, 9] -> [B, 128, 9, 9]
        self.stem = nn.Sequential(
            nn.Conv2d(6, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Deep Residual Tower (6 Blocks)
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 81)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
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
        """
        Used during self-play inference. Handles flat, unbatched arrays safely.
        
        WARNING: This returns post-softmax probabilities. MCTSTree::submit_result 
        expects raw logits and applies its own softmax. Do not wire this method 
        directly into the C++ MCTS evaluation loop.
        """
        self.eval() 
        
        # 1. Convert to tensors if passed as lists/numpy arrays
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
        
        # Apply legal mask (-1e4 prevents illegal actions from receiving probability)
        masked_logits = logits.masked_fill(~legal_mask, -1e4)
        probabilities = F.softmax(masked_logits, dim=-1)
        
        # 3. Convert back to native Python list and float for C++ pybind11 safety
        prob_list = probabilities.cpu().flatten().tolist()
        val_scalar = value.cpu().item()
        
        return prob_list, val_scalar

    def load_checkpoint(self, path):
        """Utility for loading saved model weights, supporting metadata-rich dictionaries."""
        checkpoint = torch.load(path, map_location=next(self.parameters()).device, weights_only=False)
        
        # Extract state dict if it's packed with training metadata
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback for legacy checkpoints containing only the raw state_dict
            self.load_state_dict(checkpoint)
            
        self.eval()