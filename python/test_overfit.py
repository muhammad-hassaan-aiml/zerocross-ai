import torch
import torch.nn.functional as F
import torch.optim as optim
from network import ZeroCrossNet

def test_overfit():
    print("Starting Network Overfit Test...")
    
    # Safe device selection
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 6:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running overfit test on: {device}")
    
    net = ZeroCrossNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    
    # Create a tiny static batch (Batch Size = 4)
    B = 4
    states = torch.rand((B, 6, 9, 9), device=device)
    
    # Target values between -1 and 1
    target_values = torch.tensor([[1.0], [-1.0], [0.0], [1.0]], device=device)
    
    # Target policies (one-hot for simplicity)
    target_policies = torch.zeros((B, 81), device=device)
    target_policies[0, 10] = 1.0
    target_policies[1, 42] = 1.0
    target_policies[2, 80] = 1.0
    target_policies[3, 0] = 1.0
    
    # Legal masks (allow the target actions, plus some random noise)
    legal_masks = torch.zeros((B, 81), dtype=torch.bool, device=device)
    legal_masks[0, 10] = True; legal_masks[0, 11] = True
    legal_masks[1, 42] = True; legal_masks[1, 43] = True
    legal_masks[2, 80] = True; legal_masks[2, 79] = True
    legal_masks[3, 0] = True;  legal_masks[3, 1] = True
    
    # Train on this exact same batch for 500 epochs
    epochs = 500
    for epoch in range(epochs):
        net.train()
        optimizer.zero_grad()
        
        logits, values = net(states)
        
        # Mask illegal moves
        logits = logits.masked_fill(~legal_masks, -1e4)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Calculate losses
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, target_values)
        loss = policy_loss + value_loss
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Total Loss: {loss.item():.6f} | Pi Loss: {policy_loss.item():.6f} | V Loss: {value_loss.item():.6f}")

    # Validation
    net.eval()
    with torch.no_grad():
        test_logits, test_values = net(states)
        test_logits = test_logits.masked_fill(~legal_masks, -1e4)
        test_probs = F.softmax(test_logits, dim=1)
        
        print("\n--- Overfit Verification ---")
        for i in range(B):
            pred_v = test_values[i].item()
            tgt_v = target_values[i].item()
            pred_pi_max = test_probs[i].max().item()
            pred_pi_argmax = test_probs[i].argmax().item()
            tgt_pi_argmax = target_policies[i].argmax().item()
            
            print(f"Sample {i+1}:")
            print(f"  Value: Target {tgt_v:+.2f} | Predicted {pred_v:+.4f}")
            print(f"  Policy: Target Action {tgt_pi_argmax} | Predicted Action {pred_pi_argmax} (Prob: {pred_pi_max:.4f})")
            
    if loss.item() < 0.05:
        print("\nSUCCESS: Network successfully overfit the batch down to near-zero loss. Architecture and masking are perfectly healthy!")
    else:
        print("\nFAILED: Network failed to memorize the batch. Check architecture, loss calculation, or masking logic.")

if __name__ == "__main__":
    test_overfit()