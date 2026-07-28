import torch
import os

def validate_replay_buffer(filepath="models/replay_buffer.pt"):
    print(f"Starting Data Integrity Check on {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"FAILED: File {filepath} not found.")
        return

    # Use weights_only=False since we are loading custom tuple/list objects
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    
    if not data:
        print("FAILED: Buffer is empty.")
        return

    # 1. Unwrap the metadata dictionary if it exists
    if isinstance(data, dict):
        print(f"Detected dictionary wrapper with keys: {list(data.keys())}")
        if "buffer" in data:
            buffer = data["buffer"]
        elif "data" in data:
            buffer = data["data"]
        else:
            # Column-based dict fallback
            states, policies, values = data.get("states"), data.get("policies"), data.get("values")
            if states is not None and policies is not None and values is not None:
                buffer = list(zip(states, policies, values))
            else:
                print("FAILED: Could not locate the actual buffer data inside the dictionary.")
                return
    else:
        buffer = data

    # 2. Dynamically handle column-based (tuple of tensors/lists) vs row-based (list of tuples)
    if isinstance(buffer, tuple) or (isinstance(buffer, list) and len(buffer) > 0 and not isinstance(buffer[0], (tuple, list))):
        states, policies, values = buffer[0], buffer[1], buffer[2]
        num_samples = len(states)
        iterator = zip(states, policies, values)
    else:
        num_samples = len(buffer)
        iterator = buffer

    print(f"Validating {num_samples} samples...")

    invalid_shapes = 0
    invalid_policies = 0
    invalid_values = 0
    
    for item in iterator:
        # Account for potential 4-item tuples (state, pi, v, mask) or 3-item tuples
        state, pi, v = item[0], item[1], item[2]
        
        # Cast to tensors to safely handle raw lists/numpy arrays returned by C++ bindings
        state_t = torch.as_tensor(state, dtype=torch.float32)
        pi_t = torch.as_tensor(pi, dtype=torch.float32)
        
        # Check total elements to allow both flat and [6, 9, 9] shapes (6*9*9 = 486)
        if state_t.numel() != 486 or pi_t.numel() != 81:
            invalid_shapes += 1
            
        pi_sum = pi_t.sum().item()
        if abs(pi_sum - 1.0) > 1e-4:
            invalid_policies += 1
            
        if v < -1.0 or v > 1.0:
            invalid_values += 1

    if invalid_shapes > 0 or invalid_policies > 0 or invalid_values > 0:
        print("\nFAILED: Data integrity issues found!")
        print(f"  Invalid Tensor/List Sizes: {invalid_shapes}")
        print(f"  Invalid Policies (Sum != 1.0): {invalid_policies}")
        print(f"  Invalid Values (Out of bounds): {invalid_values}")
    else:
        print("\nSUCCESS: All replay buffer samples passed integrity checks!")
        print(f"  - {num_samples} State elements match expected size (486)")
        print(f"  - {num_samples} Policy vectors match expected size (81) and sum to exactly 1.0")
        print(f"  - {num_samples} Value targets are correctly bounded in [-1.0, 1.0]")

if __name__ == "__main__":
    validate_replay_buffer()