import torch
import math
from network import ZeroCrossNet

def test_network_shapes_and_masking():
    print("Initializing ZeroCrossNet...")
    net = ZeroCrossNet()
    net.eval()
    
    # Simulate a raw flat array coming from C++ GameState::encode()
    flat_state = [0.0] * 486
    
    # Simulate a legal mask where only moves 0 and 80 are legal
    legal_mask = [False] * 81
    legal_mask[0] = True
    legal_mask[80] = True
    
    print("Testing predict() inference wrapper...")
    policy, value = net.predict(flat_state, legal_mask)
    
    # 1. Verify Pybind11 Safety Types
    assert isinstance(policy, list), f"Policy must be a Python list, got {type(policy)}"
    assert isinstance(value, float), f"Value must be a Python float, got {type(value)}"
    
    # 2. Verify Output Dimensions
    assert len(policy) == 81, f"Expected policy length 81, got {len(policy)}"
    
    # 3. Verify Value Bounds
    assert -1.0 <= value <= 1.0, f"Value {value} is strictly outside [-1, 1]"
    
    # 4. Verify Masking Logic (Illegal moves should be zeroed out)
    for i in range(81):
        if i == 0 or i == 80:
            assert policy[i] > 0.0, f"Legal move {i} should have non-zero probability."
        else:
            # Due to floating point precision, it might not be exactly 0, but very close
            assert policy[i] < 1e-10, f"Illegal move {i} should have ~0 probability, got {policy[i]}"
            
    # 5. Verify Probability Normalization
    prob_sum = sum(policy)
    assert math.isclose(prob_sum, 1.0, rel_tol=1e-5), f"Policy probabilities must sum to 1.0, got {prob_sum}"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_network_shapes_and_masking()