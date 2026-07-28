import numpy as np

def to_spatial_grid_state(state_flat):
    # Reshape to (6 channels, 3 macro_row, 3 macro_col, 3 micro_row, 3 micro_col)
    tensor = np.array(state_flat, dtype=np.float32).reshape(6, 3, 3, 3, 3)
    # Transpose to (6 channels, macro_row, micro_row, macro_col, micro_col)
    tensor = tensor.transpose(0, 1, 3, 2, 4)
    # Combine into true spatial grid (6, 9, 9)
    return tensor.reshape(6, 9, 9)

def to_flat_state(state_spatial):
    # Reshape back to (6 channels, macro_row, micro_row, macro_col, micro_col)
    tensor = state_spatial.reshape(6, 3, 3, 3, 3)
    # Transpose back to (6 channels, macro_row, macro_col, micro_row, micro_col)
    tensor = tensor.transpose(0, 1, 3, 2, 4)
    return tensor.flatten().tolist()

def to_spatial_grid_policy(policy_flat):
    # Reshape to (3 macro_row, 3 macro_col, 3 micro_row, 3 micro_col)
    tensor = np.array(policy_flat, dtype=np.float32).reshape(3, 3, 3, 3)
    # Transpose to (macro_row, micro_row, macro_col, micro_col)
    tensor = tensor.transpose(0, 2, 1, 3)
    # Combine into true spatial grid (9, 9)
    return tensor.reshape(9, 9)

def to_flat_policy(policy_spatial):
    # Reshape back to (macro_row, micro_row, macro_col, micro_col)
    tensor = policy_spatial.reshape(3, 3, 3, 3)
    # Transpose back to (macro_row, macro_col, micro_row, micro_col)
    tensor = tensor.transpose(0, 2, 1, 3)
    return tensor.flatten().tolist()

def get_symmetries(state_flat, policy_flat, reward):
    """
    Takes a flat state (486) and flat policy (81), applies D4 symmetry (rotations & reflections),
    and returns a list of 8 augmented (state, policy, reward) tuples.
    """
    # 1. Reshape flat arrays into spatial grids
    state_grid = to_spatial_grid_state(state_flat)
    policy_grid = to_spatial_grid_policy(policy_flat)
    
    augmented_data = []
    
    for i in range(4):
        # 2. Rotate by i * 90 degrees
        # State axes=(1, 2) rotates the H and W dimensions, leaving the 6 channels alone.
        rot_state = np.rot90(state_grid, k=i, axes=(1, 2))
        rot_policy = np.rot90(policy_grid, k=i, axes=(0, 1))
        
        augmented_data.append((
            to_flat_state(rot_state), 
            to_flat_policy(rot_policy), 
            reward
        ))
        
        # 3. Reflect (Horizontal flip of the rotated grids)
        flip_state = np.flip(rot_state, axis=2)
        flip_policy = np.flip(rot_policy, axis=1)
        
        augmented_data.append((
            to_flat_state(flip_state), 
            to_flat_policy(flip_policy), 
            reward
        ))
        
    return augmented_data