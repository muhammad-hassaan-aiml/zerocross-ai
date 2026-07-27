import numpy as np

def get_symmetries(state_flat, policy_flat, reward):
    """
    Takes a flat state (486) and flat policy (81), applies D4 symmetry (rotations & reflections),
    and returns a list of 8 augmented (state, policy, reward) tuples.
    """
    # 1. Reshape flat arrays into spatial grids
    state_grid = np.array(state_flat, dtype=np.float32).reshape(6, 9, 9)
    policy_grid = np.array(policy_flat, dtype=np.float32).reshape(9, 9)
    
    augmented_data = []
    
    for i in range(4):
        # 2. Rotate by i * 90 degrees
        # State axes=(1, 2) rotates the H and W dimensions, leaving the 6 channels alone.
        rot_state = np.rot90(state_grid, k=i, axes=(1, 2))
        rot_policy = np.rot90(policy_grid, k=i, axes=(0, 1))
        
        augmented_data.append((
            rot_state.flatten().tolist(), 
            rot_policy.flatten().tolist(), 
            reward
        ))
        
        # 3. Reflect (Horizontal flip of the rotated grids)
        flip_state = np.flip(rot_state, axis=2)
        flip_policy = np.flip(rot_policy, axis=1)
        
        augmented_data.append((
            flip_state.flatten().tolist(), 
            flip_policy.flatten().tolist(), 
            reward
        ))
        
    return augmented_data