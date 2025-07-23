# predict.py (Updated for Q-learning tensor model)
import pickle
import numpy as np
import random
import os

model_path = 'sdwan_results/sdwan_tensor_rl_model.pkl'
model_data = None
epsilon = 0.1  # Default epsilon for epsilon-greedy

def load_model():
    """Load the Q-learning model from pickle file."""
    global model_data, epsilon
    
    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found at {model_path}. Using random actions.")
        return False
        
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            
        print(f"[DEBUG] Loaded Q-learning model successfully")
        print(f"[DEBUG] Q-table shape: {model_data['Q'].shape}")
        print(f"[DEBUG] State dimensions: {model_data['state_dim']}")
        print(f"[DEBUG] Action dimensions: {model_data['action_dim']}")
        
        # Use the stored epsilon value
        if 'epsilon' in model_data:
            epsilon = model_data['epsilon']
            print(f"[DEBUG] Using epsilon: {epsilon}")
        
        return True
                
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def state_to_indices(state, state_dim):
    """Convert continuous state to discrete indices for Q-table lookup."""
    try:
        # Normalize state values to [0, 1] then scale to state dimensions
        normalized_state = np.array(state)
        
        # Assuming state is [dpid, in_port, bandwidth_util, network_load]
        # Map to discrete state space
        indices = []
        
        # DPID (switch ID) - map to first dimension
        dpid_idx = int(normalized_state[0]) % state_dim[0]
        indices.append(dpid_idx)
        
        # Port number - map to second dimension  
        port_idx = int(normalized_state[1]) % state_dim[1]
        indices.append(port_idx)
        
        # Bandwidth utilization (0-100) -> discrete bins
        bw_idx = min(int(normalized_state[2] / 100.0 * state_dim[2]), state_dim[2] - 1)
        indices.append(bw_idx)
        
        # Network load (0-100) -> discrete bins
        load_idx = min(int(normalized_state[3] / 100.0 * state_dim[3]), state_dim[3] - 1)
        indices.append(load_idx)
        
        return tuple(indices)
        
    except Exception as e:
        print(f"[ERROR] State indexing failed: {e}")
        # Return random valid indices
        return tuple(np.random.randint(0, dim) for dim in state_dim)

def get_network_state(datapath, in_port, packet_info=None):
    """Generate network state for RL model input."""
    try:
        dpid = datapath.id
        # Create a 4-dimensional state vector
        state = [
            float(dpid % 100),  # Switch ID
            float(in_port),     # Input port
            np.random.uniform(0, 100),  # Bandwidth utilization
            np.random.uniform(0, 100)   # Network load
        ]
        return np.array(state)
        
    except Exception as e:
        print(f"[ERROR] Error generating network state: {e}")
        return np.random.uniform(low=0.0, high=100.0, size=(4,))

def select_action(state):
    """Select action using Q-learning with epsilon-greedy policy."""
    global model_data, epsilon
    
    if model_data is None:
        print("[DEBUG] No model loaded, using random action")
        return random.choice([0, 1])
    
    try:
        # Convert state to indices for Q-table lookup
        state_indices = state_to_indices(state, model_data['state_dim'])
        print(f"[DEBUG] State indices: {state_indices}")
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Exploration: random action
            # Map from multi-dimensional action space to binary [0,1]
            action_indices = tuple(np.random.randint(0, dim) for dim in model_data['action_dim'])
            action = np.random.choice([0, 1])  # Simple binary action
            print(f"[DEBUG] Exploration - Random action: {action}")
        else:
            # Exploitation: choose best action from Q-table
            q_values = model_data['Q'][state_indices]
            print(f"[DEBUG] Q-values shape: {q_values.shape}")
            
            # Find the action with maximum Q-value
            best_action_indices = np.unravel_index(np.argmax(q_values), q_values.shape)
            print(f"[DEBUG] Best action indices: {best_action_indices}")
            
            # Map multi-dimensional action to binary action for routing
            # Use the first action dimension as primary decision
            action = best_action_indices[0] % 2  # Convert to binary [0,1]
            print(f"[DEBUG] Exploitation - Q-table action: {action}")
        
        print(f"[INFO] Selected action: {action}")
        return action
        
    except Exception as e:
        print(f"[ERROR] Q-learning prediction failed: {e}")
        return random.choice([0, 1])

def update_epsilon(new_epsilon):
    """Update exploration rate."""
    global epsilon
    epsilon = max(0.01, min(1.0, new_epsilon))  # Keep epsilon between 0.01 and 1.0
    print(f"[INFO] Updated epsilon to: {epsilon}")

def random_state():
    """Generate random state (backward compatibility)."""
    return np.random.uniform(low=0.0, high=100.0, size=(4,)).tolist()

def get_model_info():
    """Return information about the loaded model."""
    if model_data is None:
        return "No Q-learning model loaded"
    return f"Q-learning model - Q-table shape: {model_data['Q'].shape}, epsilon: {epsilon}"
