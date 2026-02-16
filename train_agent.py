import numpy as np
from phishing_env import PhishingEnv

# --- Load Dataset ---
try:
    states = np.load("states.npy")
    labels = np.load("labels.npy")
    env = PhishingEnv(states, labels)
    print(f"Dataset Loaded: {len(states)} samples.")
except FileNotFoundError:
    print("Error: .npy files not found.")
    exit()

# --- Hyperparameters ---
alpha = 0.05   
gamma = 0.95   
epsilon = 0.2  

# --- Initialize Linear Weights ---
num_features = states.shape[1]
weights = np.zeros((2, num_features))

def get_q_value(state, action):
    return np.dot(state, weights[action])

# --- Training Loop ---
print("--- Starting Training ---")

total_reward = 0
state = env.reset()

for i in range(len(states)):
    # Action selection (Epsilon-Greedy)
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1])
    else:
        q_values = [get_q_value(state, a) for a in [0, 1]]
        action = np.argmax(q_values)

    # Custom Reward Logic to handle class imbalance
    true_label = labels[i]
    if action == true_label:
        reward = 50 if true_label == 1 else 1 
    else:
        reward = -100 if true_label == 1 else -5 

    next_state, _, done = env.step(action)
    total_reward += reward

    # Q-Learning Weight Update
    if not done:
        next_q_values = [get_q_value(next_state, a) for a in [0, 1]]
        max_next_q = np.max(next_q_values)
        
        target = reward + gamma * max_next_q
        prediction = get_q_value(state, action)
        error = target - prediction
        
        # Update weights for the specific action taken
        weights[action] += alpha * error * state
        state = next_state
    else:
        break

    if i % 10000 == 0:
        print(f"Progress: {i}/{len(states)} | Reward: {total_reward}")

# --- Save Model ---
print(f"Training Complete. Final Reward: {total_reward}")
np.save("trained_weights.npy", weights)
print("Saved: trained_weights.npy")