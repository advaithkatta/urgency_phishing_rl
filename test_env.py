import numpy as np
from phishing_env import PhishingEnv

# Load the big files you just generated
states = np.load("states.npy")
labels = np.load("labels.npy")

# Start the environment
env = PhishingEnv(states, labels)
state = env.reset()

total_reward = 0

# Test with 100 random guesses
for _ in range(100):
    action = np.random.choice([0, 1])
    state, reward, done = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Test Successful! Random agent total reward: {total_reward}")