import numpy as np

class PhishingEnv:
    def __init__(self, states, labels):
        self.states = states
        self.labels = labels
        self.index = 0

    def reset(self):
        self.index = 0
        return self.states[self.index]

    def step(self, action):
        # Check if guess matches real label
        correct = action == self.labels[self.index]
        
        # Reward: +1 for correct, -1 for incorrect
        reward = 1 if correct else -1

        self.index += 1
        done = self.index >= len(self.states)

        next_state = None if done else self.states[self.index]
        return next_state, reward, done