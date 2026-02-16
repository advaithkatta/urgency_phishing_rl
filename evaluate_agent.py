import numpy as np
from sklearn.metrics import f1_score, classification_report

# 1. Load Data and the Weights you just trained
states = np.load("states.npy")
labels = np.load("labels.npy")
weights = np.load("trained_weights.npy")

print("--- Evaluating Trained RL Agent ---")

# 2. Make Predictions
# We calculate Q(s,0) and Q(s,1) for every email and pick the winner
q_values_0 = np.dot(states, weights[0])
q_values_1 = np.dot(states, weights[1])
predictions = (q_values_1 > q_values_0).astype(int)

# 3. Calculate Results
f1 = f1_score(labels, predictions)
print(f"RL Agent F1 Score: {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(labels, predictions, target_names=["Normal", "Urgent Phishing"]))