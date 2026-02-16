import pandas as pd

# Since you processed all CSVs, use your labeled master file
# If you don't have this CSV yet, run your 'create_urgency_labels.py' first
df = pd.read_csv("Enron_urgency_labeled.csv")

keywords = ["urgent", "action required", "verify", "immediately", "suspend", "24 hours"]

def baseline_predict(text):
    text = str(text).lower()
    return int(any(k in text for k in keywords))

# Assuming your TDFIFprep used 'combined_text' and 'urgent_label'
df["baseline_pred"] = df["combined_text"].apply(baseline_predict)

tp = ((df["baseline_pred"] == 1) & (df["urgent_label"] == 1)).sum()
fp = ((df["baseline_pred"] == 1) & (df["urgent_label"] == 0)).sum()
fn = ((df["baseline_pred"] == 0) & (df["urgent_label"] == 1)).sum()

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("--- Baseline Results ---")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")