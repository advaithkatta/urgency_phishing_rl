import pandas as pd

df = pd.read_csv("Enron.csv")

urgency_keywords = [
    "urgent",
    "action required",
    "verify",
    "immediately",
    "suspend",
    "24 hours"
]

def is_urgent(text):
    text = str(text).lower()
    return any(k in text for k in urgency_keywords)

df["combined_text"] = (
    df["subject"].fillna("") + " " + df["body"].fillna("")
)

df["urgent_label"] = (
    (df["label"] == 1) &
    (df["combined_text"].apply(is_urgent))
).astype(int)

print("Urgency label counts:")
print(df["urgent_label"].value_counts())

df.to_csv("Enron_urgency_labeled.csv", index=False)
