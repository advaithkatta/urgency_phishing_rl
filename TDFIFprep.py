import pandas as pd
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION ---
dataset_folder = r"C:\Users\sreer\Phishing-Email-Dataset"
urgency_words = ["urgent", "immediately", "action required", "verify", "suspend",
                 "asap", "login now", "attention", "respond now", "critical",
                 "security alert", "unauthorized", "final notice", "24 hours"]

# Holders for the master dataset
all_texts = []
all_labels = []

# Indicators for Phishing (numeric and string)
phish_indicators = ['phishing', 'phish', 'spam', '1', 1]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'\s+', ' ', text)     # Cleanup whitespace
    return text.strip()

print("--- Starting Dataset Processing ---")

for file in os.listdir(dataset_folder):
    if file.endswith(".csv"):
        path = os.path.join(dataset_folder, file)
        try:
            # Handle encoding
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin1')

            # Column detection
            text_cols = [c for c in df.columns if any(k in c.lower() for k in ['body', 'content', 'text'])]
            label_cols = [c for c in df.columns if any(k in c.lower() for k in ['label', 'class'])]

            if not text_cols or not label_cols:
                continue

            t_col = text_cols[0]
            l_col = label_cols[0]

            # Processing rows
            for _, row in df.iterrows():
                raw_text = clean_text(row[t_col])
                
                # Logic: Is it phishing? AND does it have urgency keywords?
                is_phish = row[l_col] in phish_indicators
                has_urgency = any(word in raw_text for word in urgency_words)
                
                urgency_label = 1 if (is_phish and has_urgency) else 0
                
                all_texts.append(raw_text)
                all_labels.append(urgency_label)

            print(f"Processed {file}: Current Total = {len(all_texts)}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

# --- STEP 2: TF-IDF & SAVING ---
print("\n--- Converting to TF-IDF  ---")

# Convert list to array for processing
all_labels_arr = np.array(all_labels)

# Initialize TF-IDF
# We limit to 5000 features to keep the RL state space manageable
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
states = vectorizer.fit_transform(all_texts).toarray()

# Save the files for the RL Agent
np.save('states.npy', states)
np.save('labels.npy', all_labels_arr)

print("\n--- Summary Statistics ---")
print(f"Total Messages: {len(all_labels)}")
print(f"Urgent Phishing Found: {np.sum(all_labels_arr)}")
print(f"Urgency Percentage: {(np.sum(all_labels_arr)/len(all_labels))*100:.2f}%")
print("Files Saved: states.npy, labels.npy")
