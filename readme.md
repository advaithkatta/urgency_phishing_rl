# Urgency Phishing RL

![License](https://img.shields.io/badge/license-CC%20BY--SA%204.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.6%2B-blue.svg)
![License](https://img.shields.io/github/license/yourusername/phishing-email-dataset)


Phishing attacks are one of the most prevalent cybersecurity threats, targeting people to steal sensitive information or financial data. Detecting phishing emails accurately is crucial for mitigating these risks. This repo is trained on the Enron Email Dataset (1.7 GB) and uses reinforcement learning to adapt when it doesn't guess the phishing or legitimate labels correctly.
 
## Enron Email Dataset Attributes

- **Email ID:** Unique identifier for each email.
- **Sender:** Email address of the sender.
- **Recipient:** Email address of the recipient.
- **Subject:** Subject line of the email.
- **Body:** Main content of the email.
- **Timestamp:** Date and time when the email was sent.
- **Attachments:** List of any attachments included.
- **URL Links:** Extracted URLs from the email body.
- **Labels:** 
  - `phishing` 
  - `legitimate`

## Getting Started (Docker)

1. **Clone the Repository**

   ```bash
   git clone https://github.com/rokibulroni/Phishing-Email-Dataset.git
   cd Phishing-Email-Dataset
