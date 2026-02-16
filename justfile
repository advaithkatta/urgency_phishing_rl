set shell := ["powershell.exe", "-c"]

# List available commands
default:
    @just --list


    python TDFIFprep.py
    python create_urgency_labels.py
    python prepare_features.py


    python train_agent.py
    python evaluate_agent.py
    python baseline_urgency.py

# Run everything in Docker
docker-build:
    docker build -t phishing-rl .

docker-run:
    docker run phishing-rl