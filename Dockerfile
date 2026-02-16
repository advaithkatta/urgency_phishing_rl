# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn

# Run the test environment by default
CMD ["python", "test_env.py"]