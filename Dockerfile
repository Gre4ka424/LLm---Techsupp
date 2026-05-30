# Use NVIDIA CUDA base image to support GPU acceleration
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Add Hugging Face Token for LLaMA-3 base model download
ENV HF_TOKEN="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"

# Install Python 3, pip, and essential C++ build tools required by bitsandbytes
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Explicitly install 'peft' to handle LoRA adapter weights (using latest to avoid config mismatch)
RUN pip3 install --no-cache-dir peft

# Copy model weights and application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8080

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]