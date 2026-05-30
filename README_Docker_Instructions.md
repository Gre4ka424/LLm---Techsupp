# E-Commerce LLM Support Agent

## Project Overview
This project is a fine-tuned Large Language Model (based on LLaMA-3 8B) designed for e-commerce customer support. It features a custom Context Buffer for multi-turn conversations and is deployed as a FastAPI microservice.

**Important Note:** The configurations and code provided in this repository are strictly for local execution. They do not contain valid ngrok auth tokens and are not configured for public tunneling out of the box.

## Prerequisites
To run this project locally, ensure you have the following installed on your system:
* Docker Desktop (with WSL2 enabled if on Windows)
* NVIDIA GPU with appropriate drivers (for local inference)

## How to Run the Project
1. Extract the archive (or clone the repository) and open a terminal/command prompt in the root directory (where the `docker-compose.yml` file is located).
2. Build and start the container by running the following command:
   ```bash
   docker compose up --build