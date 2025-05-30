# c10d_rendezvous: Rendezvous Scaling Performance Test

This directory contains a minimal setup to test the scaling performance of PyTorch's `c10d` rendezvous backend using Docker Compose.

## 📂 Directory Structure

- `Dockerfile` – Defines the container image with necessary dependencies.
- `generate_compose.py` – Dynamically generates a `docker-compose.yml` file with the desired number of rendezvous nodes.
- `test_rendezvous.py` – PyTorch script that runs a distributed rendezvous test.
- `run_test.sh` – Entry-point script to build the image, generate the Docker Compose file, and launch the test.

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd c10d_rendezvous
