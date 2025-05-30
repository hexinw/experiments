#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: ./run_test.sh <TOTAL_NODES>"
    exit 1
fi

TOTAL_NODES="$1"

if [ "$TOTAL_NODES" -lt 1 ]; then
    echo "Total nodes must be at least 1"
    exit 1
fi

python generate_compose.py "$TOTAL_NODES"

echo "Starting rendezvous test with:"
echo "Total nodes: $TOTAL_NODES"

# Clean up any existing containers
docker-compose down --remove-orphans

# Export variables for docker-compose
export TOTAL_NODES

docker build -t test_rendezvous:latest -f Dockerfile .

# Build the Docker image
docker-compose build

# Start the master node and worker nodes.
docker-compose up -d

# Wait for all nodes to join
echo "Waiting for nodes to join..."
sleep 3

# Monitor the logs
docker-compose logs -f

# To stop the test:
# docker-compose down
