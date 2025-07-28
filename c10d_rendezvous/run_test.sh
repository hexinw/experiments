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

# Start the master node first
echo "Starting master node..."
docker-compose up -d master

# Wait a moment for master to be ready
sleep 2

# Calculate number of worker nodes
WORKER_COUNT=$((TOTAL_NODES - 1))
BATCH_SIZE=50

if [ "$WORKER_COUNT" -gt 0 ]; then
    echo "Starting $WORKER_COUNT worker nodes in batches of $BATCH_SIZE..."

    # Start workers in batches
    for ((start=1; start<=WORKER_COUNT; start+=BATCH_SIZE)); do
        end=$((start + BATCH_SIZE - 1))
        if [ "$end" -gt "$WORKER_COUNT" ]; then
            end=$WORKER_COUNT
        fi

        echo "Starting workers $start to $end..."

        # Create a list of services to start in this batch
        services=""
        for ((i=start; i<=end; i++)); do
            if [ -n "$services" ]; then
                services="$services worker$i"
            else
                services="worker$i"
            fi
        done

        # Start the batch
        docker-compose up -d $services

        # Wait a moment between batches
        if [ "$end" -lt "$WORKER_COUNT" ]; then
            echo "Waiting 2 seconds before next batch..."
            sleep 2
        fi
    done
fi

# Wait for all nodes to join
echo "Waiting for nodes to join..."
sleep 3

# Monitor the logs
docker-compose logs -f

# To stop the test:
# docker-compose down
