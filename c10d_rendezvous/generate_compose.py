#!/usr/bin/env python3
import sys

if len(sys.argv) != 2:
    print("Usage: python generate_compose.py <TOTAL_NODES>")
    sys.exit(1)

TOTAL_NODES = int(sys.argv[1])
if TOTAL_NODES < 2:
    print("TOTAL_NODES must be at least 2 (1 master + at least 1 worker)")
    sys.exit(1)

#    network_mode: "host"
header = f"""version: '3.8'

services:
  master:
    image: test_rendezvous:latest
    container_name: test_rendezvous_master
    network_mode: host
    command: >
      --master-addr 127.0.0.1
      --master-port 29500
      --node-id 0
      --total-nodes {TOTAL_NODES}
      --min-nodes {TOTAL_NODES}
      --max-nodes {TOTAL_NODES}
      --join-timeout 600
      --last-call-timeout 30
    environment:
      - FT_LAUNCHER_LOGLEVEL=DEBUG
"""

worker_template = """
  worker{node_id}:
    image: test_rendezvous:latest
    container_name: test_rendezvous_worker{node_id}
    network_mode: host
    environment:
      - NODE_ID={node_id}
      - FT_LAUNCHER_LOGLEVEL=DEBUG
    command: >
      --master-addr 127.0.0.1
      --master-port 29500
      --node-id {node_id}
      --total-nodes {total}
      --min-nodes {total}
      --max-nodes {total}
      --join-timeout 600
      --last-call-timeout 30
"""

with open("docker-compose.yml", "w") as f:
    f.write(header)
    for node_id in range(1, TOTAL_NODES):
        f.write(worker_template.format(node_id=node_id, total=TOTAL_NODES))

print(f"✅ Generated docker-compose.yml with {TOTAL_NODES - 1} workers and 1 master.")
