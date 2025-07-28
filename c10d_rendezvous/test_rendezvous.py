#!/usr/bin/env python3
import argparse
import logging
import time
from datetime import datetime, timedelta

from nvidia_resiliency_ext.fault_tolerance._ft_rendezvous import create_handler
from torch.distributed import TCPStore
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import (
    C10dRendezvousBackend,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-addr", default="localhost", help="Master node address"
    )
    parser.add_argument(
        "--master-port", type=int, default=29500, help="Master node port"
    )
    parser.add_argument(
        "--node-id", type=int, required=True, help="Node ID (0 to total_nodes-1)"
    )
    parser.add_argument(
        "--total-nodes", type=int, required=True, help="Total number of nodes"
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        help="Minimum number of nodes (defaults to total_nodes)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Maximum number of nodes (defaults to total_nodes)",
    )
    parser.add_argument(
        "--join-timeout", type=int, default=600, help="Join timeout in seconds"
    )
    parser.add_argument(
        "--last-call-timeout", type=int, default=30, help="Last call timeout in seconds"
    )
    args = parser.parse_args()

    # Validate node_id
    if args.node_id < 0 or args.node_id >= args.total_nodes:
        raise ValueError(f"node_id must be between 0 and {args.total_nodes-1}")

    # Set min/max nodes to total_nodes if not specified
    min_nodes = args.min_nodes if args.min_nodes is not None else args.total_nodes
    max_nodes = args.max_nodes if args.max_nodes is not None else args.total_nodes

    # Validate min/max nodes
    if min_nodes > max_nodes:
        raise ValueError("min_nodes cannot be greater than max_nodes")
    if max_nodes > args.total_nodes:
        raise ValueError("max_nodes cannot be greater than total_nodes")
    if min_nodes < 1:
        raise ValueError("min_nodes must be at least 1")

    # Create TCP store
    store = TCPStore(
        host_name=args.master_addr,
        port=args.master_port,
        world_size=max_nodes,
        is_master=(args.node_id == 0),
        timeout=timedelta(seconds=args.join_timeout),
        use_libuv=True,
    )

    # Create a unique run_id
    run_id = "test_rendezvous_1"

    # Create backend
    backend = C10dRendezvousBackend(store=store, run_id=run_id)

    # Create rendezvous parameters
    params = RendezvousParameters(
        backend="c10d",
        endpoint=f"{args.master_addr}:{args.master_port}",
        run_id=run_id,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        join_timeout=str(args.join_timeout),
        last_call_timeout=str(args.last_call_timeout),
        local_addr=f"node_{args.node_id}",
    )

    # Create handler
    handler = create_handler(store, backend, params)

    start_time = time.time()
    log.info(
        f"Node {args.node_id} starting rendezvous at {datetime.fromtimestamp(start_time)}"
    )
    log.info(
        f"Configuration: total_nodes={args.total_nodes}, min_nodes={min_nodes}, max_nodes={max_nodes}"
    )

    try:
        store, rank, world_size = handler.next_rendezvous()
        end_time = time.time()
        duration = end_time - start_time

        log.info(
            f"Node {args.node_id} joined rendezvous as rank {rank} in {duration:.2f} seconds"
        )
        log.info(f"World size: {world_size}")

        # Keep the process alive to maintain the rendezvous
        while True:
            time.sleep(1)

    except Exception as e:
        log.error(f"Node {args.node_id} failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
