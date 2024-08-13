#!/bin/bash

for i in {0..3}; do
    RANK=$i WORLD_SIZE=4 ./all_reduce_nccl_example &
done

wait
