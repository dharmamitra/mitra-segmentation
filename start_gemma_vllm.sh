#!/bin/bash

for i in {0..7}
do
    port=$((3500 + i))
    masterport=$((29610 + i))
    nohup bash -c "CUDA_VISIBLE_DEVICES=$i VLLM_PORT=$masterport vllm serve google/gemma-2-9b-it --max-model-len 4048 --tensor-parallel-size 1 --port $port --api-key \"123\" --disable-frontend-multiprocessing --gpu-memory-utilization 0.8 --max-num-seqs 128 --served-model-name google-demma2" > "log_gpu_$i.out" 2>&1 &
sleep 120
done

echo "All processes started. Check log files for details."
