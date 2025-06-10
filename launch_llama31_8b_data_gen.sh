
conda activate longfact_tree
cd /checkpoint/comem/xilun/experiments/longfact_tree
python gen_data.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --max_depth 10 \
    --max_resample_attempts 5 \
    --max_examples 1 \
    --output_dir generation_traces \
    --temperature 0.7 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096