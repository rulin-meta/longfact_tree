#!/bin/bash
#SBATCH --job-name=xilun_70b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --hint=nomultithread   
#SBATCH --account=comem
#SBATCH --qos=h100_comem_high
#SBATCH --mem=600G
#SBATCH --gres=gpu:8           # Request 1 GPUs per node
#SBATCH --exclusive            # Request exclusive access to the nodes
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --chdir=/checkpoint/comem/rulin/multi-token-pred/longfact_tree
#SBATCH --output=/checkpoint/comem/rulin/multi-token-pred/longfact_tree/log/%x-%A_%a.out
#SBATCH --array=0


set -e

source /home/rulin/miniconda3/bin/activate
conda activate longfact_tree
cd /checkpoint/comem/rulin/multi-token-pred/longfact_tree


export VLLM_WORKER_MULTIPROC_METHOD=spawn

DEPTH=50
RESAMPLE_ATTEMPTS=5
MAX_EXAMPLES=1000

EXP_ID=d${DEPTH}_r${RESAMPLE_ATTEMPTS}

python gen_data.py \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --max_depth ${DEPTH} \
    --max_resample_attempts ${RESAMPLE_ATTEMPTS} \
    --max_examples ${MAX_EXAMPLES} \
    --tensor_parallel_size 8 \
    --output_dir data/longfact_tree_data_llama33_70b_veriscore_${EXP_ID}
