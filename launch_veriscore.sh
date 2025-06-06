cd /checkpoint/comem/xilun/experiments/veriscore/code

# Create local data directory
LOCAL_DATA_DIR="/tmp/veriscore_${USER}/data"
mkdir -p "${LOCAL_DATA_DIR}/cache"

# Copy data if it doesn't exist
if [ ! -f "${LOCAL_DATA_DIR}/cache/search_cache.sqlite" ]; then
    cp -r /checkpoint/comem/xilun/experiments/veriscore/data/* "${LOCAL_DATA_DIR}/"
fi

conda activate /checkpoint/comem/xilun/envs/truthteller_250512

VLLM_WORKER_MULTIPROC_METHOD=spawn python -m veriscore.veriscore_server \
--data_dir "${LOCAL_DATA_DIR}" \
--model_name_extraction vllm_mistral_extract \
--extraction_llm_backend vllm \
--extraction_prompt_format finetuned \
--model_name_verification peft_verify \
--verification_llm_backend transformers \
--verification_prompt_format finetuned


curl -X POST http://10.137.193.1:43389/veriscore -H 'Content-Type: application/json' -d '{"question": "Who is Buffett", "response": "Warren Buffett is an American billionare. He is 90 years old.", "last_sentence_only": true}'