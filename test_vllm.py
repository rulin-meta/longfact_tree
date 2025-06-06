import os
import torch
from vllm import LLM, SamplingParams

def main():
    print("Setting environment variables...")
    os.environ["TORCH_MULTIPROCESSING_START_METHOD"] = "spawn"
    
    print("\nChecking CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    print("\nInitializing vLLM model...")
    model = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True
    )
    print("vLLM model initialized successfully!")
    
    # Test a simple generation
    print("\nTesting generation...")
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
    )
    
    outputs = model.generate(
        prompts=["Tell me a short fact about the sun."],
        sampling_params=sampling_params,
    )
    
    print("\nGenerated output:")
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main() 