import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import spacy
from spacy.lang.en import English
import random
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from pathlib import Path
from tqdm import tqdm
import argparse



# Llama instruction template
LLAMA_INSTRUCTION_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, factual, and concise assistant. When answering questions, focus on providing accurate, factual information. If you're unsure about something, indicate that clearly.

<|start_header_id|>user<|end_header_id|>
{prompt}

<|start_header_id|>assistant<|end_header_id|>
"""

@dataclass
class LongFactExample:
    """Represents a single example from the LongFact dataset."""
    prompt: str
    metadata: Dict[str, Any] = None

def load_longfact_data(file_path: str) -> List[LongFactExample]:
    """Load examples from the LongFact JSONL file."""
    examples = []
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading LongFact data"):
            data = json.loads(line)
            
            # Create example from prompt
            example = LongFactExample(
                prompt=data['prompt'],  # Using 'prompt' instead of 'question'
                metadata={}  # Empty metadata for now
            )
            examples.append(example)
            
    print(f"\nDataset statistics:")
    print(f"  Total examples loaded: {len(examples)}")
    
    return examples

@dataclass
class GenerationNode:
    """Represents a node in the generation tree."""
    sentence: str
    children: List['GenerationNode']
    parent: Optional['GenerationNode']
    is_verified: bool
    score: float  # verification score
    generation_id: int  # unique ID for this generation path
    attempts: List[Tuple[str, float]]  # List of (sentence, score) for all attempts, including unfactual ones

class FactualGenerationTree:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        verify_fn = None,
        max_resample_attempts: int = 5,
        max_depth: int = 8,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        model = None,  # Accept pre-initialized model
    ):
        # Use pre-initialized model if provided
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Initialize simple English model with sentencizer
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.verify_fn = verify_fn
        self.max_resample_attempts = max_resample_attempts
        self.max_depth = max_depth
        self.temperature = temperature
        self.current_generation_id = 0

    def format_prompt(self, prompt: str, previous_sentences: List[str] = None) -> str:
        """Format the prompt with previous sentences using the model's chat template."""
        messages = [
            {"role": "system", "content": "You are a helpful, factual, and concise assistant. When answering questions, focus on providing accurate, factual information. If you're unsure about something, indicate that clearly."},
            {"role": "user", "content": prompt}
        ]
        
        # Add previous sentences as assistant's responses
        if previous_sentences:
            all_previous_sentences = " ".join(previous_sentences)
            messages.append({"role": "assistant", "content": all_previous_sentences})
        
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
        print("\nDEBUG - Formatted prompt structure:")
        print(formatted)
        print("END DEBUG\n")
        return formatted

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def generate_single_sentence(
        self,
        prompt: str,
        previous_sentences: List[str],
        temperature: Optional[float] = None
    ) -> str:
        """Generate a single sentence given the prompt and previous sentences using vLLM."""
        if temperature is None:
            temperature = self.temperature

        # Format prompt with instruction template
        formatted_prompt = self.format_prompt(prompt, previous_sentences)

        # Configure sampling parameters for Llama
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=256,  # Increased for longer responses
            top_p=0.9,
            top_k=50,
            stop=["</s>", "<|end_of_text|>"],  # Only stop at proper end tokens
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )
        
        # Generate using vLLM
        outputs = self.model.generate(
            prompts=[formatted_prompt],
            sampling_params=sampling_params,
        )
        
        # Get the generated text and clean it
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Remove any assistant role markers that might be in the output
        generated_text = generated_text.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
        
        # Split into sentences and return the first complete one
        sentences = self.split_into_sentences(generated_text)
        if not sentences:
            return ""
            
        # Return the first non-empty sentence
        for sentence in sentences:
            if sentence.strip() and not sentence.strip().lower() == "assistant":
                return sentence.strip()
        return ""

    def verify_sentence(
        self,
        question: str,
        sentence: str,
        previous_sentences: List[str]
    ) -> Tuple[bool, float]:
        """Verify if a sentence is factual."""
        # Combine previous sentences with current sentence for context
        full_answer = " ".join(previous_sentences + [sentence])
        return self.verify_fn(question, full_answer)

    def generate_with_verification(
        self,
        prompt: str,
    ) -> Tuple[GenerationNode, Dict]:
        """
        Generate a factual response with tree-based sentence verification.
        Returns the root node of the generation tree and metadata about the generation process.
        """
        root = GenerationNode(
            sentence="",  # Empty root node
            children=[],
            parent=None,
            is_verified=True,
            score=1.0,
            generation_id=self.current_generation_id,
            attempts=[]
        )
        self.current_generation_id += 1

        current_node = root
        previous_sentences = []
        metadata = {
            "total_resamples": 0,
            "verification_attempts": 0,
            "successful_paths": 0,
            "total_unfactual_attempts": 0,
            "max_depth_reached": False,
            "max_attempts_reached": 0,  # Count of times we hit max attempts
            "average_attempts_per_sentence": 0.0,
            "depth": 0,  # Actual depth reached
            "successful_sentences": 0,  # Number of sentences that passed verification
            "failed_sentences": 0,  # Number of sentences that failed all attempts
        }

        for depth in range(self.max_depth):
            # Try to generate a factual sentence
            best_sentence = None
            best_score = 0.0
            resample_count = 0
            attempts = []  # Track all attempts for this position

            while resample_count < self.max_resample_attempts:
                sentence = self.generate_single_sentence(prompt, previous_sentences)
                score = self.verify_sentence(prompt, sentence, previous_sentences)
                metadata["verification_attempts"] += 1
                
                # Record this attempt
                attempts.append((sentence, score))
                
                if score > best_score:
                    best_sentence = sentence
                    best_score = score
                    if score == 1.0:
                        break  # Found a factual sentence, no need to resample
                
                resample_count += 1
                metadata["total_resamples"] += 1
                metadata["total_unfactual_attempts"] += 1

            if best_sentence is None:
                # Couldn't generate a factual sentence after max attempts
                metadata["failed_sentences"] += 1
                metadata["max_attempts_reached"] += 1
                break

            # Create new node for the verified sentence
            new_node = GenerationNode(
                sentence=best_sentence,
                children=[],
                parent=current_node,
                is_verified=True,
                score=best_score,
                generation_id=self.current_generation_id,
                attempts=attempts
            )
            self.current_generation_id += 1
            
            current_node.children.append(new_node)
            current_node = new_node
            previous_sentences.append(best_sentence)
            metadata["successful_sentences"] += 1
            metadata["depth"] = depth + 1

        # Check if we hit max depth
        if metadata["depth"] == self.max_depth:
            metadata["max_depth_reached"] = True

        # Calculate average attempts per sentence
        if metadata["successful_sentences"] > 0:
            metadata["average_attempts_per_sentence"] = (
                metadata["verification_attempts"] / metadata["successful_sentences"]
            )

        metadata["successful_paths"] = self.count_successful_paths(root)
        return root, metadata

    def count_successful_paths(self, node: GenerationNode) -> int:
        """Count the number of complete generation paths in the tree."""
        if not node.children:
            return 1
        return sum(self.count_successful_paths(child) for child in node.children)

    def get_all_paths(self, node: GenerationNode) -> List[List[str]]:
        """Get all generation paths from the tree."""
        if not node.children:
            return [[node.sentence]] if node.sentence else [[]]
        
        paths = []
        for child in node.children:
            child_paths = self.get_all_paths(child)
            for path in child_paths:
                if node.sentence:  # Skip empty root node
                    paths.append([node.sentence] + path)
                else:
                    paths.append(path)
        return paths

    def save_generation_trace(self, root: GenerationNode, output_path: str):
        """Save the generation tree for masked DPO training, including unfactual attempts."""
        # Convert tree to a format suitable for training
        paths = self.get_all_paths(root)
        
        # Save paths, their verification scores, and all attempts
        trace_data = {
            "paths": paths,
            "scores": [self.get_path_score(path, root) for path in paths],
            "generation_ids": [self.get_path_generation_ids(path, root) for path in paths],
            "all_attempts": self.get_all_attempts(root)  # Include all attempts
        }
        
        torch.save(trace_data, output_path)

    def get_path_score(self, path: List[str], root: GenerationNode) -> float:
        """Get the average verification score for a path."""
        current = root
        scores = []
        
        for sentence in path:
            for child in current.children:
                if child.sentence == sentence:
                    scores.append(child.score)
                    current = child
                    break
        
        return sum(scores) / len(scores) if scores else 0.0

    def get_path_generation_ids(self, path: List[str], root: GenerationNode) -> List[int]:
        """Get the generation IDs for a path."""
        current = root
        ids = []
        
        for sentence in path:
            for child in current.children:
                if child.sentence == sentence:
                    ids.append(child.generation_id)
                    current = child
                    break
        
        return ids

    def get_all_attempts(self, node: GenerationNode) -> Dict[int, List[Tuple[str, float]]]:
        """Get all generation attempts (both factual and unfactual) from the tree."""
        attempts = {}
        
        def collect_attempts(current_node: GenerationNode):
            if current_node.generation_id not in attempts:
                attempts[current_node.generation_id] = current_node.attempts
            for child in current_node.children:
                collect_attempts(child)
        
        collect_attempts(node)
        return attempts

    def print_tree(self, node: GenerationNode, level: int = 0):
        """Print the tree with both factual and unfactual attempts."""
        if node.sentence:
            print("  " * level + f"- {node.sentence} (score: {node.score:.2f})")
            if node.attempts:
                print("  " * (level + 1) + "Attempts:")
                for attempt, score in node.attempts:
                    if attempt != node.sentence:  # Don't repeat the chosen sentence
                        print("  " * (level + 2) + f"× {attempt} (score: {score:.2f})")
        for child in node.children:
            self.print_tree(child, level + 1)

def generate_dataset(
    model_name: str,
    verify_fn,
    input_file: str = "/checkpoint/comem/data/eval/longfact_full/longfact_objects_train.jsonl",
    output_dir: str = "generation_traces",
    max_examples: Optional[int] = None,
    max_resample_attempts: int = 5,
    max_depth: int = 8,
    temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    model = None,  # Accept pre-initialized model
):
    """Generate and save traces for the LongFact dataset."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load examples
    examples = load_longfact_data(input_file)
    if max_examples is not None:
        examples = examples[:max_examples]
    
    # Initialize generator with pre-initialized model
    generator = FactualGenerationTree(
        model_name=model_name,
        verify_fn=verify_fn,
        max_resample_attempts=max_resample_attempts,
        max_depth=max_depth,
        temperature=temperature,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        model=model  # Pass the pre-initialized model
    )
    
    # Process each example
    for i, example in enumerate(tqdm(examples, desc="Generating traces")):
        # Generate response
        root_node, metadata = generator.generate_with_verification(example.prompt)
        
        # Save trace
        trace_data = {
            "prompt": example.prompt,
            "metadata": example.metadata,
            "generation_metadata": metadata,
            "paths": generator.get_all_paths(root_node),
            "scores": [generator.get_path_score(path, root_node) for path in generator.get_all_paths(root_node)],
            "generation_ids": [generator.get_path_generation_ids(path, root_node) for path in generator.get_all_paths(root_node)],
            "all_attempts": generator.get_all_attempts(root_node)
        }
        
        # Save to file
        output_file = output_path / f"trace_{i:04d}.pt"
        torch.save(trace_data, output_file)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"\nProcessed {i + 1} examples")
            print(f"Latest example metadata: {metadata}")
            print(f"Saved to {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate factual response trees using Llama model.')
    
    parser.add_argument('--model_name', 
                       type=str,
                       default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Name of the model to use for generation')
    
    parser.add_argument('--max_depth',
                       type=int,
                       default=8,
                       help='Maximum number of sentences to generate per prompt')
    
    parser.add_argument('--max_resample_attempts',
                       type=int,
                       default=5,
                       help='Maximum attempts to generate a factual sentence')
    
    parser.add_argument('--max_examples',
                       type=int,
                       default=None,
                       help='Maximum number of examples to process (None for all)')
    
    parser.add_argument('--output_dir',
                       type=str,
                       default="generation_traces",
                       help='Directory to save generation traces')
    
    parser.add_argument('--temperature',
                       type=float,
                       default=0.7,
                       help='Sampling temperature for generation')
    
    parser.add_argument('--tensor_parallel_size',
                       type=int,
                       default=1,
                       help='Number of GPUs to use for tensor parallelism')
    
    parser.add_argument('--gpu_memory_utilization',
                       type=float,
                       default=0.9,
                       help='GPU memory utilization fraction')
    
    parser.add_argument('--max_model_len',
                       type=int,
                       default=4096,
                       help='Maximum model context length')
    
    parser.add_argument('--input_file',
                       type=str,
                       default="/checkpoint/comem/data/eval/longfact_full/longfact_objects_train.jsonl",
                       help='Input JSONL file containing prompts')
    
    args = parser.parse_args()
    
    # Initialize vLLM first with explicit settings
    print("\nInitializing vLLM model...")
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True
    )
    print("vLLM model initialized successfully!")

    # Example verification function (replace with actual implementation)
    def dummy_verify_fn(question: str, answer: str) -> Tuple[bool, float]:
        return random.random()

    from veriscore import veriscore
    def veriscore_verify_fn(question: str, answer: str) -> Tuple[bool, float]:
        return veriscore(question, answer)
    
    
    print(f"\nStarting generation with parameters:")
    print(f"  Model: {args.model_name}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Max resample attempts: {args.max_resample_attempts}")
    print(f"  Max examples: {args.max_examples if args.max_examples else 'all'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  Max model length: {args.max_model_len}")
    print(f"  Input file: {args.input_file}")
    print("\nStarting generation...\n")

    # Generate traces for the dataset
    generate_dataset(
        model_name=args.model_name,
        verify_fn=veriscore_verify_fn,
        max_examples=args.max_examples,
        max_resample_attempts=args.max_resample_attempts,
        max_depth=args.max_depth,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        input_file=args.input_file,
        output_dir=args.output_dir,
        model=model  # Pass the initialized model
    )

