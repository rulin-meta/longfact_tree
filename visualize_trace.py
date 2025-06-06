import torch
from pathlib import Path
import json
from typing import List, Dict, Any
import argparse
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table

console = Console()

def clean_sentence(sentence: str) -> str:
    """Clean up sentence by removing role prefixes."""
    # Remove "assistant:" prefix if present
    if sentence.lower().startswith("assistant:"):
        sentence = sentence[len("assistant:"):].strip()
    # Remove "user:" prefix if present
    if sentence.lower().startswith("user:"):
        sentence = sentence[len("user:"):].strip()
    return sentence

def load_trace(trace_path: str) -> Dict[str, Any]:
    """Load a trace file."""
    return torch.load(trace_path)

def visualize_tree(trace_data: Dict[str, Any], show_attempts: bool = True):
    """Visualize the generation tree with all attempts."""
    # Create main tree
    tree = Tree("[bold blue]Generation Tree[/bold blue]")
    
    # Add prompt
    tree.add(Panel(f"[bold green]Prompt:[/bold green] {trace_data['prompt']}", style="green"))
    
    # Add metadata
    meta_table = Table(show_header=True, header_style="bold magenta")
    meta_table.add_column("Metric")
    meta_table.add_column("Value")
    
    for key, value in trace_data['generation_metadata'].items():
        meta_table.add_row(str(key), str(value))
    
    tree.add(Panel(meta_table, title="Generation Metadata", style="magenta"))
    
    # Add all paths
    paths = trace_data['paths']
    scores = trace_data['scores']
    
    # Find the best path (highest score)
    best_idx = scores.index(max(scores))
    best_path = paths[best_idx]
    best_score = scores[best_idx]
    
    # Clean up sentences in best path
    cleaned_best_path = [clean_sentence(sentence) for sentence in best_path]
    
    # Add best path in a highlighted panel
    best_path_text = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(cleaned_best_path)])
    tree.add(Panel(
        f"[bold green]Best Path (Score: {best_score:.2f})[/bold green]\n\n{best_path_text}",
        title="Highest Scoring Path",
        style="green"
    ))
    
    # Add all paths in tree format
    for i, (path, score) in enumerate(zip(paths, scores)):
        path_tree = tree.add(f"[yellow]Path {i+1}[/yellow] (score: {score:.2f})")
        for j, sentence in enumerate(path):
            cleaned_sentence = clean_sentence(sentence)
            path_tree.add(f"{j+1}. {cleaned_sentence}")
    
    # Add all attempts if requested
    if show_attempts:
        attempts_tree = tree.add("[red]All Generation Attempts[/red]")
        for gen_id, attempts in trace_data['all_attempts'].items():
            attempt_node = attempts_tree.add(f"Generation {gen_id}")
            for sentence, score in attempts:
                cleaned_sentence = clean_sentence(sentence)
                attempt_node.add(f"Score {score:.2f}: {cleaned_sentence}")
    
    # Add final answer panel with concatenated best path
    final_answer = " ".join(cleaned_best_path)
    tree.add(Panel(
        f"[bold cyan]Final Answer (Score: {best_score:.2f})[/bold cyan]\n\n{final_answer}",
        title="Best Path as Single Answer",
        style="cyan"
    ))
    
    # Print the tree
    console.print(tree)

def main():
    parser = argparse.ArgumentParser(description='Visualize generation traces')
    parser.add_argument('trace_path', type=str, help='Path to the trace file')
    parser.add_argument('--no-attempts', action='store_true', help='Hide generation attempts')
    args = parser.parse_args()
    
    # Load and visualize trace
    trace_data = load_trace(args.trace_path)
    visualize_tree(trace_data, not args.no_attempts)

if __name__ == "__main__":
    main() 