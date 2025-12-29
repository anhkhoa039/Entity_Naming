"""
Semantic naming utilities (Module IV).

Given cluster examples (centroids/boundaries), build LLM-ready prompts.
"""

import json
import subprocess
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import requests


# Improved prompt template with few-shot examples and clearer instructions
PROMPT_TEMPLATE = """You are an expert taxonomist. Your task is to identify the most specific semantic category name for a set of entities.

Guidelines:
- Use a specific, descriptive category name (2-5 words typically)
- Avoid overly generic terms like "Entity", "Thing", "Item"
- Avoid overly specific terms that include location/time unless necessary
- The name should be a noun phrase that clearly describes what all examples have in common

Examples:
Core: ["Barack Obama", "George W. Bush", "Bill Clinton"] → "US Presidents"
Core: ["World War I", "World War II", "Korean War"] → "Wars"
Core: ["New York", "California", "Texas"] → "US States"
Core: ["Republican Party", "Democratic Party", "Green Party"] → "Political Parties"

Now, provide a category name for:
Core Examples: {centroids}
{boundaries_section}

Return ONLY the category name (2-5 words), no quotes, no explanation, no extra text."""


def filter_noisy_boundaries(centroids: List[str], boundaries: List[str], similarity_threshold: float = 0.3) -> List[str]:
    """
    Filter out boundary examples that are too dissimilar from centroids.
    Uses simple keyword overlap as a heuristic.
    """
    if not boundaries:
        return []
    
    # Extract key words from centroids (simple approach)
    centroid_words = set()
    for c in centroids:
        words = re.findall(r'\b\w+\b', c.lower())
        centroid_words.update(words)
    
    filtered = []
    for b in boundaries:
        boundary_words = set(re.findall(r'\b\w+\b', b.lower()))
        if boundary_words:
            overlap = len(centroid_words & boundary_words) / len(boundary_words)
            if overlap >= similarity_threshold:
                filtered.append(b)
    
    return filtered[:3]  # Limit to top 3 filtered boundaries


def generate_prompt(centroids: List[str], boundaries: List[str], use_filtered_boundaries: bool = True) -> str:
    """
    Generate an improved prompt with filtered boundaries.
    """
    centroids_str = ", ".join(centroids[:8])  # Limit to 8 for clarity
    
    if use_filtered_boundaries:
        filtered_boundaries = filter_noisy_boundaries(centroids, boundaries)
        if filtered_boundaries:
            boundaries_section = f"Edge Examples (less typical): {', '.join(filtered_boundaries)}"
        else:
            boundaries_section = ""
    else:
        if boundaries:
            boundaries_section = f"Edge Examples: {', '.join(boundaries[:3])}"
        else:
            boundaries_section = ""
    
    return PROMPT_TEMPLATE.format(
        centroids=centroids_str,
        boundaries_section=boundaries_section
    )


def build_prompts(examples: Dict[int, Dict[str, List[str]]], use_filtered_boundaries: bool = True) -> Dict[int, str]:
    prompts = {}
    for cid, ex in examples.items():
        prompts[int(cid)] = generate_prompt(
            ex.get("centroids", []), 
            ex.get("boundaries", []),
            use_filtered_boundaries=use_filtered_boundaries
        )
    return prompts


def load_examples(path: str) -> Dict[int, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # keys may be str; normalize to int
    normalized = {}
    for k, v in data.items() if isinstance(data, dict) else enumerate(data):
        try:
            cid = int(k)
        except Exception:
            cid = k
        normalized[cid] = v
    return normalized


def save_prompts(prompts: Dict[int, str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def normalize_name(name: str) -> str:
    """Normalize a name for comparison (lowercase, remove extra spaces)."""
    if not name:
        return ""
    name = name.strip()
    # Remove quotes if present
    name = re.sub(r'^["\']|["\']$', '', name)
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name)
    return name.lower()


def validate_name(name: str, centroids: List[str]) -> Tuple[bool, str]:
    """
    Validate and potentially improve a generated name.
    Returns (is_valid, improved_name).
    """
    name = normalize_name(name)
    
    # Reject if too generic
    generic_terms = {"entity", "thing", "item", "object", "category", "type", "class"}
    if name in generic_terms or len(name) < 2:
        return False, ""
    
    # Reject if too long (likely includes explanation)
    if len(name.split()) > 8:
        return False, ""
    
    # Capitalize properly for output
    words = name.split()
    if len(words) > 0:
        # Title case for multi-word, sentence case for single word
        if len(words) == 1:
            improved = words[0].capitalize()
        else:
            improved = " ".join(w.capitalize() for w in words)
        return True, improved
    
    return False, ""


def resolve_duplicates(results: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Resolve duplicate names by making them more specific.
    """
    # Count name occurrences
    name_counts = Counter()
    for cid, data in results.items():
        name = normalize_name(data.get("name", ""))
        if name:
            name_counts[name] += 1
    
    # Find duplicates
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    
    if not duplicates:
        return results
    
    # For each duplicate, try to make names more specific
    for dup_name, count in duplicates.items():
        clusters_with_name = [
            cid for cid, data in results.items()
            if normalize_name(data.get("name", "")) == dup_name
        ]
        
        # Try to differentiate based on centroids
        for idx, cid in enumerate(clusters_with_name):
            centroids = results[cid].get("centroids", [])
            if not centroids:
                continue
            
            # Extract distinguishing features
            words = []
            for cent in centroids[:3]:
                # Look for location/time/type indicators
                if any(x in cent.lower() for x in ["united states", "us", "american", "usa"]):
                    words.append("US")
                elif any(x in cent.lower() for x in ["united kingdom", "uk", "british"]):
                    words.append("UK")
                elif any(x in cent.lower() for x in ["canada", "canadian"]):
                    words.append("Canadian")
                elif any(x in cent.lower() for x in ["australia", "australian"]):
                    words.append("Australian")
            
            if words:
                # Add distinguishing prefix
                original = results[cid]["name"]
                prefix = words[0]  # Use most common
                new_name = f"{prefix} {original}"
                results[cid]["name"] = new_name
    
    return results


def call_ollama(
    prompts: Dict[int, str], 
    model: str, 
    examples: Dict[int, Dict[str, List[str]]] = None, 
    temperature: float = 0.1,
    resolve_duplicate_names: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Run prompts through a local Ollama model with improved naming.
    Returns {cluster_id: {"centroids": List[str], "boundaries": List[str], "name": str}}
    
    Args:
        prompts: Dictionary of cluster_id -> prompt text
        model: Ollama model name
        examples: Dictionary of cluster examples (for extracting centroids/boundaries)
        temperature: Sampling temperature (default 0.1 for deterministic output)
        resolve_duplicate_names: If True, automatically resolve duplicate names
    """
    results = {}
    for cid, prompt in prompts.items():
        try:
            # Use Ollama API with temperature parameter
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            output = response.json().get("response", "").strip()
            # Clean ANSI sequences and pick first non-empty line
            cleaned_lines = [
                line for line in output.splitlines()
                if line.strip() and not line.startswith("\x1b")
            ]
            raw_name = cleaned_lines[0] if cleaned_lines else ""
        except (requests.RequestException, KeyError, ValueError) as e:
            raw_name = ""
        
        # Extract centroids and boundaries from examples if available
        if examples and cid in examples:
            centroids = examples[cid].get("centroids", [])
            boundaries = examples[cid].get("boundaries", [])
        else:
            centroids = []
            boundaries = []
        
        # Validate and clean the name
        is_valid, cleaned_name = validate_name(raw_name, centroids)
        if not is_valid:
            # Fallback: generate a simple name from centroids
            if centroids:
                # Use first word of first centroid as fallback
                first_word = centroids[0].split()[0] if centroids[0] else "Unknown"
                cleaned_name = f"{first_word} Category"
            else:
                cleaned_name = "Unknown Category"
        
        results[int(cid)] = {
            "centroids": centroids,
            "boundaries": boundaries,
            "name": cleaned_name
        }
    
    # Resolve duplicates if requested
    if resolve_duplicate_names:
        results = resolve_duplicates(results)
    
    return results


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--examples_path", required=True, help="Path to cluster_examples.json")
    ap.add_argument("--output_path", required=True, help="Where to write naming_prompts.json")
    ap.add_argument("--ollama_model", type=str, default=None, help="Optional: run prompts through Ollama model")
    args = ap.parse_args()

    examples = load_examples(args.examples_path)
    prompts = build_prompts(examples)
    save_prompts(prompts, args.output_path)
    print(f"Saved {len(prompts)} prompts to {args.output_path}")

    if args.ollama_model:
        results = call_ollama(prompts, args.ollama_model, examples)
        result_path = args.output_path.replace(".json", "_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved Ollama naming results to {result_path}")


if __name__ == "__main__":
    main()
