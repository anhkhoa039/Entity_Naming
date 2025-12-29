"""Utility to convert PILE NER JSON format to CONLL2003‑style JSON.

The input file contains entries with:
    - `sentences`: list of lists (each inner list is a sentence with tokens)
    - `entities`: list of dicts with `type`, `sentence_idx`, `start_word_idx`, `end_word_idx`
    - `id`
We produce entries with:
    - `str_words`: flattened list of tokens from all sentences
    - `tags_ner`: BIO tags with entity type (e.g. B‑person, I‑person)
    - `tags_esi`: BIO tags without type (B, I, O)
    - `tags_net`: lower‑case entity type for each token inside an entity, O otherwise
    - `id`: original id
"""
import json
import argparse
from pathlib import Path

def load_pile(path: Path):
    """Load JSONL file (newline-delimited JSON) or single JSON with 'documents' array."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # Try to parse as single JSON object first
    try:
        data = json.loads(content)
        # If it has a 'documents' key, use that
        if isinstance(data, dict) and "documents" in data:
            return data["documents"]
        # If it's already a list, return it
        elif isinstance(data, list):
            return data
        # Otherwise, wrap it in a list
        else:
            return [data]
    except json.JSONDecodeError:
        # Fall back to JSONL parsing
        data = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data

def convert_entry(entry):
    # Flatten sentences into a single list of words
    sentences = entry.get("sentences", [])
    words = []
    for sentence in sentences:
        words.extend(sentence)
    
    length = len(words)
    tags_ner = ["O"] * length
    tags_esi = ["O"] * length
    tags_net = ["O"] * length
    
    # Process entities
    for ent in entry.get("entities", []):
        start = ent["start_word_idx"]
        end = ent["end_word_idx"]  # exclusive
        ent_type = ent["type"].lower().replace(" ", "_")  # Handle multi-word types like "programming language"
        
        # Ensure indices are within bounds
        if start >= length or end > length:
            continue
            
        # BIO tags with type for tags_ner
        tags_ner[start] = f"B-{ent_type}"
        for i in range(start + 1, end):
            tags_ner[i] = f"I-{ent_type}"
        
        # Simple B/I/O for tags_esi
        tags_esi[start] = "B"
        for i in range(start + 1, end):
            tags_esi[i] = "I"
        
        # Net tags (just the type) for tokens inside the span
        for i in range(start, end):
            tags_net[i] = ent_type
    
    return {
        "str_words": words,
        "tags_ner": tags_ner,
        "tags_esi": tags_esi,
        "tags_net": tags_net,
        "id": entry.get("id", "unknown"),
    }

def convert_dataset(pile_path: Path, out_path: Path):
    data = load_pile(pile_path)
    converted = []
    for idx, item in enumerate(data):
        entry = convert_entry(item)
        entry["id"] = idx  # Use integer ID for compatibility with torch.tensor(..., dtype=torch.long)
        converted.append(entry)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(converted)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PILE NER to CONLL2003 JSON")
    parser.add_argument("--input", type=Path, required=True, help="Path to pile_ner_train.json")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    args = parser.parse_args()
    convert_dataset(args.input, args.output)
    print(f"Converted {args.input} → {args.output}")
