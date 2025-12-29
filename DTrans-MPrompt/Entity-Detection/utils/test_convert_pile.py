import json
import tempfile
import pathlib
import subprocess
import sys

from convert_pile_to_conll import convert_dataset

def test_conversion(tmp_path: pathlib.Path):
    # Create a tiny PILE NER example matching the user-provided data
    example = {
        "str_words": ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
        "entities": [
            {"type": "organisation", "sentence_idx": 0, "start_word_idx": 0, "end_word_idx": 1},
            {"type": "misc", "sentence_idx": 0, "start_word_idx": 2, "end_word_idx": 3},
            {"type": "misc", "sentence_idx": 0, "start_word_idx": 6, "end_word_idx": 7},
        ],
        "id": 0,
    }
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump([example], f, ensure_ascii=False, indent=2)
    # Run conversion
    convert_dataset(input_file, output_file)
    # Load result and verify tags
    with open(output_file, "r", encoding="utf-8") as f:
        result = json.load(f)[0]
    assert result["tags_ner"] == ["B-organisation", "O", "B-misc", "O", "O", "O", "B-misc", "O", "O"]
    assert result["tags_esi"] == ["B", "O", "B", "O", "O", "O", "B", "O", "O"]
    assert result["tags_net"] == ["organisation", "O", "misc", "O", "O", "O", "misc", "O", "O"]
