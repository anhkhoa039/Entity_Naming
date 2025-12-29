# Pile NER to CoNLL2003 Format Conversion

## Overview
This document describes the conversion of the Pile NER dataset from its original format to the CoNLL2003 format.

## Conversion Results

### Files Generated
- **Input**: `pile_ner_train.json` (130 MB)
- **Output**: `pile_ner_train_conll.json` (457 MB)
- **Tag Mapping**: `pile_ner_tag_to_id.json` (2.0 MB)

### Dataset Statistics
- **Total entries converted**: 44,663
- **Entity types**: 11,792 (compared to CoNLL2003's 4 types)
- **Total NER tags**: 23,585 (including B- and I- prefixes for each type, plus O)

## Format Comparison

### Original Pile NER Format
```json
{
  "id": "ner_0",
  "entities": [
    {
      "type": "programming language",
      "sentence_idx": 0,
      "start_word_idx": 9,
      "end_word_idx": 10
    }
  ],
  "sentences": [
    ["Q", ":", "Position", "character", "based", "on", "enemy", "coordinates", "in", "lua", ...]
  ]
}
```

### Converted CoNLL2003 Format
```json
{
  "str_words": ["Q", ":", "Position", "character", "based", "on", "enemy", "coordinates", "in", "lua", ...],
  "tags_ner": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "B-programming_language", ...],
  "tags_esi": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "B", ...],
  "tags_net": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "programming_language", ...],
  "id": "ner_0"
}
```

## Field Descriptions

### Output Fields
- **str_words**: Flattened list of all tokens from all sentences
- **tags_ner**: BIO tags with entity type (e.g., `B-programming_language`, `I-person`, `O`)
- **tags_esi**: BIO tags without type (`B`, `I`, `O`)
- **tags_net**: Entity type for tokens inside entities, `O` for tokens outside entities
- **id**: Original entry ID

## Conversion Process

### Key Transformations
1. **Sentence Flattening**: Multiple sentences are flattened into a single `str_words` list
2. **Entity Type Normalization**: Multi-word entity types (e.g., "programming language") are converted to underscore format (e.g., "programming_language")
3. **BIO Tagging**: Entities are converted to BIO format:
   - `B-{type}`: Beginning of an entity
   - `I-{type}`: Inside an entity (continuation)
   - `O`: Outside any entity

### Usage
To convert additional Pile NER files:
```bash
python utils/convert_pile_to_conll.py \
  --input dataset/pile_ner_train.json \
  --output dataset/pile_ner_train_conll.json
```

## Tag Mapping Structure

The `pile_ner_tag_to_id.json` file contains:
- **ner**: Mapping of BIO tags to IDs (e.g., `"B-person": 0`)
- **span**: Mapping of span tags (`B`, `I`, `O`) to IDs
- **type**: Mapping of entity types to IDs
- **template**: Templates for each entity type (for prompt-based methods)

## Notes
- The Pile NER dataset is significantly more diverse than CoNLL2003 (11,792 vs 4 entity types)
- Entity types include programming concepts, scientific terms, locations, organizations, and many domain-specific categories
- The conversion preserves all original entity annotations and IDs
