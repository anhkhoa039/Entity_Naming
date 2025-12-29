# Semantic Naming Improvements

## Issues Identified in Original Results

After analyzing `naming_results_8b.json`, several issues were identified:

1. **Duplicate Names**: Multiple clusters share the same name (e.g., "Election" appears 8 times, "Political Parties" appears 10+ times)
2. **Too Generic**: Names like "Election", "Party", "Politician" are too broad and not descriptive
3. **Inconsistent Specificity**: Some names are overly specific ("Political parties in Western Canada and Australia") while others are too generic
4. **Noisy Boundary Examples**: Boundary examples sometimes include items that don't belong (e.g., "West Aberdeenshire" in an election cluster), confusing the model
5. **Incorrect Names**: Some names don't match the examples well (e.g., "Last Name" for a cluster of politicians)

## Improvements Implemented

### 1. **Enhanced Prompt Template**
- Added few-shot examples showing desired output format
- Clear guidelines on specificity (2-5 words, avoid generic terms)
- Better structured prompt with explicit instructions

### 2. **Boundary Filtering** (`filter_noisy_boundaries`)
- Filters out boundary examples that are too dissimilar from centroids
- Uses keyword overlap as a heuristic (30% threshold)
- Limits to top 3 filtered boundaries to avoid confusion
- Prevents noisy boundaries from misleading the LLM

### 3. **Name Validation** (`validate_name`)
- Rejects overly generic terms ("entity", "thing", "item")
- Rejects names that are too long (likely include explanations)
- Proper capitalization (Title case for multi-word, sentence case for single word)
- Fallback mechanism if validation fails

### 4. **Duplicate Resolution** (`resolve_duplicates`)
- Automatically detects duplicate names across clusters
- Attempts to differentiate duplicates by adding location/context prefixes
- Uses centroid analysis to find distinguishing features (e.g., "US", "UK", "Canadian")
- Example: Multiple "Election" clusters → "US Elections", "UK Elections", "Canadian Elections"

### 5. **Improved Prompt Generation**
- Limits centroids to 8 examples for clarity
- Conditionally includes boundaries only if they pass filtering
- Better formatting and structure

## Usage

The improved functions are backward compatible. New features are enabled by default:

```python
from icc.naming import build_prompts, call_ollama

# Build prompts with filtered boundaries (default)
prompts = build_prompts(examples, use_filtered_boundaries=True)

# Call Ollama with duplicate resolution (default)
results = call_ollama(
    prompts, 
    model="llama3.2:3b",
    examples=examples,
    temperature=0.1,  # Low temperature for deterministic output
    resolve_duplicate_names=True  # Auto-resolve duplicates
)
```

## Expected Improvements

1. **Reduced Duplicates**: Automatic resolution should eliminate most duplicate names
2. **Better Specificity**: Few-shot examples guide the model toward appropriate specificity
3. **Cleaner Output**: Boundary filtering removes confusing examples
4. **More Accurate Names**: Validation ensures names are appropriate and properly formatted
5. **Better Differentiation**: Duplicate resolution adds context to distinguish similar clusters

## Future Enhancements (Optional)

1. **Multi-Candidate Generation**: Generate 3-5 candidate names and select the best using a scoring mechanism
2. **Semantic Similarity Validation**: Use embeddings to verify names match cluster content
3. **Interactive Refinement**: Allow manual review and correction of generated names
4. **Domain-Specific Templates**: Custom prompt templates for different domains (politics, science, etc.)
5. **Confidence Scoring**: Assign confidence scores to generated names based on consistency

