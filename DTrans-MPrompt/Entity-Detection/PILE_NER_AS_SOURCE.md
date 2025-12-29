# Using Pile NER as Source Dataset

## Overview

Pile NER is now configured as a **SOURCE dataset** for domain adaptation, similar to how CoNLL2003 was used before. With **11,793 entity types** and **35,730 training examples**, Pile NER provides much richer entity coverage for transfer learning.

## Quick Start

### Train a Single Target Dataset

```bash
./train_pile_ner.sh <target_dataset>
```

**Examples:**
```bash
./train_pile_ner.sh politics    # Train politics with pile_ner as source
./train_pile_ner.sh ai          # Train ai with pile_ner as source
./train_pile_ner.sh music       # Train music with pile_ner as source
```

### Manual Command Format

```bash
sh run_script.sh 0,1 <target_dataset> True False pile_ner Train
```

**Examples:**
```bash
sh run_script.sh 0,1 politics True False pile_ner Train
sh run_script.sh 0,1 ai True False pile_ner Train
sh run_script.sh 0,1 bionlp13cg True False pile_ner Train
sh run_script.sh 0,1 literature True False pile_ner Train
sh run_script.sh 0,1 music True False pile_ner Train
sh run_script.sh 0,1 science True False pile_ner Train
sh run_script.sh 0,1 twitter True False pile_ner Train
```

## Available Target Datasets

- **politics** - Political domain entities
- **ai** - AI/ML domain entities
- **bionlp13cg** - Biomedical entities
- **literature** - Literary entities
- **music** - Music domain entities
- **science** - Scientific entities
- **twitter** - Social media entities

## Why Use Pile NER as Source?

### Comparison: CoNLL2003 vs Pile NER

| Feature | CoNLL2003 | Pile NER |
|---------|-----------|----------|
| Entity Types | 4 | **11,793** |
| Training Examples | ~14K | **35,730** |
| Domain Coverage | News only | **Multi-domain** |
| Entity Diversity | Low | **Very High** |

### Benefits

1. **Massive Entity Coverage**: 11,793 entity types vs CoNLL2003's 4
2. **Multi-Domain**: Covers programming, science, locations, organizations, etc.
3. **Better Transfer**: Rich entity knowledge transfers better to target domains
4. **More Training Data**: 2.5x more training examples than CoNLL2003

## Training Workflow

### 1. Train on Target Dataset

```bash
./train_pile_ner.sh politics
```

This will:
- Use Pile NER (11,793 types) as source
- Train on politics dataset as target
- Save model to `ptms/politics/checkpoint-best/`

### 2. Evaluate on Dev Set

```bash
sh run_script.sh 0,1 politics False True pile_ner dev
```

### 3. Evaluate on Test Set

```bash
sh run_script.sh 0,1 politics False True pile_ner test
```

## Output Structure

After training, you'll have:

```
ptms/
├── politics/
│   ├── checkpoint-best/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── log.txt
│   ├── dev_pred_spans.json
│   └── test_pred_spans.json
├── ai/
│   └── ...
└── music/
    └── ...
```

## Training All Datasets

To train all target datasets with Pile NER as source:

```bash
./train_with_pile_ner_source.sh
```

This will sequentially train:
- politics
- ai
- bionlp13cg
- literature
- music
- science
- twitter

## Expected Performance

Using Pile NER as source should provide **better performance** than CoNLL2003 because:

1. **Better Entity Coverage**: Pile NER's 11,793 types likely overlap more with target domains
2. **Diverse Training**: Multi-domain source helps generalization
3. **More Examples**: Larger source dataset provides better feature learning

## Comparison Commands

### Old Approach (CoNLL2003 as source):
```bash
sh run_script.sh 0,1 politics True False conll2003 Train
```

### New Approach (Pile NER as source):
```bash
sh run_script.sh 0,1 politics True False pile_ner Train
```

## Troubleshooting

### "No target dataset specified"
Make sure to provide the target dataset name:
```bash
./train_pile_ner.sh politics  # ✓ Correct
./train_pile_ner.sh           # ✗ Wrong - missing target
```

### Out of Memory
Reduce batch size in `run_script.sh`:
```bash
TRAIN_BATCH=16        # Reduce from 32
TRAIN_BATCH_SRC=16    # Reduce from 32
```

### Different GPUs
Modify the GPU IDs:
```bash
sh run_script.sh 0 politics True False pile_ner Train     # Single GPU
sh run_script.sh 2,3 politics True False pile_ner Train   # GPUs 2,3
```

## Next Steps

1. **Train**: `./train_pile_ner.sh politics`
2. **Evaluate**: Check dev/test performance
3. **Compare**: Compare with CoNLL2003 baseline results
4. **Analyze**: Review which entity types transfer best

## Summary

✅ **Pile NER is now the source dataset** (like CoNLL2003 was before)  
✅ **Target datasets**: politics, ai, music, science, etc.  
✅ **Command**: `./train_pile_ner.sh <target_dataset>`  
✅ **Better coverage**: 11,793 entity types vs 4  

Start training with:
```bash
./train_pile_ner.sh politics
```
