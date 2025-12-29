# Training Pile NER Dataset

This guide explains how to train the Pile NER dataset using the DTrans-MPrompt framework.

## Dataset Preparation ✅

The Pile NER dataset has been successfully converted to CoNLL2003 format with the following files:

### Dataset Files
- **`pile_ner_train.json`** (366 MB) - 35,730 training examples
- **`pile_ner_dev.json`** (46 MB) - 4,466 validation examples  
- **`pile_ner_test.json`** (46 MB) - 4,467 test examples
- **`pile_ner_tag_to_id.json`** (2.0 MB) - Tag mappings for 11,792 entity types

### Dataset Statistics
- **Total examples**: 44,663
- **Entity types**: 11,792 (compared to CoNLL2003's 4 types)
- **Split ratio**: 80% train / 10% dev / 10% test
- **Format**: Same as CoNLL2003 (str_words, tags_ner, tags_esi, tags_net, id)

## Training

### Quick Start

To train the Pile NER model with CoNLL2003 as the source dataset:

```bash
./train_pile_ner.sh
```

Or manually:

```bash
sh run_script.sh 0,1 pile_ner True False conll2003 Train
```

### Command Breakdown

The training command follows this pattern:
```bash
sh run_script.sh <GPU_IDs> <target_dataset> <do_train> <do_test> <source_dataset> <eval_mode>
```

**Parameters:**
- `0,1` - GPU IDs to use (0 and 1)
- `pile_ner` - Target dataset name
- `True` - Enable training
- `False` - Disable testing during training
- `conll2003` - Source dataset for domain adaptation
- `Train` - Evaluation mode (not used during training)

### Training Configuration

The default training parameters (defined in `run_script.sh`):

```bash
Learning Rate:        1e-5
Weight Decay:         1e-4
Epochs:              50
Batch Size (target): 32
Batch Size (source): 32
Eval Batch Size:     32
Warmup Steps:        500
Mean Alpha:          0.995
Consistency Weight:  100.0
Consistency Rampup:  5
Max Sequence Length: 128
```

### Output

Training outputs will be saved to:
```
ptms/pile_ner/
├── checkpoint-best/          # Best model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── log.txt                   # Training logs
```

## Evaluation

### Development Set

To evaluate on the development set:

```bash
sh run_script.sh 0,1 pile_ner False True conll2003 dev
```

This will:
- Load the best checkpoint from `ptms/pile_ner/checkpoint-best/`
- Evaluate on `pile_ner_dev.json`
- Save predictions to `ptms/pile_ner/dev_pred_spans.json`

### Test Set

To evaluate on the test set:

```bash
sh run_script.sh 0,1 pile_ner False True conll2003 test
```

This will:
- Load the best checkpoint from `ptms/pile_ner/checkpoint-best/`
- Evaluate on `pile_ner_test.json`
- Save predictions to `ptms/pile_ner/test_pred_spans.json`

## Model Architecture

The training uses a **multi-level span detection** approach with:

1. **Span Model**: BERT-based model for entity span detection
2. **Momentum Model**: Exponential moving average model for consistency regularization
3. **Multi-task Learning**: 
   - BIO tagging
   - Start-End detection
   - Tie-Break mechanism
4. **Domain Adaptation**: Uses CoNLL2003 as source domain

## Expected Performance

The model will be evaluated using:
- **Precision**: Percentage of predicted entities that are correct
- **Recall**: Percentage of true entities that are found
- **F1 Score**: Harmonic mean of precision and recall

Results will be reported for:
- BIO tagging
- Start-End detection
- Tie-Break mechanism
- Ensemble (best performing method)

## Monitoring Training

Training progress can be monitored via:

1. **Console output**: Real-time loss and metrics
2. **Log file**: `ptms/pile_ner/log.txt`
3. **Checkpoints**: Saved every 100 steps (configurable)

The model automatically evaluates on dev/test sets every 100 steps and saves the best checkpoint.

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors, try:
- Reducing batch size in `run_script.sh`:
  ```bash
  TRAIN_BATCH=16        # Reduce from 32
  TRAIN_BATCH_SRC=16    # Reduce from 32
  ```
- Using fewer GPUs
- Reducing max sequence length

### Slow Training

The Pile NER dataset is large (35K+ training examples with 11K+ entity types). Training may take several hours or days depending on your hardware.

### Different Source Dataset

To use a different source dataset instead of CoNLL2003:

```bash
sh run_script.sh 0,1 pile_ner True False <other_dataset> Train
```

Where `<other_dataset>` can be: `ai`, `bionlp13cg`, `literature`, `music`, `politics`, `science`, or `twitter`.

## Comparison with CoNLL2003

| Aspect | CoNLL2003 | Pile NER |
|--------|-----------|----------|
| Entity Types | 4 | 11,792 |
| Training Examples | ~14K | 35,730 |
| Domain | News | Multi-domain |
| Complexity | Low | Very High |

The Pile NER dataset is significantly more challenging due to its:
- Massive number of entity types
- Multi-domain nature
- Complex entity categories (programming, science, etc.)

## Next Steps

After training:

1. **Evaluate** on dev/test sets
2. **Analyze** predictions in `*_pred_spans.json`
3. **Compare** with CoNLL2003 baseline
4. **Fine-tune** hyperparameters if needed
5. **Export** model for deployment

## References

- Original Pile NER format: See `dataset/PILE_NER_CONVERSION_README.md`
- Conversion script: `utils/convert_pile_to_conll.py`
- Training script: `run_script.py`
