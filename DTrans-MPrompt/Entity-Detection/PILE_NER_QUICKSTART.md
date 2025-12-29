# Pile NER Training - Quick Start Guide

## ✅ Setup Complete!

All files are ready for training the Pile NER dataset.

## 📁 Files Created

### Dataset Files (in `dataset/`)
- ✅ `pile_ner_train.json` (365 MB) - 35,730 training examples
- ✅ `pile_ner_dev.json` (45 MB) - 4,466 validation examples
- ✅ `pile_ner_test.json` (46 MB) - 4,467 test examples
- ✅ `pile_ner_tag_to_id.json` (2 MB) - Mappings for 11,793 entity types

### Training Scripts
- ✅ `train_pile_ner.sh` - Main training script (with conda activation)
- ✅ `PILE_NER_TRAINING.md` - Detailed training documentation
- ✅ `dataset/PILE_NER_CONVERSION_README.md` - Dataset conversion documentation

## 🚀 Start Training

### Option 1: Using the Training Script (Recommended)

```bash
./train_pile_ner.sh
```

This script will:
1. Activate the `dtrans` conda environment
2. Start training with CoNLL2003 as source dataset
3. Save checkpoints to `ptms/pile_ner/`
4. Log progress to `ptms/pile_ner/log.txt`

### Option 2: Manual Command

```bash
conda activate dtrans
sh run_script.sh 0,1 pile_ner True False conll2003 Train
```

## 📊 Dataset Statistics

- **Total examples**: 44,663
- **Entity types**: 11,793 (vs CoNLL2003's 4 types)
- **Average entities per example**: ~20
- **Training split**: 80% / 10% dev / 10% test

## 🎯 Training Configuration

- **Model**: BERT-base-cased
- **Learning Rate**: 1e-5
- **Batch Size**: 32 (both source and target)
- **Epochs**: 50
- **GPUs**: 0,1 (configurable)
- **Source Dataset**: CoNLL2003
- **Target Dataset**: Pile NER

## 📈 Monitoring Progress

Training progress can be monitored via:
- **Console output**: Real-time metrics
- **Log file**: `ptms/pile_ner/log.txt`
- **Checkpoints**: Best model saved to `ptms/pile_ner/checkpoint-best/`

## 🧪 Evaluation

After training, evaluate on dev/test sets:

```bash
# Development set
sh run_script.sh 0,1 pile_ner False True conll2003 dev

# Test set
sh run_script.sh 0,1 pile_ner False True conll2003 test
```

## 📚 Documentation

For detailed information, see:
- **Training Guide**: `PILE_NER_TRAINING.md`
- **Dataset Conversion**: `dataset/PILE_NER_CONVERSION_README.md`

## ⚠️ Important Notes

1. **Training Time**: Expect several hours to days depending on hardware
2. **Memory**: Requires ~16GB+ GPU memory (reduce batch size if OOM)
3. **Disk Space**: Ensure sufficient space for checkpoints (~1-2GB)

## 🔧 Troubleshooting

### Out of Memory
Edit `run_script.sh` and reduce batch sizes:
```bash
TRAIN_BATCH=16        # Reduce from 32
TRAIN_BATCH_SRC=16    # Reduce from 32
```

### Different GPUs
Change GPU IDs in the command:
```bash
sh run_script.sh 0 pile_ner True False conll2003 Train  # Single GPU
sh run_script.sh 2,3 pile_ner True False conll2003 Train  # GPUs 2,3
```

## 🎉 Ready to Go!

Everything is set up. Just run:
```bash
./train_pile_ner.sh
```

Good luck with your training! 🚀
