#!/bin/bash

# Training target datasets using Pile NER as SOURCE dataset
# Pile NER has 11,793 entity types, making it a rich source for domain adaptation
# 
# Usage examples:
#   Train politics:  sh run_script.sh 0,1 politics True False pile_ner Train
#   Train ai:        sh run_script.sh 0,1 ai True False pile_ner Train
#   Train music:     sh run_script.sh 0,1 music True False pile_ner Train

# Activate conda environment
echo "Activating dtrans environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dtrans

echo "=========================================="
echo "Training with Pile NER as Source Dataset"
echo "=========================================="
echo ""
echo "Pile NER Statistics:"
echo "  • Entity types: 11,793"
echo "  • Training examples: 35,730"
echo "  • Multi-domain coverage"
echo ""

# Check if target dataset is provided
if [ -z "$1" ]; then
    echo "ERROR: No target dataset specified!"
    echo ""
    echo "Usage: $0 <target_dataset>"
    echo ""
    echo "Available target datasets:"
    echo "  • politics"
    echo "  • ai"
    echo "  • bionlp13cg"
    echo "  • literature"
    echo "  • music"
    echo "  • science"
    echo "  • twitter"
    echo ""
    echo "Example:"
    echo "  $0 politics"
    echo ""
    exit 1
fi

TARGET_DATASET=$1

echo "Target Dataset: $TARGET_DATASET"
echo "Source Dataset: pile_ner"
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Train target dataset with Pile NER as source
sh run_script.sh 0,1 "$TARGET_DATASET" True False pile_ner Train

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: ptms/${TARGET_DATASET}_from_pile_ner/checkpoint-best/"
echo ""
echo "To run inference on dev set:"
echo "  sh run_script.sh 0,1 $TARGET_DATASET False True pile_ner dev"
echo ""
echo "To run inference on test set:"
echo "  sh run_script.sh 0,1 $TARGET_DATASET False True pile_ner test"
echo ""
echo "Note: Models are saved to separate directories based on source dataset:"
echo "  • ptms/${TARGET_DATASET}_from_pile_ner/     (this model)"
echo "  • ptms/${TARGET_DATASET}_from_conll2003/    (if trained with CoNLL2003)"
echo ""

