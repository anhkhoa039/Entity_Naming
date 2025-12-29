# ICC Pipeline Evaluation

This module provides comprehensive evaluation metrics for the Iterative Cluster-Critic (ICC) pipeline results.

## Features

### 1. Clustering Quality Metrics (No Mapping Required)
- **Adjusted Mutual Information (AMI)**: Measures agreement between clusterings, adjusted for chance
- **Adjusted Rand Index (ARI)**: Measures similarity between clusterings, adjusted for chance
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Homogeneity**: Each cluster contains only members of a single class
- **Completeness**: All members of a given class are assigned to the same cluster
- **Silhouette Score**: Measures how similar objects are to their own cluster vs other clusters

### 2. Mapping-Based Metrics (Hungarian Algorithm)
- **Precision**: Accuracy of cluster assignments after optimal mapping
- **Recall**: Coverage of true classes after optimal mapping
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall accuracy after optimal mapping
- **Cluster Mapping**: Optimal assignment of clusters to true classes

### 3. Cluster Analysis
- **Cluster Purity**: Percentage of dominant class in each cluster
- **Cluster Statistics**: Size distribution, counts, etc.
- **Per-Class Performance**: How well each true entity type is captured

### 4. Visualization
- Confusion matrices
- Purity distributions
- Convergence plots
- Performance charts

## Quick Start

### Basic Usage

```python
from icc.evaluation import ClusteringEvaluator

# Create evaluator with your results
evaluator = ClusteringEvaluator(
    true_labels=ground_truth_labels,
    pred_labels=predicted_cluster_ids,
    embeddings=entity_embeddings  # Optional, for silhouette score
)

# Print comprehensive report
results = evaluator.print_report()

# Get specific metrics
ami = evaluator.adjusted_mutual_information()
f1 = evaluator.mapping_based_metrics()['F1-Score']

# Analyze cluster purity
cluster_purity = evaluator.get_cluster_purity()
```

### Using the Evaluation Notebook

1. Run your ICC pipeline (icc_demo.ipynb)
2. Open evaluation_demo.ipynb
3. Update the TARGET and SPLIT variables
4. Run all cells to get comprehensive evaluation

## Metrics Interpretation

### Clustering Quality Metrics
- **Range**: Most metrics range from 0 to 1 (higher is better)
- **AMI/ARI**: Can be negative, but typically [0, 1]
- **Good scores**: Generally > 0.7 is considered good, > 0.8 is excellent

### Mapping-Based Metrics
- **Precision**: What percentage of cluster assignments are correct?
- **Recall**: What percentage of true entities are correctly clustered?
- **F1-Score**: Balanced measure of precision and recall

### Cluster Purity
- **Range**: [0, 1], higher is better
- **Interpretation**: 
  - 1.0 = Perfect purity (all items in cluster belong to same class)
  - 0.5 = Half the items belong to dominant class
  - Lower values indicate mixed clusters

## Example Output

```
======================================================================
CLUSTERING EVALUATION REPORT
======================================================================

CLUSTERING QUALITY METRICS (No Mapping Required)
----------------------------------------------------------------------
  Adjusted Mutual Information (AMI):  0.7234
  Adjusted Rand Index (ARI):          0.6891
  V-Measure:                           0.7456
  Homogeneity:                         0.7123
  Completeness:                        0.7812
  Silhouette Score:                    0.3421

MAPPING-BASED METRICS (Hungarian Algorithm)
----------------------------------------------------------------------
  Precision:                           0.7654
  Recall:                              0.7654
  F1-Score:                            0.7654
  Accuracy:                            0.7654

CLUSTER STATISTICS
----------------------------------------------------------------------
  Number of Predicted Clusters:        72
  Number of True Classes:              45
  Average Cluster Size:                48.67
  Min Cluster Size:                    3
  Max Cluster Size:                    575
  Std Cluster Size:                    74.51

======================================================================
```

## Convergence Analysis

Track metrics across refinement iterations:

```python
from icc.evaluation import evaluate_convergence, plot_convergence

# Assuming you saved labels at each iteration
label_history = [labels_iter0, labels_iter1, labels_iter2, ...]

# Evaluate convergence
convergence_df = evaluate_convergence(
    label_history=label_history,
    true_labels=ground_truth
)

# Plot convergence
plot_convergence(convergence_df, metrics=['AMI', 'F1-Score', 'V-Measure'])
```

## Files

- `icc/evaluation.py`: Main evaluation module
- `evaluation_demo.ipynb`: Comprehensive demonstration notebook
- `EVALUATION_README.md`: This file

## Requirements

- numpy
- pandas
- scikit-learn
- scipy
- matplotlib (for plotting)
- seaborn (for plotting)

## Tips

1. **Always check cluster purity**: Low purity indicates mixed clusters that may need refinement
2. **Compare AMI across iterations**: Should increase as refinement improves clustering
3. **Examine confusion matrix**: Shows which clusters/classes are being confused
4. **Per-class analysis**: Some entity types may be harder to cluster than others
5. **Silhouette score**: Lower scores may indicate overlapping clusters in embedding space

## Troubleshooting

### "No ground truth labels available"
- Make sure your dataset has `tags_ner` field
- Check that entity extraction is working correctly

### "Metrics are very low"
- This is expected for unsupervised clustering
- Compare against baselines (random clustering, single cluster)
- Focus on improvement across iterations

### "Too many clusters"
- Adjust Dip-means parameters (alpha, min_size)
- Use critic refinement to merge similar clusters

## References

See the main framework document for theoretical background on evaluation metrics.
