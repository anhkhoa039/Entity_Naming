"""
Evaluation metrics for ICC Pipeline
Provides comprehensive evaluation of clustering results against ground truth
"""

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
    silhouette_score,
    adjusted_rand_score
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import pandas as pd


class ClusteringEvaluator:
    """
    Comprehensive evaluator for clustering results
    """
    
    def __init__(self, true_labels, pred_labels, embeddings=None):
        """
        Initialize evaluator
        
        Args:
            true_labels: Ground truth labels (array-like)
            pred_labels: Predicted cluster labels (array-like)
            embeddings: Optional embeddings for silhouette score (array-like)
        """
        self.true_labels = np.array(true_labels)
        self.pred_labels = np.array(pred_labels)
        self.embeddings = embeddings
        
        # Validate inputs
        if len(self.true_labels) != len(self.pred_labels):
            raise ValueError("true_labels and pred_labels must have same length")
    
    def evaluate_all(self):
        """
        Compute all evaluation metrics
        
        Returns:
            dict: Dictionary containing all metrics
        """
        results = {}
        
        # Clustering metrics (no mapping required)
        results['AMI'] = self.adjusted_mutual_information()
        results['ARI'] = self.adjusted_rand_index()
        results['V-Measure'] = self.v_measure()
        results['Homogeneity'] = self.homogeneity()
        results['Completeness'] = self.completeness()
        
        # Mapping-based metrics
        mapping_results = self.mapping_based_metrics()
        results.update(mapping_results)
        
        # Silhouette score (if embeddings provided)
        if self.embeddings is not None:
            results['Silhouette'] = self.silhouette()
        
        # Cluster statistics
        stats = self.cluster_statistics()
        results.update(stats)
        
        return results
    
    def adjusted_mutual_information(self):
        """
        Compute Adjusted Mutual Information (AMI)
        Measures agreement between clusterings, adjusted for chance
        Range: [-1, 1], higher is better
        """
        return adjusted_mutual_info_score(self.true_labels, self.pred_labels)
    
    def adjusted_rand_index(self):
        """
        Compute Adjusted Rand Index (ARI)
        Measures similarity between clusterings, adjusted for chance
        Range: [-1, 1], higher is better
        """
        return adjusted_rand_score(self.true_labels, self.pred_labels)
    
    def v_measure(self):
        """
        Compute V-Measure
        Harmonic mean of homogeneity and completeness
        Range: [0, 1], higher is better
        """
        return v_measure_score(self.true_labels, self.pred_labels)
    
    def homogeneity(self):
        """
        Compute Homogeneity
        Each cluster contains only members of a single class
        Range: [0, 1], higher is better
        """
        return homogeneity_score(self.true_labels, self.pred_labels)
    
    def completeness(self):
        """
        Compute Completeness
        All members of a given class are assigned to the same cluster
        Range: [0, 1], higher is better
        """
        return completeness_score(self.true_labels, self.pred_labels)
    
    def silhouette(self):
        """
        Compute Silhouette Score
        Measures how similar objects are to their own cluster vs other clusters
        Range: [-1, 1], higher is better
        """
        if self.embeddings is None:
            return None
        
        # Only compute if we have enough clusters
        n_clusters = len(np.unique(self.pred_labels))
        if n_clusters < 2 or n_clusters >= len(self.pred_labels):
            return None
        
        return silhouette_score(self.embeddings, self.pred_labels)
    
    def mapping_based_metrics(self):
        """
        Compute precision, recall, F1 using Hungarian algorithm for optimal mapping
        
        Returns:
            dict: Precision, Recall, F1, and mapping information
        """
        # Get unique labels
        true_classes = np.unique(self.true_labels)
        pred_clusters = np.unique(self.pred_labels)
        
        # Create confusion matrix
        n_true = len(true_classes)
        n_pred = len(pred_clusters)
        conf_matrix = np.zeros((n_pred, n_true), dtype=int)
        
        # Build confusion matrix
        for i, pred_cluster in enumerate(pred_clusters):
            for j, true_class in enumerate(true_classes):
                conf_matrix[i, j] = np.sum(
                    (self.pred_labels == pred_cluster) & (self.true_labels == true_class)
                )
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        
        # Calculate metrics based on optimal mapping
        tp = conf_matrix[row_ind, col_ind].sum()
        total = len(self.true_labels)
        
        # Calculate per-class metrics
        fp = total - tp  # Items assigned to wrong cluster
        fn = total - tp  # Items not assigned to correct cluster
        
        precision = tp / total if total > 0 else 0
        recall = tp / total if total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store mapping
        mapping = {}
        for pred_idx, true_idx in zip(row_ind, col_ind):
            mapping[pred_clusters[pred_idx]] = true_classes[true_idx]
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': tp / total if total > 0 else 0,
            'cluster_mapping': mapping,
            'confusion_matrix': conf_matrix
        }
    
    def cluster_statistics(self):
        """
        Compute cluster statistics
        
        Returns:
            dict: Statistics about clusters
        """
        pred_unique, pred_counts = np.unique(self.pred_labels, return_counts=True)
        true_unique, true_counts = np.unique(self.true_labels, return_counts=True)
        
        return {
            'n_predicted_clusters': len(pred_unique),
            'n_true_classes': len(true_unique),
            'avg_cluster_size': np.mean(pred_counts),
            'min_cluster_size': np.min(pred_counts),
            'max_cluster_size': np.max(pred_counts),
            'std_cluster_size': np.std(pred_counts)
        }
    
    def get_cluster_purity(self):
        """
        Calculate purity for each predicted cluster
        
        Returns:
            dict: Cluster ID -> (purity, dominant_class, size)
        """
        cluster_info = {}
        
        for cluster_id in np.unique(self.pred_labels):
            # Get true labels for this cluster
            cluster_mask = self.pred_labels == cluster_id
            cluster_true_labels = self.true_labels[cluster_mask]
            
            # Find dominant class
            unique, counts = np.unique(cluster_true_labels, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            dominant_count = np.max(counts)
            
            # Calculate purity
            purity = dominant_count / len(cluster_true_labels)
            
            cluster_info[cluster_id] = {
                'purity': purity,
                'dominant_class': dominant_class,
                'size': len(cluster_true_labels),
                'dominant_count': dominant_count
            }
        
        return cluster_info
    
    def print_report(self):
        """
        Print a comprehensive evaluation report
        """
        results = self.evaluate_all()
        
        print("=" * 70)
        print("CLUSTERING EVALUATION REPORT")
        print("=" * 70)
        print()
        
        print("CLUSTERING QUALITY METRICS (No Mapping Required)")
        print("-" * 70)
        print(f"  Adjusted Mutual Information (AMI):  {results['AMI']:.4f}")
        print(f"  Adjusted Rand Index (ARI):          {results['ARI']:.4f}")
        print(f"  V-Measure:                           {results['V-Measure']:.4f}")
        print(f"  Homogeneity:                         {results['Homogeneity']:.4f}")
        print(f"  Completeness:                        {results['Completeness']:.4f}")
        if 'Silhouette' in results and results['Silhouette'] is not None:
            print(f"  Silhouette Score:                    {results['Silhouette']:.4f}")
        print()
        
        print("MAPPING-BASED METRICS (Hungarian Algorithm)")
        print("-" * 70)
        print(f"  Precision:                           {results['Precision']:.4f}")
        print(f"  Recall:                              {results['Recall']:.4f}")
        print(f"  F1-Score:                            {results['F1-Score']:.4f}")
        print(f"  Accuracy:                            {results['Accuracy']:.4f}")
        print()
        
        print("CLUSTER STATISTICS")
        print("-" * 70)
        print(f"  Number of Predicted Clusters:        {results['n_predicted_clusters']}")
        print(f"  Number of True Classes:              {results['n_true_classes']}")
        print(f"  Average Cluster Size:                {results['avg_cluster_size']:.2f}")
        print(f"  Min Cluster Size:                    {results['min_cluster_size']}")
        print(f"  Max Cluster Size:                    {results['max_cluster_size']}")
        print(f"  Std Cluster Size:                    {results['std_cluster_size']:.2f}")
        print()
        
        print("=" * 70)
        
        return results
    
    def plot_confusion_matrix(self, figsize=(12, 10), cmap='Blues'):
        """
        Plot confusion matrix with optimal mapping
        
        Args:
            figsize: Figure size
            cmap: Colormap
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return
        
        results = self.mapping_based_metrics()
        conf_matrix = results['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap)
        plt.title('Confusion Matrix (Predicted Clusters vs True Classes)')
        plt.xlabel('True Classes')
        plt.ylabel('Predicted Clusters')
        plt.tight_layout()
        plt.show()


def evaluate_convergence(label_history, true_labels, embeddings_history=None):
    """
    Evaluate convergence across iterations
    
    Args:
        label_history: List of label arrays, one per iteration
        true_labels: Ground truth labels
        embeddings_history: Optional list of embeddings per iteration
    
    Returns:
        DataFrame with metrics per iteration
    """
    results = []
    
    for i, pred_labels in enumerate(label_history):
        evaluator = ClusteringEvaluator(
            true_labels, 
            pred_labels,
            embeddings_history[i] if embeddings_history else None
        )
        
        metrics = evaluator.evaluate_all()
        metrics['iteration'] = i
        results.append(metrics)
    
    df = pd.DataFrame(results)
    return df


def plot_convergence(convergence_df, metrics=None):
    """
    Plot convergence of metrics over iterations
    
    Args:
        convergence_df: DataFrame from evaluate_convergence
        metrics: List of metrics to plot (default: AMI, F1, V-Measure)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    if metrics is None:
        metrics = ['AMI', 'F1-Score', 'V-Measure']
    
    plt.figure(figsize=(10, 6))
    
    for metric in metrics:
        if metric in convergence_df.columns:
            plt.plot(convergence_df['iteration'], convergence_df[metric], 
                    marker='o', label=metric)
    
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Convergence of ICC Refinement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
