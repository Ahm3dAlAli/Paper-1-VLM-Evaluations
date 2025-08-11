"""
Enhanced Metrics for Open-Set Recognition (OSR) Evaluation
Includes OSCR, AUROC, and comprehensive open-set metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.integrate import trapz
import pandas as pd
from typing import Dict, List, Tuple, Optional
import seaborn as sns


class OpenSetMetrics:
    """Comprehensive metrics for open-set recognition evaluation."""
    
    @staticmethod
    def calculate_oscr(known_scores: np.ndarray, unknown_scores: np.ndarray, 
                      known_labels: np.ndarray, predicted_labels: np.ndarray,
                      num_classes: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate Open Set Classification Rate (OSCR) curve.
        
        OSCR measures the trade-off between correct classification rate (CCR) on known classes
        and false positive rate (FPR) on unknown samples.
        
        Args:
            known_scores: Confidence scores for known samples
            unknown_scores: Confidence scores for unknown samples
            known_labels: True labels for known samples
            predicted_labels: Predicted labels for known samples
            num_classes: Number of known classes
            
        Returns:
            ccr_values: Correct classification rates at different thresholds
            fpr_values: False positive rates at different thresholds
            oscr_auc: Area under the OSCR curve
        """
        # Combine all scores and create threshold range
        all_scores = np.concatenate([known_scores, unknown_scores])
        thresholds = np.unique(all_scores)
        thresholds = np.sort(thresholds)[::-1]  # Descending order
        
        ccr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            # Known samples: accepted if score >= threshold AND correctly classified
            known_accepted = known_scores >= threshold
            known_correct = predicted_labels == known_labels
            known_correct_accepted = known_accepted & known_correct
            
            # CCR: fraction of known samples correctly classified and accepted
            ccr = np.sum(known_correct_accepted) / len(known_labels) if len(known_labels) > 0 else 0
            
            # FPR: fraction of unknown samples incorrectly accepted as known
            unknown_accepted = unknown_scores >= threshold
            fpr = np.sum(unknown_accepted) / len(unknown_scores) if len(unknown_scores) > 0 else 0
            
            ccr_values.append(ccr)
            fpr_values.append(fpr)
        
        # Convert to numpy arrays
        ccr_values = np.array(ccr_values)
        fpr_values = np.array(fpr_values)
        
        # Sort by FPR for proper curve
        sort_idx = np.argsort(fpr_values)
        fpr_values = fpr_values[sort_idx]
        ccr_values = ccr_values[sort_idx]
        
        # Calculate area under curve using trapezoidal rule
        oscr_auc = trapz(ccr_values, fpr_values)
        
        return ccr_values, fpr_values, oscr_auc
    
    @staticmethod
    def calculate_open_set_f1(tp: int, fp: int, tn: int, fn: int, 
                             unknown_as_positive: bool = False) -> Dict[str, float]:
        """
        Calculate F1 score variants for open-set scenarios.
        
        Args:
            tp: True positives (known correctly classified as known)
            fp: False positives (unknown incorrectly classified as known)
            tn: True negatives (unknown correctly rejected)
            fn: False negatives (known incorrectly rejected)
            unknown_as_positive: Whether to treat unknown detection as positive class
            
        Returns:
            Dictionary with various F1 scores and related metrics
        """
        # Standard metrics
        precision_known = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_known = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_known = 2 * (precision_known * recall_known) / (precision_known + recall_known) \
                   if (precision_known + recall_known) > 0 else 0
        
        # Unknown detection metrics
        precision_unknown = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_unknown = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_unknown = 2 * (precision_unknown * recall_unknown) / (precision_unknown + recall_unknown) \
                     if (precision_unknown + recall_unknown) > 0 else 0
        
        # Macro F1 (average of known and unknown)
        macro_f1 = (f1_known + f1_unknown) / 2
        
        # Open-set F1 (harmonic mean of known classification and unknown detection)
        open_set_f1 = 2 * (f1_known * f1_unknown) / (f1_known + f1_unknown) \
                      if (f1_known + f1_unknown) > 0 else 0
        
        return {
            'precision_known': precision_known,
            'recall_known': recall_known,
            'f1_known': f1_known,
            'precision_unknown': precision_unknown,
            'recall_unknown': recall_unknown,
            'f1_unknown': f1_unknown,
            'macro_f1': macro_f1,
            'open_set_f1': open_set_f1,
            'accuracy': (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        }
    
    @staticmethod
    def calculate_wilderness_risk(known_scores: np.ndarray, unknown_scores: np.ndarray,
                                 threshold: float) -> float:
        """
        Calculate Wilderness Risk - the risk of misclassifying unknown samples.
        
        Lower wilderness risk indicates better open-set performance.
        
        Args:
            known_scores: Confidence scores for known samples
            unknown_scores: Confidence scores for unknown samples
            threshold: Decision threshold
            
        Returns:
            Wilderness risk value
        """
        # Fraction of unknown samples that would be misclassified as known
        unknown_misclassified = np.sum(unknown_scores >= threshold)
        total_unknown = len(unknown_scores)
        
        # Wilderness risk
        risk = unknown_misclassified / total_unknown if total_unknown > 0 else 0
        
        return risk
    
    @staticmethod
    def calculate_open_space_risk(embeddings_known: np.ndarray, 
                                 embeddings_unknown: np.ndarray,
                                 margin: float = 0.1) -> float:
        """
        Calculate Open Space Risk - measures how much of the feature space
        is incorrectly covered by the known class decision boundaries.
        
        Args:
            embeddings_known: Embeddings of known samples
            embeddings_unknown: Embeddings of unknown samples
            margin: Margin for decision boundary
            
        Returns:
            Open space risk value
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Fit KNN on known embeddings
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(embeddings_known)
        
        # Get distances for unknown samples
        distances, _ = knn.kneighbors(embeddings_unknown)
        mean_distances = distances.mean(axis=1)
        
        # Calculate risk as fraction of unknown samples within margin
        within_margin = np.sum(mean_distances <= margin)
        open_space_risk = within_margin / len(embeddings_unknown) if len(embeddings_unknown) > 0 else 0
        
        return open_space_risk
    
    @staticmethod
    def plot_oscr_curve(ccr_values: np.ndarray, fpr_values: np.ndarray, 
                       oscr_auc: float, title: str = "OSCR Curve",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Open Set Classification Rate (OSCR) curve.
        
        Args:
            ccr_values: Correct classification rates
            fpr_values: False positive rates
            oscr_auc: Area under OSCR curve
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot OSCR curve
        ax.plot(fpr_values, ccr_values, 'b-', linewidth=2, 
                label=f'OSCR (AUC = {oscr_auc:.3f})')
        
        # Plot baseline (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Baseline')
        
        # Styling
        ax.set_xlabel('False Positive Rate (FPR) on Unknown', fontsize=12)
        ax.set_ylabel('Correct Classification Rate (CCR) on Known', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add area shading
        ax.fill_between(fpr_values, ccr_values, alpha=0.2, color='blue')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_threshold_analysis(known_scores: np.ndarray, unknown_scores: np.ndarray,
                               title: str = "Score Distribution Analysis",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot score distributions for known and unknown samples with optimal threshold.
        
        Args:
            known_scores: Confidence scores for known samples
            unknown_scores: Confidence scores for unknown samples
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure with optimal threshold
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Histograms
        ax1.hist(known_scores, bins=50, alpha=0.6, label='Known', color='blue', density=True)
        ax1.hist(unknown_scores, bins=50, alpha=0.6, label='Unknown', color='red', density=True)
        
        # Find optimal threshold (maximize difference between known and unknown)
        all_scores = np.concatenate([known_scores, unknown_scores])
        thresholds = np.linspace(all_scores.min(), all_scores.max(), 100)
        
        separations = []
        for thresh in thresholds:
            known_acc = np.mean(known_scores >= thresh)
            unknown_rej = np.mean(unknown_scores < thresh)
            separation = (known_acc + unknown_rej) / 2
            separations.append(separation)
        
        optimal_idx = np.argmax(separations)
        optimal_threshold = thresholds[optimal_idx]
        
        ax1.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Optimal Threshold = {optimal_threshold:.3f}')
        ax1.set_xlabel('Confidence Score', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Score Distributions for Known vs Unknown Samples', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance metrics vs threshold
        precisions = []
        recalls = []
        f1s = []
        
        for thresh in thresholds:
            tp = np.sum(known_scores >= thresh)
            fn = np.sum(known_scores < thresh)
            fp = np.sum(unknown_scores >= thresh)
            tn = np.sum(unknown_scores < thresh)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        ax2.plot(thresholds, precisions, label='Precision', linewidth=2)
        ax2.plot(thresholds, recalls, label='Recall', linewidth=2)
        ax2.plot(thresholds, f1s, label='F1-Score', linewidth=2)
        ax2.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2)
        ax2.set_xlabel('Threshold', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Performance Metrics vs Threshold', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, optimal_threshold


class ComprehensiveOpenSetEvaluator:
    """Complete open-set evaluation framework with all metrics."""
    
    def __init__(self):
        self.metrics = OpenSetMetrics()
        self.results = {}
    
    def evaluate(self, known_embeddings: np.ndarray, unknown_embeddings: np.ndarray,
                known_labels: np.ndarray, known_predictions: np.ndarray,
                known_scores: np.ndarray, unknown_scores: np.ndarray,
                model_name: str = "Model") -> Dict:
        """
        Perform comprehensive open-set evaluation.
        
        Args:
            known_embeddings: Embeddings for known samples
            unknown_embeddings: Embeddings for unknown samples
            known_labels: True labels for known samples
            known_predictions: Predicted labels for known samples
            known_scores: Confidence scores for known samples
            unknown_scores: Confidence scores for unknown samples
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Calculate OSCR
        ccr, fpr, oscr_auc = self.metrics.calculate_oscr(
            known_scores, unknown_scores, known_labels, known_predictions,
            num_classes=len(np.unique(known_labels))
        )
        
        # Find optimal threshold
        _, optimal_threshold = self.metrics.plot_threshold_analysis(
            known_scores, unknown_scores
        )
        
        # Calculate confusion matrix elements at optimal threshold
        tp = np.sum((known_scores >= optimal_threshold) & (known_predictions == known_labels))
        fn = np.sum((known_scores < optimal_threshold) | (known_predictions != known_labels))
        fp = np.sum(unknown_scores >= optimal_threshold)
        tn = np.sum(unknown_scores < optimal_threshold)
        
        # Calculate F1 scores
        f1_metrics = self.metrics.calculate_open_set_f1(tp, fp, tn, fn)
        
        # Calculate wilderness risk
        wilderness_risk = self.metrics.calculate_wilderness_risk(
            known_scores, unknown_scores, optimal_threshold
        )
        
        # Calculate open space risk
        open_space_risk = self.metrics.calculate_open_space_risk(
            known_embeddings, unknown_embeddings
        )
        
        # Calculate standard AUROC for unknown detection
        y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        y_scores = np.concatenate([known_scores, unknown_scores])
        fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr_roc, tpr_roc)
        
        # Compile results
        results = {
            'model_name': model_name,
            'oscr_auc': oscr_auc,
            'auroc': auroc,
            'optimal_threshold': optimal_threshold,
            'wilderness_risk': wilderness_risk,
            'open_space_risk': open_space_risk,
            **f1_metrics,
            'ccr_curve': ccr,
            'fpr_curve': fpr,
            'confusion_matrix': {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        }
        
        self.results[model_name] = results
        return results
    
    def compare_models(self, save_dir: str = './results') -> pd.DataFrame:
        """
        Compare multiple models and generate comprehensive report.
        
        Args:
            save_dir: Directory to save comparison results
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models evaluated yet")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'OSCR AUC': metrics['oscr_auc'],
                'AUROC': metrics['auroc'],
                'Open-Set F1': metrics['open_set_f1'],
                'Wilderness Risk': metrics['wilderness_risk'],
                'Open Space Risk': metrics['open_space_risk'],
                'Known F1': metrics['f1_known'],
                'Unknown F1': metrics['f1_unknown']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Open-Set Recognition Performance Comparison', fontsize=16)
        
        # Plot different metrics
        metrics_to_plot = [
            ('OSCR AUC', True), ('AUROC', True), ('Open-Set F1', True),
            ('Wilderness Risk', False), ('Open Space Risk', False), ('Known F1', True)
        ]
        
        for idx, (metric, higher_better) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
            bars = ax.bar(df['Model'], df[metric], color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} {"↑" if higher_better else "↓"}')
            ax.set_ylim(0, 1.1 if metric != 'Wilderness Risk' and metric != 'Open Space Risk' else None)
            ax.grid(True, alpha=0.3)
            
            # Rotate x labels
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        
        # Save detailed report
        df.to_csv(f'{save_dir}/model_comparison.csv', index=False)
        
        return df