"""
Enhanced VLM Pipeline - Core Components
Extends the existing VLM pipeline with multi-modal analysis capabilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import pandas as pd
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class MultiModalEmbeddingExtractor:
    """Fixed embedding extractor that ensures consistent dimensions."""
    
    def __init__(self, vlm_model, processor, device='cuda'):
        self.vlm_model = vlm_model
        self.processor = processor
        self.device = device
    
    def _extract_batch_features(self, batch_images):
        """Extract features for a batch ensuring consistent dimensions."""
        batch_features = {}
        
        with torch.no_grad():
            # Process images
            inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
            
            # Hook into different layers
            activations = {}
            hooks = []
            
            def get_activation(name):
                def hook(model, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    activations[name] = output.detach()
                return hook
            
            # Register hooks for different levels
            vision_model = self.vlm_model.vision_model
            
            # Pre-embedding: Early transformer layer
            if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
                if len(vision_model.encoder.layers) > 6:
                    early_layer = vision_model.encoder.layers[len(vision_model.encoder.layers)//2]
                    hooks.append(early_layer.register_forward_hook(get_activation('pre_embedding')))
                
                # Embedding: Second-to-last layer
                if len(vision_model.encoder.layers) > 1:
                    embedding_layer = vision_model.encoder.layers[-2]
                    hooks.append(embedding_layer.register_forward_hook(get_activation('embedding')))
            
            # Forward pass
            final_features = self.vlm_model.get_image_features(**inputs)
            activations['post_embedding'] = final_features
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            
            # Process activations ensuring consistent batch size
            batch_size = len(batch_images)
            
            for level_name, activation in activations.items():
                if activation is not None:
                    processed_features = self._process_activation_fixed(activation, batch_size)
                    
                    if level_name == 'pre_embedding':
                        batch_features['pre_embedding'] = processed_features
                    elif level_name == 'embedding':
                        batch_features['embedding'] = processed_features
                    elif level_name == 'post_embedding':
                        batch_features['post_embedding'] = processed_features
        
        return batch_features
    
    def _process_activation_fixed(self, activation, expected_batch_size):
        """Process activation ensuring correct batch size."""
        if activation is None:
            return [np.zeros(768) for _ in range(expected_batch_size)]
        
        # Handle different tensor shapes
        if len(activation.shape) == 3:  # [batch, seq_len, hidden]
            # Take CLS token (first token) and ensure correct batch size
            if activation.shape[0] == expected_batch_size:
                processed = activation[:, 0, :].cpu().numpy()
            else:
                # If batch size mismatch, take what we can and pad/truncate
                cls_tokens = activation[:, 0, :].cpu().numpy()
                if len(cls_tokens) > expected_batch_size:
                    processed = cls_tokens[:expected_batch_size]
                else:
                    # Pad with zeros if needed
                    padding_needed = expected_batch_size - len(cls_tokens)
                    padding = np.zeros((padding_needed, cls_tokens.shape[1]))
                    processed = np.vstack([cls_tokens, padding])
                    
        elif len(activation.shape) == 4:  # [batch, channels, height, width]
            pooled = activation.mean(dim=(-2, -1)).cpu().numpy()
            if len(pooled) == expected_batch_size:
                processed = pooled
            else:
                # Handle batch size mismatch
                if len(pooled) > expected_batch_size:
                    processed = pooled[:expected_batch_size]
                else:
                    padding_needed = expected_batch_size - len(pooled)
                    padding = np.zeros((padding_needed, pooled.shape[1]))
                    processed = np.vstack([pooled, padding])
                    
        elif len(activation.shape) == 2:  # [batch, features]
            processed = activation.cpu().numpy()
            if len(processed) != expected_batch_size:
                if len(processed) > expected_batch_size:
                    processed = processed[:expected_batch_size]
                else:
                    padding_needed = expected_batch_size - len(processed)
                    padding = np.zeros((padding_needed, processed.shape[1]))
                    processed = np.vstack([processed, padding])
        else:
            # Flatten and ensure correct batch size
            flat = activation.reshape(activation.shape[0], -1).cpu().numpy()
            if len(flat) != expected_batch_size:
                if len(flat) > expected_batch_size:
                    processed = flat[:expected_batch_size]
                else:
                    padding_needed = expected_batch_size - len(flat)
                    padding = np.zeros((padding_needed, flat.shape[1]))
                    processed = np.vstack([flat, padding])
            else:
                processed = flat
        
        return [processed[i] for i in range(len(processed))]

class AdvancedFeatureAnalyzer:
    """Advanced feature analysis with comprehensive statistics."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def compute_comprehensive_statistics(self, features: np.ndarray, labels: np.ndarray, 
                                       known_mask: np.ndarray = None) -> Dict:
        """
        Compute comprehensive statistics for feature space quality.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Sample labels
            known_mask: Boolean mask for known samples
            
        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {}
        
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Basic statistics
        stats['feature_dim'] = features.shape[1]
        stats['num_samples'] = features.shape[0]
        stats['sparsity'] = np.mean(features == 0)
        stats['mean_activation'] = np.mean(features)
        stats['std_activation'] = np.std(features)
        
        # Normalize features for distance calculations
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Class-wise analysis
        unique_labels = np.unique(labels)
        class_centroids = []
        class_spreads = []
        
        for label in unique_labels:
            class_mask = labels == label
            class_features = features_norm[class_mask]
            
            if len(class_features) > 0:
                centroid = np.mean(class_features, axis=0)
                class_centroids.append(centroid)
                
                # Intra-class spread
                distances = cdist(class_features, [centroid], metric='cosine').flatten()
                class_spreads.append(np.mean(distances))
        
        class_centroids = np.array(class_centroids)
        
        # Inter-class distances
        if len(class_centroids) > 1:
            inter_class_distances = cdist(class_centroids, class_centroids, metric='cosine')
            np.fill_diagonal(inter_class_distances, np.nan)
            stats['mean_inter_class_distance'] = np.nanmean(inter_class_distances)
            stats['min_inter_class_distance'] = np.nanmin(inter_class_distances)
            stats['max_inter_class_distance'] = np.nanmax(inter_class_distances)
        else:
            stats['mean_inter_class_distance'] = 0
            stats['min_inter_class_distance'] = 0
            stats['max_inter_class_distance'] = 0
        
        # Intra-class statistics
        stats['mean_intra_class_distance'] = np.mean(class_spreads) if class_spreads else 0
        stats['std_intra_class_distance'] = np.std(class_spreads) if class_spreads else 0
        
        # Discriminability ratio
        stats['discriminability_ratio'] = (stats['mean_inter_class_distance'] / 
                                         (stats['mean_intra_class_distance'] + 1e-8))
        
        # Clustering quality metrics
        if len(unique_labels) > 1 and len(features) > len(unique_labels):
            try:
                stats['silhouette_score'] = silhouette_score(features_norm, labels, metric='cosine')
            except:
                stats['silhouette_score'] = 0
        
        # Known vs Unknown analysis (if known_mask provided)
        if known_mask is not None:
            known_features = features_norm[known_mask]
            unknown_features = features_norm[~known_mask]
            
            if len(known_features) > 0 and len(unknown_features) > 0:
                known_centroid = np.mean(known_features, axis=0)
                unknown_centroid = np.mean(unknown_features, axis=0)
                
                # Known-Unknown separation
                stats['known_unknown_separation'] = 1 - np.dot(known_centroid, unknown_centroid) / (
                    np.linalg.norm(known_centroid) * np.linalg.norm(unknown_centroid) + 1e-8
                )
                
                # Wilderness risk approximation
                known_to_unknown_distances = cdist(known_features, unknown_features, metric='cosine')
                stats['wilderness_risk'] = np.mean(np.min(known_to_unknown_distances, axis=1))
        
        # Feature importance (variance-based)
        feature_variance = np.var(features, axis=0)
        stats['feature_importance_entropy'] = -np.sum(
            (feature_variance / np.sum(feature_variance)) * 
            np.log(feature_variance / np.sum(feature_variance) + 1e-8)
        )
        
        return stats
    
    def compare_feature_spaces(self, feature_dict: Dict[str, np.ndarray], 
                             labels: np.ndarray, known_mask: np.ndarray = None) -> Dict:
        """
        Compare multiple feature spaces (e.g., VLM vs ResNet, different levels).
        
        Args:
            feature_dict: Dictionary mapping space names to features
            labels: Sample labels
            known_mask: Boolean mask for known samples
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        # Analyze each feature space
        for space_name, features in feature_dict.items():
            if features.size > 0:
                stats = self.compute_comprehensive_statistics(features, labels, known_mask)
                comparison_results[space_name] = stats
        
        # Cross-space comparisons
        space_names = list(feature_dict.keys())
        if len(space_names) > 1:
            cross_comparisons = {}
            
            for i, space1 in enumerate(space_names):
                for j, space2 in enumerate(space_names[i+1:], i+1):
                    feat1 = feature_dict[space1]
                    feat2 = feature_dict[space2]
                    
                    if feat1.size > 0 and feat2.size > 0:
                        # Flatten if needed
                        if len(feat1.shape) > 2:
                            feat1 = feat1.reshape(feat1.shape[0], -1)
                        if len(feat2.shape) > 2:
                            feat2 = feat2.reshape(feat2.shape[0], -1)
                        
                        # Compute correlation if dimensions match
                        min_samples = min(len(feat1), len(feat2))
                        min_dim = min(feat1.shape[1], feat2.shape[1])
                        
                        if min_samples > 1 and min_dim > 1:
                            # Sample-wise correlation
                            correlation = np.corrcoef(
                                feat1[:min_samples, :min_dim].flatten(),
                                feat2[:min_samples, :min_dim].flatten()
                            )[0, 1]
                            
                            cross_comparisons[f'{space1}_vs_{space2}'] = {
                                'correlation': correlation if not np.isnan(correlation) else 0,
                                'dim_ratio': feat1.shape[1] / feat2.shape[1],
                                'performance_ratio': (
                                    comparison_results[space1]['discriminability_ratio'] /
                                    (comparison_results[space2]['discriminability_ratio'] + 1e-8)
                                )
                            }
            
            comparison_results['cross_comparisons'] = cross_comparisons
        
        return comparison_results


class EnhancedOpenSetEvaluator:
    """Enhanced open-set evaluator with comprehensive metrics."""
    
    def __init__(self):
        self.evaluation_cache = {}
    
    def calculate_enhanced_oscr(self, known_embeddings: np.ndarray, unknown_embeddings: np.ndarray,
                               known_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict:
        """
        Calculate enhanced OSCR with additional metrics.
        
        Args:
            known_embeddings: Embeddings for known samples
            unknown_embeddings: Embeddings for unknown samples  
            known_labels: True labels for known samples
            predicted_labels: Predicted labels for known samples
            
        Returns:
            Dictionary with OSCR metrics and curves
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import roc_curve, auc
        
        # Fit KNN on known embeddings for confidence scoring
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn.fit(known_embeddings)
        
        # Compute confidence scores
        known_distances, _ = knn.kneighbors(known_embeddings)
        known_scores = 1 / (1 + known_distances.mean(axis=1))
        
        unknown_distances, _ = knn.kneighbors(unknown_embeddings)
        unknown_scores = 1 / (1 + unknown_distances.mean(axis=1))
        
        # Calculate OSCR curve
        thresholds = np.linspace(0, 1, 100)
        ccr_values = []  # Correct Classification Rate
        fpr_values = []  # False Positive Rate
        
        for threshold in thresholds:
            # Known samples: correct AND above threshold
            known_accepted = known_scores >= threshold
            known_correct = predicted_labels == known_labels
            known_correct_accepted = known_accepted & known_correct
            
            ccr = np.mean(known_correct_accepted) if len(known_correct_accepted) > 0 else 0
            
            # Unknown samples: incorrectly accepted
            unknown_accepted = unknown_scores >= threshold
            fpr = np.mean(unknown_accepted) if len(unknown_accepted) > 0 else 0
            
            ccr_values.append(ccr)
            fpr_values.append(fpr)
        
        # Calculate AUC
        oscr_auc = auc(fpr_values, ccr_values)
        
        # Find optimal threshold (maximize CCR - FPR)
        ccr_array = np.array(ccr_values)
        fpr_array = np.array(fpr_values)
        optimal_idx = np.argmax(ccr_array - fpr_array)
        optimal_threshold = thresholds[optimal_idx]
        
        # Additional metrics at optimal threshold
        known_accepted_opt = known_scores >= optimal_threshold
        unknown_accepted_opt = unknown_scores >= optimal_threshold
        
        # Confusion matrix elements
        tp = np.sum(known_accepted_opt & (predicted_labels == known_labels))
        fn = np.sum(~known_accepted_opt | (predicted_labels != known_labels))
        fp = np.sum(unknown_accepted_opt)
        tn = np.sum(~unknown_accepted_opt)
        
        # Calculate F1 scores
        precision_known = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_known = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_known = 2 * precision_known * recall_known / (precision_known + recall_known) if (precision_known + recall_known) > 0 else 0
        
        # Unknown detection metrics
        precision_unknown = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_unknown = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_unknown = 2 * precision_unknown * recall_unknown / (precision_unknown + recall_unknown) if (precision_unknown + recall_unknown) > 0 else 0
        
        return {
            'oscr_auc': oscr_auc,
            'ccr_values': ccr_values,
            'fpr_values': fpr_values,
            'optimal_threshold': optimal_threshold,
            'known_scores': known_scores,
            'unknown_scores': unknown_scores,
            'f1_known': f1_known,
            'f1_unknown': f1_unknown,
            'open_set_f1': 2 * f1_known * f1_unknown / (f1_known + f1_unknown) if (f1_known + f1_unknown) > 0 else 0,
            'wilderness_risk': np.mean(unknown_scores >= optimal_threshold),
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }
    
    def calculate_cross_modal_alignment(self, image_embeddings: np.ndarray, 
                                      text_embeddings: np.ndarray,
                                      labels: np.ndarray) -> Dict:
        """
        Calculate cross-modal alignment metrics.
        
        Args:
            image_embeddings: Image embeddings
            text_embeddings: Text embeddings  
            labels: Corresponding labels
            
        Returns:
            Alignment metrics
        """
        # Normalize embeddings
        image_norm = image_embeddings / (np.linalg.norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
        text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Overall alignment
        overall_alignment = np.mean([
            np.dot(img, txt) for img, txt in zip(image_norm, text_norm)
        ])
        
        # Class-wise alignment
        unique_labels = np.unique(labels)
        class_alignments = {}
        
        for label in unique_labels:
            label_mask = labels == label
            if np.sum(label_mask) > 0:
                class_images = image_norm[label_mask]
                class_texts = text_norm[label_mask]
                
                class_alignment = np.mean([
                    np.dot(img, txt) for img, txt in zip(class_images, class_texts)
                ])
                class_alignments[label] = class_alignment
        
        # Cross-class interference
        cross_class_interference = 0
        if len(unique_labels) > 1:
            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i+1:]:
                    mask1 = labels == label1
                    mask2 = labels == label2
                    
                    if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                        # Average cross-class similarity
                        cross_sim = np.mean([
                            np.dot(img1, txt2) 
                            for img1 in image_norm[mask1]
                            for txt2 in text_norm[mask2]
                        ])
                        cross_class_interference += cross_sim
            
            cross_class_interference /= (len(unique_labels) * (len(unique_labels) - 1) / 2)
        
        return {
            'overall_alignment': overall_alignment,
            'class_alignments': class_alignments,
            'cross_class_interference': cross_class_interference,
            'alignment_quality': overall_alignment - cross_class_interference,
            'alignment_std': np.std(list(class_alignments.values())) if class_alignments else 0
        }


class VisualizationEngine:
    """Enhanced visualization engine for multi-modal analysis."""
    
    def __init__(self, output_dir='./visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_multi_level_evolution(self, feature_dict: Dict[str, Dict[str, np.ndarray]], 
                                  labels: np.ndarray, known_mask: np.ndarray,
                                  save_path: str = None) -> plt.Figure:
        """
        Plot feature evolution across different processing levels.
        
        Args:
            feature_dict: Nested dict {model: {level: features}}
            labels: Sample labels
            known_mask: Boolean mask for known samples
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        models = list(feature_dict.keys())
        levels = ['pre_embedding', 'embedding', 'post_embedding']
        
        fig, axes = plt.subplots(len(models), len(levels), figsize=(18, 6*len(models)))
        if len(models) == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, model_name in enumerate(models):
            for level_idx, level in enumerate(levels):
                ax = axes[model_idx, level_idx]
                
                if level in feature_dict[model_name]:
                    features = feature_dict[model_name][level]
                    
                    if features.size > 0:
                        # Reduce dimensionality for visualization
                        if len(features.shape) > 2:
                            features = features.reshape(features.shape[0], -1)
                        
                        if features.shape[1] > 2:
                            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features)-1))
                            features_2d = reducer.fit_transform(features)
                        else:
                            features_2d = features
                        
                        # Plot known vs unknown
                        known_features = features_2d[known_mask]
                        unknown_features = features_2d[~known_mask]
                        
                        # Plot by class for known samples
                        unique_known = np.unique(labels[known_mask])
                        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_known)))
                        
                        for i, label in enumerate(unique_known):
                            label_mask = labels[known_mask] == label
                            if np.sum(label_mask) > 0:
                                ax.scatter(known_features[label_mask, 0], known_features[label_mask, 1],
                                         c=[colors[i]], label=f'Known: {label}', alpha=0.7, s=30)
                        
                        # Plot unknown samples
                        if len(unknown_features) > 0:
                            ax.scatter(unknown_features[:, 0], unknown_features[:, 1],
                                     c='red', label='Unknown', alpha=0.5, s=30, marker='x')
                
                ax.set_title(f'{model_name} - {level.replace("_", " ").title()}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                if model_idx == 0 and level_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Level Feature Evolution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_statistics_comparison(self, stats_dict: Dict[str, Dict], 
                                         save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive feature statistics comparison.
        
        Args:
            stats_dict: Dictionary of statistics for different feature spaces
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        metrics = ['discriminability_ratio', 'silhouette_score', 'sparsity', 'known_unknown_separation']
        metric_names = ['Discriminability', 'Silhouette Score', 'Sparsity', 'Known-Unknown Sep.']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            spaces = list(stats_dict.keys())
            values = [stats_dict[space].get(metric, 0) for space in spaces]
            
            bars = ax.bar(spaces, values, alpha=0.7, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(spaces))))
            
            ax.set_title(f'{name} Comparison')
            ax.set_ylabel(name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Feature Space Quality Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_oscr_analysis(self, oscr_results: Dict, title: str = "OSCR Analysis",
                          save_path: str = None) -> plt.Figure:
        """
        Plot comprehensive OSCR analysis.
        
        Args:
            oscr_results: Results from calculate_enhanced_oscr
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # OSCR Curve
        ax1 = axes[0, 0]
        ax1.plot(oscr_results['fpr_values'], oscr_results['ccr_values'], 
                'b-', linewidth=3, label=f"OSCR (AUC = {oscr_results['oscr_auc']:.3f})")
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Baseline')
        ax1.fill_between(oscr_results['fpr_values'], oscr_results['ccr_values'], alpha=0.2)
        ax1.set_xlabel('False Positive Rate on Unknown')
        ax1.set_ylabel('Correct Classification Rate on Known')
        ax1.set_title('OSCR Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Score Distribution
        ax2 = axes[0, 1]
        ax2.hist(oscr_results['known_scores'], bins=30, alpha=0.6, 
                label='Known', color='blue', density=True)
        ax2.hist(oscr_results['unknown_scores'], bins=30, alpha=0.6, 
                label='Unknown', color='red', density=True)
        ax2.axvline(oscr_results['optimal_threshold'], color='green', linestyle='--',
                   label=f"Optimal Threshold = {oscr_results['optimal_threshold']:.3f}")
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance Metrics
        ax3 = axes[1, 0]
        metrics = ['F1 Known', 'F1 Unknown', 'Open-Set F1', 'Wilderness Risk']
        values = [oscr_results['f1_known'], oscr_results['f1_unknown'], 
                 oscr_results['open_set_f1'], oscr_results['wilderness_risk']]
        colors = ['green', 'blue', 'purple', 'red']
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_title('Performance Metrics')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confusion Matrix
        ax4 = axes[1, 1]
        cm = oscr_results['confusion_matrix']
        confusion_data = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['Rejected', 'Accepted'],
                   yticklabels=['Unknown', 'Known'])
        ax4.set_title('Confusion Matrix')
        ax4.set_xlabel('Prediction')
        ax4.set_ylabel('Ground Truth')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig