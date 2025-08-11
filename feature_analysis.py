"""
Enhanced Feature Space Analysis for VLM and ResNet Comparison
Includes pre-embedding, embedding, and classification space analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from umap import UMAP
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import pandas as pd


class FeatureSpaceAnalyzer:
    """Comprehensive feature space analysis for different embedding levels."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_levels = {}
        self.analysis_results = {}
    
    def extract_intermediate_features(self, model, images, layer_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple layers of a model.
        
        Args:
            model: The model (VLM or ResNet)
            images: Input images
            layer_names: Names of layers to extract features from
            
        Returns:
            Dictionary mapping layer names to features
        """
        features = {}
        handles = []
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = model(images)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return features
    
    def compute_feature_statistics(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Compute comprehensive statistics for feature space.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Sample labels
            
        Returns:
            Dictionary with various statistics
        """
        stats = {}
        
        # Basic statistics
        stats['mean'] = np.mean(features, axis=0)
        stats['std'] = np.std(features, axis=0)
        stats['sparsity'] = np.mean(features == 0)
        
        # Class-wise statistics
        unique_labels = np.unique(labels)
        class_means = []
        class_stds = []
        
        for label in unique_labels:
            class_features = features[labels == label]
            class_means.append(np.mean(class_features, axis=0))
            class_stds.append(np.std(class_features, axis=0))
        
        class_means = np.array(class_means)
        
        # Inter-class and intra-class distances
        inter_class_dist = cdist(class_means, class_means, metric='cosine')
        np.fill_diagonal(inter_class_dist, np.nan)
        stats['mean_inter_class_distance'] = np.nanmean(inter_class_dist)
        
        intra_class_distances = []
        for label in unique_labels:
            class_features = features[labels == label]
            if len(class_features) > 1:
                distances = cdist(class_features, class_features, metric='cosine')
                np.fill_diagonal(distances, np.nan)
                intra_class_distances.append(np.nanmean(distances))
        
        stats['mean_intra_class_distance'] = np.mean(intra_class_distances)
        stats['discriminability_ratio'] = stats['mean_inter_class_distance'] / stats['mean_intra_class_distance']
        
        # Clustering quality
        if len(unique_labels) > 1 and len(features) > len(unique_labels):
            stats['silhouette_score'] = silhouette_score(features, labels, metric='cosine')
        
        # Feature importance (variance-based)
        feature_variance = np.var(features, axis=0)
        stats['feature_importance'] = feature_variance / np.sum(feature_variance)
        
        return stats
    
    def visualize_feature_evolution(self, feature_dict: Dict[str, np.ndarray], 
                                   labels: np.ndarray, 
                                   title: str = "Feature Evolution",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize how features evolve through network layers.
        
        Args:
            feature_dict: Dictionary mapping layer names to features
            labels: Sample labels
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_layers = len(feature_dict)
        fig, axes = plt.subplots(2, n_layers, figsize=(6*n_layers, 12))
        
        if n_layers == 1:
            axes = axes.reshape(2, 1)
        
        # Color map for classes
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for idx, (layer_name, features) in enumerate(feature_dict.items()):
            # Reduce dimensionality for visualization
            if features.shape[1] > 2:
                # Use UMAP for better preservation of structure
                reducer = UMAP(n_components=2, random_state=42)
                features_2d = reducer.fit_transform(features)
            else:
                features_2d = features
            
            # Plot 1: Scatter plot
            ax1 = axes[0, idx]
            for label in unique_labels:
                mask = labels == label
                ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[color_map[label]], label=label, alpha=0.6, s=30)
            
            ax1.set_title(f'{layer_name}\nFeature Space', fontsize=10)
            ax1.set_xlabel('Component 1')
            ax1.set_ylabel('Component 2')
            if idx == 0:
                ax1.legend(bbox_to_anchor=(0, 1), loc='upper left')
            
            # Plot 2: Feature statistics
            ax2 = axes[1, idx]
            stats = self.compute_feature_statistics(features, labels)
            
            metrics = ['Sparsity', 'Silhouette', 'Discriminability']
            values = [
                stats.get('sparsity', 0),
                stats.get('silhouette_score', 0),
                stats.get('discriminability_ratio', 0) / 10  # Scale for visualization
            ]
            
            bars = ax2.bar(metrics, values, color=['blue', 'green', 'red'], alpha=0.7)
            ax2.set_ylim(0, 1)
            ax2.set_title(f'{layer_name}\nFeature Quality Metrics', fontsize=10)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_embedding_spaces(self, embeddings_dict: Dict[str, np.ndarray],
                               labels: np.ndarray,
                               known_mask: np.ndarray,
                               title: str = "Embedding Space Comparison",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare different embedding spaces (VLM vs ResNet, with known/unknown separation).
        
        Args:
            embeddings_dict: Dictionary mapping model names to embeddings
            labels: Sample labels
            known_mask: Boolean mask for known samples
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_models = len(embeddings_dict)
        fig, axes = plt.subplots(2, n_models, figsize=(8*n_models, 16))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
            # Reduce dimensionality
            if embeddings.shape[1] > 2:
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=30)
                embeddings_2d = reducer.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings
            
            # Plot 1: Known vs Unknown separation
            ax1 = axes[0, idx]
            
            # Plot unknown samples
            unknown_mask = ~known_mask
            ax1.scatter(embeddings_2d[unknown_mask, 0], embeddings_2d[unknown_mask, 1],
                       c='red', label='Unknown', alpha=0.5, s=50, marker='x')
            
            # Plot known samples with class colors
            known_embeddings = embeddings_2d[known_mask]
            known_labels = labels[known_mask]
            unique_known = np.unique(known_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_known)))
            
            for i, label in enumerate(unique_known):
                mask = known_labels == label
                ax1.scatter(known_embeddings[mask, 0], known_embeddings[mask, 1],
                           c=[colors[i]], label=f'Known: {label}', alpha=0.7, s=30)
            
            ax1.set_title(f'{model_name}\nKnown vs Unknown Separation', fontsize=12)
            ax1.set_xlabel('UMAP 1')
            ax1.set_ylabel('UMAP 2')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 2: Density plot
            ax2 = axes[1, idx]
            
            # Create 2D histogram
            h, xedges, yedges = np.histogram2d(embeddings_2d[:, 0], embeddings_2d[:, 1], bins=50)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
            im = ax2.imshow(h.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
            
            # Overlay known/unknown regions
            ax2.scatter(embeddings_2d[known_mask, 0], embeddings_2d[known_mask, 1],
                       c='white', s=1, alpha=0.3, label='Known regions')
            ax2.scatter(embeddings_2d[unknown_mask, 0], embeddings_2d[unknown_mask, 1],
                       c='red', s=1, alpha=0.3, label='Unknown regions')
            
            ax2.set_title(f'{model_name}\nEmbedding Density', fontsize=12)
            ax2.set_xlabel('UMAP 1')
            ax2.set_ylabel('UMAP 2')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_text_image_alignment(self, image_embeddings: np.ndarray,
                                   text_embeddings: Dict[str, np.ndarray],
                                   labels: np.ndarray,
                                   title: str = "Text-Image Alignment Analysis",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze alignment between image embeddings and text prompt embeddings.
        
        Args:
            image_embeddings: Image embeddings
            text_embeddings: Dictionary of text embeddings for different prompts
            labels: Image labels
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Reduce dimensionality for visualization
        all_embeddings = np.vstack([image_embeddings] + list(text_embeddings.values()))
        reducer = UMAP(n_components=2, random_state=42)
        all_reduced = reducer.fit_transform(all_embeddings)
        
        image_reduced = all_reduced[:len(image_embeddings)]
        text_reduced_dict = {}
        start_idx = len(image_embeddings)
        for prompt_name, text_emb in text_embeddings.items():
            end_idx = start_idx + len(text_emb)
            text_reduced_dict[prompt_name] = all_reduced[start_idx:end_idx]
            start_idx = end_idx
        
        # Plot 1: Overall alignment
        ax1 = axes[0, 0]
        
        # Plot images
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax1.scatter(image_reduced[mask, 0], image_reduced[mask, 1],
                       c=[colors[i]], label=f'Images: {label}', alpha=0.5, s=30)
        
        # Plot text embeddings
        markers = ['*', 'D', '^', 's', 'P']
        for idx, (prompt_name, text_reduced) in enumerate(text_reduced_dict.items()):
            ax1.scatter(text_reduced[:, 0], text_reduced[:, 1],
                       marker=markers[idx % len(markers)], s=200,
                       edgecolors='black', linewidth=2,
                       label=f'Text: {prompt_name}', alpha=0.9)
        
        ax1.set_title('Image-Text Embedding Alignment', fontsize=12)
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Distance heatmap
        ax2 = axes[0, 1]
        
        # Compute mean text embedding for each class
        mean_text_embeddings = []
        text_labels = []
        for prompt_name, text_emb in text_embeddings.items():
            if len(text_emb) == len(unique_labels):
                mean_text_embeddings.extend(text_emb)
                text_labels.extend([f"{prompt_name}_{label}" for label in unique_labels])
        
        if mean_text_embeddings:
            # Compute mean image embedding for each class
            mean_image_embeddings = []
            for label in unique_labels:
                mask = labels == label
                mean_image_embeddings.append(np.mean(image_embeddings[mask], axis=0))
            
            # Compute distance matrix
            distances = cdist(mean_image_embeddings, mean_text_embeddings, metric='cosine')
            
            # Plot heatmap
            sns.heatmap(distances, xticklabels=text_labels, yticklabels=unique_labels,
                       cmap='RdBu_r', center=0.5, ax=ax2, cbar_kws={'label': 'Cosine Distance'})
            ax2.set_title('Image-Text Class Distance Matrix', fontsize=12)
            ax2.set_xlabel('Text Prompts')
            ax2.set_ylabel('Image Classes')
        
        # Plot 3: Alignment scores distribution
        ax3 = axes[1, 0]
        
        alignment_scores = []
        for i, label in enumerate(labels):
            img_emb = image_embeddings[i]
            # Find corresponding text embedding
            label_idx = np.where(unique_labels == label)[0][0]
            
            for prompt_name, text_emb in text_embeddings.items():
                if len(text_emb) > label_idx:
                    text_emb_vec = text_emb[label_idx]
                    score = 1 - cdist([img_emb], [text_emb_vec], metric='cosine')[0, 0]
                    alignment_scores.append({
                        'score': score,
                        'prompt': prompt_name,
                        'label': label
                    })
        
        # Convert to DataFrame for easier plotting
        df_scores = pd.DataFrame(alignment_scores)
        
        # Box plot by prompt
        df_scores.boxplot(column='score', by='prompt', ax=ax3)
        ax3.set_title('Alignment Score Distribution by Prompt', fontsize=12)
        ax3.set_xlabel('Prompt Type')
        ax3.set_ylabel('Alignment Score')
        
        # Plot 4: Class-wise alignment
        ax4 = axes[1, 1]
        
        # Calculate mean alignment per class
        class_alignments = df_scores.groupby('label')['score'].mean().sort_values(ascending=False)
        
        bars = ax4.bar(range(len(class_alignments)), class_alignments.values)
        ax4.set_xticks(range(len(class_alignments)))
        ax4.set_xticklabels(class_alignments.index, rotation=45, ha='right')
        ax4.set_title('Mean Alignment Score by Class', fontsize=12)
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Mean Alignment Score')
        
        # Color bars by score
        norm = plt.Normalize(vmin=class_alignments.min(), vmax=class_alignments.max())
        colors = plt.cm.RdYlGn(norm(class_alignments.values))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class MultiClassifierEvaluator:
    """Evaluate multiple classifiers on the same feature space."""
    
    def __init__(self):
        self.classifiers = {}
        self.results = {}
    
    def add_classifier(self, name: str, classifier):
        """Add a classifier to evaluate."""
        self.classifiers[name] = classifier
    
    def evaluate_all(self, train_features: np.ndarray, train_labels: np.ndarray,
                    test_features: np.ndarray, test_labels: np.ndarray,
                    test_known_mask: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all classifiers on the same data.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            test_features: Test features
            test_labels: Test labels
            test_known_mask: Boolean mask for known test samples
            
        Returns:
            DataFrame with evaluation results
        """
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        # Default classifiers if none added
        if not self.classifiers:
            self.classifiers = {
                'KNN-3': KNeighborsClassifier(n_neighbors=3),
                'KNN-5': KNeighborsClassifier(n_neighbors=5),
                'SVM-Linear': SVC(kernel='linear', probability=True),
                'SVM-RBF': SVC(kernel='rbf', probability=True),
                'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100)
            }
        
        results = []
        
        for name, classifier in self.classifiers.items():
            # Train classifier
            classifier.fit(train_features, train_labels)
            
            # Predict on test set
            predictions = classifier.predict(test_features)
            
            # Get confidence scores
            if hasattr(classifier, 'predict_proba'):
                scores = classifier.predict_proba(test_features).max(axis=1)
            elif hasattr(classifier, 'decision_function'):
                scores = classifier.decision_function(test_features).max(axis=1)
            else:
                scores = np.ones(len(predictions))  # Default confidence
            
            # Evaluate on known samples
            known_predictions = predictions[test_known_mask]
            known_labels = test_labels[test_known_mask]
            known_scores = scores[test_known_mask]
            
            # Evaluate on unknown samples
            unknown_scores = scores[~test_known_mask]
            
            # Calculate metrics
            accuracy = accuracy_score(known_labels, known_predictions)
            f1 = f1_score(known_labels, known_predictions, average='weighted')
            
            # Calculate open-set metrics
            threshold = np.percentile(known_scores, 5)  # 95% acceptance rate
            unknown_rejection_rate = np.mean(unknown_scores < threshold)
            
            results.append({
                'Classifier': name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Unknown Rejection Rate': unknown_rejection_rate,
                'Mean Known Confidence': np.mean(known_scores),
                'Mean Unknown Confidence': np.mean(unknown_scores)
            })
        
        return pd.DataFrame(results)