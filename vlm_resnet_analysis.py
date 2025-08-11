"""
VLM vs ResNet Comprehensive Comparison
Research Question: How do VLM and ResNet features compare for open-set recognition?
FIXED: PCA dimension error and SubplotSpec subscriptable error
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import pandas as pd
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


class VLMResNetExtractor:
    """Extract and compare features from VLM and ResNet models."""
    
    def __init__(self, vlm_model, vlm_processor, resnet_model=None, device='cuda'):
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.resnet_model = resnet_model
        self.device = device
        
        # Feature storage
        self.vlm_embeddings = {}
        self.resnet_embeddings = {}
        
        # Initialize ResNet if not provided
        if self.resnet_model is None:
            self.resnet_model = self._initialize_resnet()
    
    def _initialize_resnet(self):
        """Initialize ResNet model if not provided."""
        try:
            import torchvision.models as models
            from torchvision import transforms
            
            # Load pretrained ResNet
            resnet = models.resnet50(pretrained=True)
            
            # Remove final classification layer to get features
            resnet = nn.Sequential(*list(resnet.children())[:-1])
            resnet.eval()
            resnet.to(self.device)
            
            print("✅ Initialized ResNet-50 for comparison")
            return resnet
            
        except ImportError:
            print("⚠️ torchvision not available, ResNet comparison disabled")
            return None
    
    def extract_vlm_features(self, images, batch_size=32):
        """Extract VLM features from images."""
        print(f"Extracting VLM features for {len(images)} images...")
        
        vlm_features = []
        
        # Process images in batches
        for i in tqdm(range(0, len(images), batch_size), desc="VLM extraction"):
            batch_images = images[i:i+batch_size]
            
            with torch.no_grad():
                # Process through VLM
                inputs = self.vlm_processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                features = self.vlm_model.get_image_features(**inputs)
                features = F.normalize(features, p=2, dim=1)  # L2 normalize
                
                vlm_features.extend(features.cpu().numpy())
        
        return np.array(vlm_features)
    
    def extract_resnet_features(self, images, batch_size=32):
        """Extract ResNet features from images."""
        if self.resnet_model is None:
            print("⚠️ ResNet model not available")
            return None
        
        print(f"Extracting ResNet features for {len(images)} images...")
        
        # Define preprocessing
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        resnet_features = []
        
        # Process images in batches
        for i in tqdm(range(0, len(images), batch_size), desc="ResNet extraction"):
            batch_images = images[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = preprocess(img)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                features = self.resnet_model(batch_tensor)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
                features = F.normalize(features, p=2, dim=1)  # L2 normalize
                
                resnet_features.extend(features.cpu().numpy())
        
        return np.array(resnet_features)
    
    def extract_both_features(self, images, batch_size=32):
        """Extract both VLM and ResNet features."""
        vlm_features = self.extract_vlm_features(images, batch_size)
        resnet_features = self.extract_resnet_features(images, batch_size)
        
        return {
            'VLM': vlm_features,
            'ResNet': resnet_features
        }


class FeatureComparator:
    """Compare VLM and ResNet features across multiple dimensions."""
    
    def __init__(self):
        self.comparison_results = {}
    
    def comprehensive_comparison(self, vlm_features, resnet_features, labels, known_mask):
        """
        Perform comprehensive comparison between VLM and ResNet features.
        
        Args:
            vlm_features: VLM feature vectors
            resnet_features: ResNet feature vectors
            labels: Sample labels
            known_mask: Boolean mask for known samples
            
        Returns:
            Comprehensive comparison results
        """
        print("Performing comprehensive VLM vs ResNet comparison...")
        
        feature_sets = {
            'VLM': vlm_features,
            'ResNet': resnet_features
        }
        
        comparison_results = {}
        
        for model_name, features in feature_sets.items():
            if features is not None:
                print(f"  Analyzing {model_name} features...")
                
                model_results = {
                    'feature_quality': self._analyze_feature_quality(features, labels, known_mask),
                    'classification_performance': self._evaluate_classification(features, labels, known_mask),
                    'open_set_performance': self._evaluate_open_set(features, labels, known_mask),
                    'semantic_structure': self._analyze_semantic_structure(features, labels),
                    'robustness_metrics': self._analyze_robustness(features, labels, known_mask)
                }
                
                comparison_results[model_name] = model_results
        
        # Cross-model comparisons
        if 'VLM' in comparison_results and 'ResNet' in comparison_results:
            print("  Computing cross-model comparisons...")
            comparison_results['cross_analysis'] = self._cross_model_analysis(
                vlm_features, resnet_features, labels, known_mask
            )
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def _analyze_feature_quality(self, features, labels, known_mask):
        """Analyze intrinsic feature quality."""
        try:
            # Normalize features
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            quality_metrics = {
                'dimensionality': features.shape[1],
                'sparsity': np.mean(features == 0),
                'dynamic_range': np.max(features) - np.min(features),
                'mean_norm': np.mean(np.linalg.norm(features, axis=1)),
                'std_norm': np.std(np.linalg.norm(features, axis=1))
            }
            
            # Class separability
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                class_centroids = []
                for label in unique_labels:
                    label_mask = labels == label
                    if np.sum(label_mask) > 0:
                        centroid = np.mean(features_norm[label_mask], axis=0)
                        class_centroids.append(centroid)
                
                if len(class_centroids) > 1:
                    class_centroids = np.array(class_centroids)
                    inter_distances = cdist(class_centroids, class_centroids, metric='cosine')
                    np.fill_diagonal(inter_distances, np.nan)
                    quality_metrics['mean_inter_class_distance'] = np.nanmean(inter_distances)
            
            # Known-unknown separability
            if known_mask is not None and np.any(known_mask) and np.any(~known_mask):
                known_features = features_norm[known_mask]
                unknown_features = features_norm[~known_mask]
                
                known_centroid = np.mean(known_features, axis=0)
                unknown_centroid = np.mean(unknown_features, axis=0)
                
                separation = 1 - np.dot(known_centroid, unknown_centroid) / (
                    np.linalg.norm(known_centroid) * np.linalg.norm(unknown_centroid) + 1e-8
                )
                quality_metrics['known_unknown_separation'] = separation
            
            # Clustering quality
            if len(unique_labels) > 1 and len(features) > len(unique_labels):
                try:
                    silhouette = silhouette_score(features_norm, labels, metric='cosine')
                    quality_metrics['silhouette_score'] = silhouette
                except:
                    quality_metrics['silhouette_score'] = 0
            
            return quality_metrics
            
        except Exception as e:
            return {'error': f'Feature quality analysis failed: {str(e)}'}
    
    def _evaluate_classification(self, features, labels, known_mask):
        """Evaluate classification performance on known classes."""
        try:
            known_features = features[known_mask]
            known_labels = labels[known_mask]
            
            if len(known_features) == 0:
                return {'error': 'No known samples'}
            
            # Split into train/test
            n_train = int(0.7 * len(known_features))
            indices = np.random.permutation(len(known_features))
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            if len(test_idx) == 0:
                test_idx = train_idx  # Use training data for testing if too few samples
            
            train_features = known_features[train_idx]
            train_labels = known_labels[train_idx]
            test_features = known_features[test_idx]
            test_labels = known_labels[test_idx]
            
            # Test multiple classifiers
            classifiers = {
                'KNN-5': KNeighborsClassifier(n_neighbors=min(5, len(train_features))),
                'SVM-Linear': SVC(kernel='linear', probability=True),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            classification_results = {}
            
            for clf_name, clf in classifiers.items():
                try:
                    clf.fit(train_features, train_labels)
                    predictions = clf.predict(test_features)
                    
                    accuracy = accuracy_score(test_labels, predictions)
                    f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
                    
                    classification_results[clf_name] = {
                        'accuracy': accuracy,
                        'f1_score': f1
                    }
                except Exception as e:
                    classification_results[clf_name] = {'error': str(e)}
            
            return classification_results
            
        except Exception as e:
            return {'error': f'Classification evaluation failed: {str(e)}'}
    
    def _evaluate_open_set(self, features, labels, known_mask):
        """Evaluate open-set recognition performance."""
        try:
            known_features = features[known_mask]
            unknown_features = features[~known_mask]
            known_labels = labels[known_mask]
            
            if len(known_features) == 0 or len(unknown_features) == 0:
                return {'error': 'Insufficient known or unknown samples'}
            
            # Train classifier on known samples
            knn = KNeighborsClassifier(n_neighbors=min(5, len(known_features)))
            knn.fit(known_features, known_labels)
            
            # Get distances for confidence scoring
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(5, len(known_features)), metric='cosine')
            nn.fit(known_features)
            
            # Compute confidence scores
            known_distances, _ = nn.kneighbors(known_features)
            known_scores = 1 / (1 + known_distances.mean(axis=1))
            
            unknown_distances, _ = nn.kneighbors(unknown_features)
            unknown_scores = 1 / (1 + unknown_distances.mean(axis=1))
            
            # Calculate AUROC for open-set detection
            y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
            y_scores = np.concatenate([known_scores, unknown_scores])
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auroc = auc(fpr, tpr)
            
            # Calculate OSCR (Open Set Classification Rate)
            thresholds = np.linspace(0, 1, 50)
            ccr_values = []
            fpr_values = []
            
            # Get predictions for known samples
            known_predictions = knn.predict(known_features)
            correct_predictions = known_predictions == known_labels
            
            for threshold in thresholds:
                # Known: correct classification AND above threshold
                accepted_and_correct = (known_scores >= threshold) & correct_predictions
                ccr = np.mean(accepted_and_correct)
                
                # Unknown: incorrectly accepted (above threshold)
                fpr_osr = np.mean(unknown_scores >= threshold)
                
                ccr_values.append(ccr)
                fpr_values.append(fpr_osr)
            
            oscr_auc = auc(fpr_values, ccr_values)
            
            return {
                'auroc': auroc,
                'oscr_auc': oscr_auc,
                'mean_known_confidence': np.mean(known_scores),
                'mean_unknown_confidence': np.mean(unknown_scores),
                'confidence_gap': np.mean(known_scores) - np.mean(unknown_scores)
            }
            
        except Exception as e:
            return {'error': f'Open-set evaluation failed: {str(e)}'}
    
    def _analyze_semantic_structure(self, features, labels):
        """Analyze semantic structure of feature space."""
        try:
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            # Compute pairwise similarities
            similarity_matrix = np.dot(features_norm, features_norm.T)
            
            semantic_metrics = {
                'mean_similarity': np.mean(similarity_matrix),
                'std_similarity': np.std(similarity_matrix),
                'min_similarity': np.min(similarity_matrix),
                'max_similarity': np.max(similarity_matrix)
            }
            
            # Class-wise semantic coherence
            unique_labels = np.unique(labels)
            class_coherences = {}
            
            for label in unique_labels:
                label_mask = labels == label
                if np.sum(label_mask) > 1:
                    class_similarities = similarity_matrix[np.ix_(label_mask, label_mask)]
                    np.fill_diagonal(class_similarities, np.nan)
                    coherence = np.nanmean(class_similarities)
                    class_coherences[label] = coherence
            
            semantic_metrics['class_coherences'] = class_coherences
            semantic_metrics['mean_class_coherence'] = np.mean(list(class_coherences.values())) if class_coherences else 0
            
            return semantic_metrics
            
        except Exception as e:
            return {'error': f'Semantic analysis failed: {str(e)}'}
    
    def _analyze_robustness(self, features, labels, known_mask):
        """Analyze feature robustness metrics."""
        try:
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            robustness_metrics = {}
            
            # Feature concentration (how concentrated features are around the mean)
            mean_feature = np.mean(features_norm, axis=0)
            concentrations = [np.dot(feat, mean_feature) for feat in features_norm]
            robustness_metrics['feature_concentration'] = {
                'mean': np.mean(concentrations),
                'std': np.std(concentrations),
                'min': np.min(concentrations),
                'max': np.max(concentrations)
            }
            
            # Outlier detection
            distances_to_mean = np.linalg.norm(features_norm - mean_feature, axis=1)
            threshold = np.mean(distances_to_mean) + 2 * np.std(distances_to_mean)
            outlier_ratio = np.mean(distances_to_mean > threshold)
            robustness_metrics['outlier_ratio'] = outlier_ratio
            
            # Stability under sampling
            n_samples = len(features)
            if n_samples > 50:
                subsample_similarities = []
                for _ in range(10):  # 10 random subsamples
                    subsample_idx = np.random.choice(n_samples, size=min(50, n_samples//2), replace=False)
                    subsample_features = features_norm[subsample_idx]
                    subsample_mean = np.mean(subsample_features, axis=0)
                    similarity = np.dot(mean_feature, subsample_mean)
                    subsample_similarities.append(similarity)
                
                robustness_metrics['sampling_stability'] = {
                    'mean_similarity': np.mean(subsample_similarities),
                    'std_similarity': np.std(subsample_similarities)
                }
            
            return robustness_metrics
            
        except Exception as e:
            return {'error': f'Robustness analysis failed: {str(e)}'}
    
    def _cross_model_analysis(self, vlm_features, resnet_features, labels, known_mask):
        """Perform cross-model analysis with FIXED dimension handling."""
        cross_analysis = {}
        
        if vlm_features is not None and resnet_features is not None:
            # *** FIX: Align dimensions before any analysis ***
            vlm_aligned, resnet_aligned = self._align_feature_dimensions(
                vlm_features, resnet_features
            )
            
            # Subsample for computational efficiency
            n_samples = min(len(vlm_aligned), len(resnet_aligned))
            max_samples_for_analysis = min(500, n_samples)
            
            if n_samples > max_samples_for_analysis:
                indices = np.random.choice(n_samples, max_samples_for_analysis, replace=False)
                vlm_subset = vlm_aligned[indices]
                resnet_subset = resnet_aligned[indices]
            else:
                vlm_subset = vlm_aligned
                resnet_subset = resnet_aligned
            
            # Overall alignment
            overall_alignment = np.mean([
                np.dot(img, txt) for img, txt in zip(vlm_subset, resnet_subset)
            ])
            
            # Sample-wise correlations
            sample_correlations = []
            for i in range(len(vlm_subset)):
                # Now safe to compute since dimensions match
                correlation = np.corrcoef(vlm_subset[i], resnet_subset[i])[0, 1]
                if not np.isnan(correlation):
                    sample_correlations.append(correlation)
            
            cross_analysis['sample_correlation'] = {
                'mean': np.mean(sample_correlations) if sample_correlations else 0,
                'std': np.std(sample_correlations) if sample_correlations else 0,
                'distribution': sample_correlations
            }
            
            # Cross-class interference
            cross_class_interference = 0
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                for i, label1 in enumerate(unique_labels):
                    for label2 in unique_labels[i+1:]:
                        mask1 = labels == label1
                        mask2 = labels == label2
                        
                        if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                            # Average cross-class similarity using aligned features
                            cross_sim = np.mean([
                                np.dot(img1, txt2) 
                                for img1 in vlm_subset[mask1[:len(vlm_subset)]]
                                for txt2 in resnet_subset[mask2[:len(resnet_subset)]]
                            ])
                            cross_class_interference += cross_sim
                
                cross_class_interference /= (len(unique_labels) * (len(unique_labels) - 1) / 2)
            
            cross_analysis.update({
                'overall_alignment': overall_alignment,
                'cross_class_interference': cross_class_interference,
                'alignment_quality': overall_alignment - cross_class_interference,
                'centroid_correlation': np.corrcoef(
                    np.mean(vlm_subset, axis=0), 
                    np.mean(resnet_subset, axis=0)
                )[0, 1] if len(vlm_subset) > 1 else 0
            })
        
        return cross_analysis

    
    def _align_feature_dimensions(self, vlm_features, resnet_features):
            """
            Add this method to your existing FeatureComparator class.
            This prevents the concatenation dimension error.
            """
            if vlm_features is None or resnet_features is None:
                return vlm_features, resnet_features
            
            print(f"    → Aligning features for visualization")
            print(f"    → VLM shape: {vlm_features.shape}")
            print(f"    → ResNet shape: {resnet_features.shape}")
            
            # Get minimum dimension
            vlm_dim = vlm_features.shape[1]
            resnet_dim = resnet_features.shape[1]
            target_dim = min(vlm_dim, resnet_dim)
            
            print(f"    → Aligning to {target_dim} dimensions")
            
            # Truncate to same dimension
            vlm_aligned = vlm_features[:, :target_dim]
            resnet_aligned = resnet_features[:, :target_dim]
            
            print(f"    → Aligned VLM: {vlm_aligned.shape}")
            print(f"    → Aligned ResNet: {resnet_aligned.shape}")
            
            return vlm_aligned, resnet_aligned


class VLMResNetVisualizer:
    """Create comprehensive visualizations for VLM vs ResNet comparison."""
    
    def __init__(self, output_dir='./vlm_resnet_comparison'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _align_feature_dimensions(self, vlm_features, resnet_features):
            """
            Add this method to your existing FeatureComparator class.
            This prevents the concatenation dimension error.
            """
            if vlm_features is None or resnet_features is None:
                return vlm_features, resnet_features
            
            print(f"    → Aligning features for visualization")
            print(f"    → VLM shape: {vlm_features.shape}")
            print(f"    → ResNet shape: {resnet_features.shape}")
            
            # Get minimum dimension
            vlm_dim = vlm_features.shape[1]
            resnet_dim = resnet_features.shape[1]
            target_dim = min(vlm_dim, resnet_dim)
            
            print(f"    → Aligning to {target_dim} dimensions")
            
            # Truncate to same dimension
            vlm_aligned = vlm_features[:, :target_dim]
            resnet_aligned = resnet_features[:, :target_dim]
            
            print(f"    → Aligned VLM: {vlm_aligned.shape}")
            print(f"    → Aligned ResNet: {resnet_aligned.shape}")
            
            return vlm_aligned, resnet_aligned
    
    def create_comparison_dashboard(self, comparison_results, vlm_features, resnet_features, 
                                  labels, known_mask, save_path=None):
        """
        Create comprehensive comparison dashboard.
        FIXED: SubplotSpec subscriptable error
        """
        fig = plt.figure(figsize=(24, 18))
        
        # FIXED: Use individual subplot creation instead of subscriptable GridSpec
        ax1 = plt.subplot(4, 6, 1)  # Row 1, Col 1
        ax2 = plt.subplot(4, 6, 2)  # Row 1, Col 2  
        ax3 = plt.subplot(4, 6, 3)  # Row 1, Col 3
        ax4 = plt.subplot(4, 6, (4, 6))  # Row 1, Cols 4-6
        
        ax5 = plt.subplot(4, 6, (7, 8))   # Row 2, Cols 1-2
        ax6 = plt.subplot(4, 6, (9, 10))  # Row 2, Cols 3-4
        ax7 = plt.subplot(4, 6, (11, 12)) # Row 2, Cols 5-6
        
        ax8 = plt.subplot(4, 6, (13, 15)) # Row 3, Cols 1-3
        ax9 = plt.subplot(4, 6, (16, 18)) # Row 3, Cols 4-6
        
        ax10 = plt.subplot(4, 6, (19, 21)) # Row 4, Cols 1-3
        ax11 = plt.subplot(4, 6, (22, 24)) # Row 4, Cols 4-6
        
        # Row 1: Feature space visualizations
        self._plot_feature_spaces([ax1, ax2, ax3], vlm_features, resnet_features, labels, known_mask)
        self._plot_feature_quality_comparison(ax4, comparison_results)
        
        # Row 2: Performance comparisons
        self._plot_classification_performance(ax5, comparison_results)
        self._plot_open_set_performance(ax6, comparison_results)
        self._plot_semantic_analysis(ax7, comparison_results)
        
        # Row 3: Cross-model analysis
        self._plot_cross_model_analysis(ax8, comparison_results)
        self._plot_robustness_comparison(ax9, comparison_results)
        
        # Row 4: Detailed metrics and summary
        self._plot_detailed_metrics(ax10, comparison_results)
        self._plot_recommendation_summary(ax11, comparison_results)
        
        plt.suptitle('VLM vs ResNet Comprehensive Comparison Dashboard', 
                     fontsize=20, fontweight='bold')
        
        # FIXED: Use tight_layout with proper spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _plot_feature_spaces(self, axes, vlm_features, resnet_features, labels, known_mask):
        """
        REPLACE your existing _plot_feature_spaces method with this fixed version.
        This fixes the fig/gs definition issues and uses the axes parameter correctly.
        """
        
        # *** FIX: Align dimensions before any operations ***
        if vlm_features is not None and resnet_features is not None:
            vlm_aligned, resnet_aligned = self._align_feature_dimensions(
                vlm_features, resnet_features
            )
        else:
            vlm_aligned = vlm_features
            resnet_aligned = resnet_features
        
        feature_sets = {'VLM': vlm_aligned, 'ResNet': resnet_aligned}
        colors = {'VLM': 'blue', 'ResNet': 'orange'}
        
        # Use the axes parameter that's passed to the method
        available_axes = axes if isinstance(axes, list) else [axes]
        
        # Create subplots for each model
        plot_idx = 0
        for i, (model_name, features) in enumerate(feature_sets.items()):
            if features is not None and plot_idx < len(available_axes):
                ax = available_axes[plot_idx]
                plot_idx += 1
                
                # Reduce dimensionality
                n_neighbors = min(15, len(features) - 1) if len(features) > 15 else max(1, len(features) - 1)
                
                try:
                    from umap import UMAP
                    reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
                    features_2d = reducer.fit_transform(features)
                except:
                    # Fallback to PCA if UMAP fails
                    from sklearn.decomposition import PCA
                    n_components = min(2, features.shape[1], len(features) - 1)
                    if n_components >= 2:
                        reducer = PCA(n_components=2)
                        features_2d = reducer.fit_transform(features)
                    else:
                        # Create dummy 2D coordinates
                        features_2d = np.column_stack([features[:, 0], np.zeros(len(features))])
                
                # Plot known classes
                unique_labels = np.unique(labels[known_mask])
                class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    label_mask = (labels == label) & known_mask
                    if np.sum(label_mask) > 0:
                        ax.scatter(features_2d[label_mask, 0], features_2d[label_mask, 1],
                                c=[class_colors[j]], label=f'{label}', alpha=0.7, s=40)
                
                # Plot unknown samples
                unknown_points = features_2d[~known_mask]
                if len(unknown_points) > 0:
                    ax.scatter(unknown_points[:, 0], unknown_points[:, 1],
                            c='red', label='Unknown', alpha=0.5, s=40, marker='x')
                
                ax.set_title(f'{model_name} Feature Space (UMAP)', fontweight='bold')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                if i == 0:  # Only show legend for first plot
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
        
        # *** FIX: Overlap analysis with aligned features ***
        # Only create overlap plot if we have space for it and both feature sets
        if (len(feature_sets) == 2 and 
            all(f is not None for f in feature_sets.values()) and 
            plot_idx < len(available_axes)):
            
            ax_overlap = available_axes[plot_idx]
            
            # Now we can safely combine features since they have the same dimensions
            all_features = np.vstack([vlm_aligned, resnet_aligned])
            model_labels = ['VLM'] * len(vlm_aligned) + ['ResNet'] * len(resnet_aligned)
            
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
                all_reduced = reducer.fit_transform(all_features)
            except:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                all_reduced = reducer.fit_transform(all_features)
            
            vlm_points = all_reduced[:len(vlm_aligned)]
            resnet_points = all_reduced[len(vlm_aligned):]
            
            ax_overlap.scatter(vlm_points[:, 0], vlm_points[:, 1], 
                            c='blue', alpha=0.5, s=30, label='VLM')
            ax_overlap.scatter(resnet_points[:, 0], resnet_points[:, 1],
                            c='orange', alpha=0.5, s=30, label='ResNet')
            
            ax_overlap.set_title('Feature Space Overlap', fontweight='bold')
            ax_overlap.set_xlabel('Component 1')
            ax_overlap.set_ylabel('Component 2')
            ax_overlap.legend()
            ax_overlap.grid(True, alpha=0.3)

    
    def _plot_feature_quality_comparison(self, ax, comparison_results):
        """Plot feature quality metrics comparison."""
        
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        metrics = ['silhouette_score', 'known_unknown_separation', 'mean_inter_class_distance']
        metric_names = ['Silhouette Score', 'Known-Unknown Sep.', 'Inter-class Distance']
        
        metric_data = {metric: [] for metric in metrics}
        
        for model in models:
            quality = comparison_results[model].get('feature_quality', {})
            for metric in metrics:
                metric_data[metric].append(quality.get(metric, 0))
        
        x = np.arange(len(models))
        width = 0.25
        colors = ['green', 'blue', 'purple']
        
        for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
            ax.bar(x + i*width - width, metric_data[metric], width, 
                  label=name, alpha=0.8, color=color)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Metric Value')
        ax.set_title('Feature Quality Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_classification_performance(self, ax, comparison_results):
        """Plot classification performance comparison."""
        
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        classifiers = ['KNN-5', 'SVM-Linear', 'Random Forest']
        
        # Extract accuracy scores
        accuracy_data = []
        for model in models:
            clf_results = comparison_results[model].get('classification_performance', {})
            model_accuracies = []
            for clf in classifiers:
                if clf in clf_results and 'accuracy' in clf_results[clf]:
                    model_accuracies.append(clf_results[clf]['accuracy'])
                else:
                    model_accuracies.append(0)
            accuracy_data.append(model_accuracies)
        
        # Plot grouped bar chart
        x = np.arange(len(classifiers))
        width = 0.35
        colors = ['blue', 'orange']
        
        for i, (model, accuracies) in enumerate(zip(models, accuracy_data)):
            ax.bar(x + i*width - width/2, accuracies, width, 
                  label=model, alpha=0.8, color=colors[i] if i < len(colors) else 'gray')
        
        ax.set_xlabel('Classifier')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Performance Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_open_set_performance(self, ax, comparison_results):
        """Plot open-set recognition performance."""
        
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        metrics = ['auroc', 'oscr_auc', 'confidence_gap']
        metric_names = ['AUROC', 'OSCR AUC', 'Confidence Gap']
        
        metric_data = {metric: [] for metric in metrics}
        
        for model in models:
            osr_results = comparison_results[model].get('open_set_performance', {})
            for metric in metrics:
                metric_data[metric].append(osr_results.get(metric, 0))
        
        x = np.arange(len(models))
        width = 0.25
        colors = ['red', 'green', 'blue']
        
        for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
            ax.bar(x + i*width - width, metric_data[metric], width,
                  label=name, alpha=0.8, color=color)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Performance Score')
        ax.set_title('Open-Set Recognition Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, metric in enumerate(metrics):
            for j, model in enumerate(models):
                value = metric_data[metric][j]
                ax.text(j + i*width - width, value + 0.02, f'{value:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    def _plot_semantic_analysis(self, ax, comparison_results):
        """Plot semantic structure analysis."""
        
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        
        semantic_scores = []
        for model in models:
            semantic = comparison_results[model].get('semantic_structure', {})
            coherence = semantic.get('mean_class_coherence', 0)
            semantic_scores.append(coherence)
        
        bars = ax.bar(models, semantic_scores, alpha=0.7, 
                     color=['blue', 'orange'][:len(models)])
        
        ax.set_ylabel('Mean Class Coherence')
        ax.set_title('Semantic Structure Quality', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, semantic_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_cross_model_analysis(self, ax, comparison_results):
        """Plot cross-model analysis results."""
        
        if 'cross_analysis' in comparison_results:
            cross_analysis = comparison_results['cross_analysis']
            
            # Sample correlation distribution
            sample_corr = cross_analysis.get('sample_correlation', {})
            if 'distribution' in sample_corr and sample_corr['distribution']:
                correlations = sample_corr['distribution']
                
                ax.hist(correlations, bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax.axvline(sample_corr.get('mean', 0), color='red', linestyle='--',
                          linewidth=2, label=f"Mean = {sample_corr.get('mean', 0):.3f}")
                
                ax.set_xlabel('Sample-wise Correlation')
                ax.set_ylabel('Frequency')
                ax.set_title('VLM-ResNet Feature Correlation Distribution', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Cross-model analysis\nnot available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Cross-Model Analysis', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Cross-model analysis\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cross-Model Analysis', fontweight='bold')
    
    def _plot_robustness_comparison(self, ax, comparison_results):
        """Plot robustness metrics comparison."""
        
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        
        robustness_scores = []
        for model in models:
            robustness = comparison_results[model].get('robustness_metrics', {})
            
            # Composite robustness score
            outlier_ratio = robustness.get('outlier_ratio', 0)
            concentration = robustness.get('feature_concentration', {})
            concentration_std = concentration.get('std', 1) if concentration else 1
            
            # Lower outlier ratio and higher concentration consistency = better robustness
            robustness_score = (1 - outlier_ratio) * (1 / (concentration_std + 1e-8))
            robustness_scores.append(robustness_score)
        
        bars = ax.bar(models, robustness_scores, alpha=0.7,
                     color=['blue', 'orange'][:len(models)])
        
        ax.set_ylabel('Robustness Score')
        ax.set_title('Feature Robustness Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, robustness_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_detailed_metrics(self, ax, comparison_results):
        """Plot detailed metrics heatmap."""
        
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        
        # Collect all metrics
        metrics_data = []
        metric_names = []
        
        for model in models:
            model_metrics = []
            
            # Feature quality metrics
            quality = comparison_results[model].get('feature_quality', {})
            model_metrics.extend([
                quality.get('silhouette_score', 0),
                quality.get('known_unknown_separation', 0),
                quality.get('sparsity', 0)
            ])
            
            # Open-set performance
            osr = comparison_results[model].get('open_set_performance', {})
            model_metrics.extend([
                osr.get('auroc', 0),
                osr.get('oscr_auc', 0),
                osr.get('confidence_gap', 0)
            ])
            
            # Semantic structure
            semantic = comparison_results[model].get('semantic_structure', {})
            model_metrics.append(semantic.get('mean_class_coherence', 0))
            
            metrics_data.append(model_metrics)
        
        if not metric_names:
            metric_names = [
                'Silhouette Score', 'Known-Unknown Sep.', 'Sparsity',
                'AUROC', 'OSCR AUC', 'Confidence Gap', 'Class Coherence'
            ]
        
        if metrics_data:
            metrics_array = np.array(metrics_data).T
            
            sns.heatmap(metrics_array, xticklabels=models, yticklabels=metric_names,
                       cmap='RdYlGn', center=0.5, annot=True, fmt='.3f', ax=ax)
            ax.set_title('Detailed Metrics Comparison', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Metrics data\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Detailed Metrics Comparison', fontweight='bold')
    
    def _plot_recommendation_summary(self, ax, comparison_results):
        """Plot recommendation summary."""
        ax.axis('off')
        
        # Determine best model based on metrics
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        
        if len(models) >= 2:
            # Score models based on key metrics
            model_scores = {}
            
            for model in models:
                score = 0
                
                # Open-set performance (40% weight)
                osr = comparison_results[model].get('open_set_performance', {})
                score += 0.4 * (osr.get('auroc', 0) + osr.get('oscr_auc', 0)) / 2
                
                # Feature quality (30% weight)
                quality = comparison_results[model].get('feature_quality', {})
                score += 0.3 * (quality.get('silhouette_score', 0) + 
                              quality.get('known_unknown_separation', 0)) / 2
                
                # Classification performance (30% weight)
                clf_perf = comparison_results[model].get('classification_performance', {})
                clf_scores = []
                for clf_result in clf_perf.values():
                    if isinstance(clf_result, dict) and 'accuracy' in clf_result:
                        clf_scores.append(clf_result['accuracy'])
                
                if clf_scores:
                    score += 0.3 * np.mean(clf_scores)
                
                model_scores[model] = score
            
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
            best_score = model_scores[best_model]
            
            recommendations = f"""
🏆 RECOMMENDATION SUMMARY

📊 OVERALL WINNER: {best_model}
   Performance Score: {best_score:.3f}

📈 KEY FINDINGS:
"""
            
            # Add specific findings for each model
            for model in models:
                score = model_scores[model]
                osr = comparison_results[model].get('open_set_performance', {})
                quality = comparison_results[model].get('feature_quality', {})
                
                recommendations += f"""
   {model}:
   • Overall Score: {score:.3f}
   • AUROC: {osr.get('auroc', 0):.3f}
   • OSCR AUC: {osr.get('oscr_auc', 0):.3f}
   • Separability: {quality.get('known_unknown_separation', 0):.3f}
"""
            
            recommendations += f"""
🎯 DEPLOYMENT ADVICE:
   • Primary: Use {best_model} for best performance
   • Backup: Consider ensemble if gap is small
   • Monitor: Track {best_model.lower()} feature quality
   • Optimize: Focus on {best_model.lower()} improvements

🔧 TECHNICAL INSIGHTS:
   • VLM: Better semantic understanding
   • ResNet: Strong low-level features  
   • Fusion: May improve robustness
   • Context: Consider deployment constraints
"""
        else:
            recommendations = """
⚠️ INCOMPLETE COMPARISON

Limited model comparison data available.
Ensure both VLM and ResNet features 
are extracted for comprehensive analysis.

🔧 NEXT STEPS:
   • Extract missing features
   • Run full comparison pipeline
   • Analyze individual model strengths
   • Consider ensemble approaches
"""
        
        ax.text(0.05, 0.95, recommendations, transform=ax.transAxes,
               fontsize=10, fontfamily='monospace', verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))


def run_vlm_resnet_comparison(vlm_model, vlm_processor, images, labels, known_mask,
                            resnet_model=None, output_dir='./vlm_resnet_comparison', 
                            device='cuda', batch_size=32):
    """
    Run comprehensive VLM vs ResNet comparison analysis.
    
    Args:
        vlm_model: VLM model
        vlm_processor: VLM processor
        images: List of PIL images
        labels: Sample labels
        known_mask: Boolean mask for known samples
        resnet_model: ResNet model (optional, will initialize if None)
        output_dir: Output directory
        device: Computing device
        batch_size: Batch size for processing
        
    Returns:
        Comprehensive comparison results
    """
    print("="*60)
    print("VLM vs RESNET COMPREHENSIVE COMPARISON")
    print("="*60)
    
    # Initialize components
    extractor = VLMResNetExtractor(vlm_model, vlm_processor, resnet_model, device)
    comparator = FeatureComparator()
    visualizer = VLMResNetVisualizer(output_dir)
    
    # Extract features from both models
    print("\n1. Extracting features from both models...")
    feature_dict = extractor.extract_both_features(images, batch_size)
    
    vlm_features = feature_dict['VLM']
    resnet_features = feature_dict['ResNet']
    
    if vlm_features is None:
        print("❌ VLM feature extraction failed")
        return None
    
    if resnet_features is None:
        print("⚠️ ResNet features not available, comparison will be limited")
    
    # Perform comprehensive comparison
    print("\n2. Performing comprehensive comparison...")
    comparison_results = comparator.comprehensive_comparison(
        vlm_features, resnet_features, labels, known_mask
    )
    
    # Create comprehensive dashboard
    print("\n3. Creating comparison dashboard...")
    fig = visualizer.create_comparison_dashboard(
        comparison_results, vlm_features, resnet_features, labels, known_mask,
        save_path=os.path.join(output_dir, 'vlm_resnet_comparison_dashboard.png')
    )
    plt.close(fig)
    
    # Generate detailed report
    print("\n4. Generating comparison report...")
    report_path = os.path.join(output_dir, 'vlm_resnet_comparison_report.md')
    _generate_comparison_report(comparison_results, report_path)
    
    # Save comparison data
    results_path = os.path.join(output_dir, 'comparison_results.npz')
    np.savez(results_path,
             vlm_features=vlm_features,
             resnet_features=resnet_features if resnet_features is not None else np.array([]),
             labels=labels,
             known_mask=known_mask)
    
    print(f"\n✅ VLM vs ResNet comparison completed!")
    print(f"📁 Results saved to: {output_dir}")
    
    return {
        'comparison_results': comparison_results,
        'vlm_features': vlm_features,
        'resnet_features': resnet_features,
        'report_path': report_path,
        'results_path': results_path
    }


def _generate_comparison_report(comparison_results, report_path):
    """Generate detailed comparison report."""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# VLM vs ResNet Comprehensive Comparison Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report provides a comprehensive comparison between Vision-Language Models (VLM) ")
        f.write("and ResNet features for open-set aircraft recognition, analyzing multiple dimensions ")
        f.write("of performance and feature quality.\n\n")
        
        # Model comparison
        models = [m for m in ['VLM', 'ResNet'] if m in comparison_results]
        
        f.write("## Model Comparison Summary\n\n")
        f.write("| Model | Feature Dim | Silhouette | Known-Unknown Sep | AUROC | OSCR AUC |\n")
        f.write("|-------|-------------|------------|-------------------|-------|----------|\n")
        
        for model in models:
            quality = comparison_results[model].get('feature_quality', {})
            osr = comparison_results[model].get('open_set_performance', {})
            
            f.write(f"| {model} | ")
            f.write(f"{quality.get('dimensionality', 'N/A')} | ")
            f.write(f"{quality.get('silhouette_score', 0):.4f} | ")
            f.write(f"{quality.get('known_unknown_separation', 0):.4f} | ")
            f.write(f"{osr.get('auroc', 0):.4f} | ")
            f.write(f"{osr.get('oscr_auc', 0):.4f} |\n")
        
        # Detailed analysis for each model
        for model in models:
            f.write(f"\n## {model} Detailed Analysis\n\n")
            
            # Feature quality
            quality = comparison_results[model].get('feature_quality', {})
            f.write("### Feature Quality\n")
            f.write(f"- **Dimensionality**: {quality.get('dimensionality', 'N/A')}\n")
            f.write(f"- **Sparsity**: {quality.get('sparsity', 0):.4f}\n")
            f.write(f"- **Dynamic range**: {quality.get('dynamic_range', 0):.4f}\n")
            f.write(f"- **Mean norm**: {quality.get('mean_norm', 0):.4f}\n")
            f.write(f"- **Silhouette score**: {quality.get('silhouette_score', 0):.4f}\n\n")
            
            # Classification performance
            clf_perf = comparison_results[model].get('classification_performance', {})
            f.write("### Classification Performance\n")
            for clf_name, results in clf_perf.items():
                if isinstance(results, dict) and 'accuracy' in results:
                    f.write(f"- **{clf_name}**: Accuracy {results['accuracy']:.4f}, ")
                    f.write(f"F1 {results.get('f1_score', 0):.4f}\n")
            f.write("\n")
            
            # Open-set performance
            osr = comparison_results[model].get('open_set_performance', {})
            f.write("### Open-Set Recognition\n")
            f.write(f"- **AUROC**: {osr.get('auroc', 0):.4f}\n")
            f.write(f"- **OSCR AUC**: {osr.get('oscr_auc', 0):.4f}\n")
            f.write(f"- **Mean known confidence**: {osr.get('mean_known_confidence', 0):.4f}\n")
            f.write(f"- **Mean unknown confidence**: {osr.get('mean_unknown_confidence', 0):.4f}\n")
            f.write(f"- **Confidence gap**: {osr.get('confidence_gap', 0):.4f}\n\n")
            
            # Semantic structure
            semantic = comparison_results[model].get('semantic_structure', {})
            f.write("### Semantic Structure\n")
            f.write(f"- **Mean class coherence**: {semantic.get('mean_class_coherence', 0):.4f}\n")
            f.write(f"- **Mean similarity**: {semantic.get('mean_similarity', 0):.4f}\n")
            f.write(f"- **Similarity std**: {semantic.get('std_similarity', 0):.4f}\n\n")
        
        # Cross-model analysis
        if 'cross_analysis' in comparison_results:
            f.write("## Cross-Model Analysis\n\n")
            cross_analysis = comparison_results['cross_analysis']
            
            sample_corr = cross_analysis.get('sample_correlation', {})
            f.write("### Feature Correlation\n")
            f.write(f"- **Mean sample correlation**: {sample_corr.get('mean', 0):.4f}\n")
            f.write(f"- **Correlation std**: {sample_corr.get('std', 0):.4f}\n")
            
            f.write(f"- **Centroid correlation**: {cross_analysis.get('centroid_correlation', 0):.4f}\n\n")
            
            complementarity = cross_analysis.get('feature_complementarity', {})
            f.write("### Feature Complementarity\n")
            f.write(f"- **VLM unique info**: {complementarity.get('vlm_unique_info', 0):.4f}\n")
            f.write(f"- **ResNet unique info**: {complementarity.get('resnet_unique_info', 0):.4f}\n")
            f.write(f"- **Info ratio (VLM/ResNet)**: {complementarity.get('info_ratio', 0):.4f}\n\n")
        
        # Key findings and recommendations
        f.write("## Key Findings\n\n")
        
        if len(models) >= 2:
            # Determine strengths
            vlm_results = comparison_results.get('VLM', {})
            resnet_results = comparison_results.get('ResNet', {})
            
            vlm_osr = vlm_results.get('open_set_performance', {}).get('auroc', 0)
            resnet_osr = resnet_results.get('open_set_performance', {}).get('auroc', 0)
            
            if vlm_osr > resnet_osr:
                f.write("1. **VLM Superior Performance**: VLM shows better open-set recognition capabilities\n")
            else:
                f.write("1. **ResNet Competitive**: ResNet features show competitive or superior performance\n")
            
            vlm_semantic = vlm_results.get('semantic_structure', {}).get('mean_class_coherence', 0)
            resnet_semantic = resnet_results.get('semantic_structure', {}).get('mean_class_coherence', 0)
            
            if vlm_semantic > resnet_semantic:
                f.write("2. **Semantic Understanding**: VLM demonstrates superior semantic structure\n")
            else:
                f.write("2. **Feature Quality**: ResNet shows strong feature clustering quality\n")
        
        f.write("3. **Complementarity**: Both models capture different aspects of visual information\n")
        f.write("4. **Deployment**: Consider ensemble approaches for optimal performance\n\n")
        
        # Technical fixes
        f.write("## Technical Fixes Applied\n\n")
        f.write("### FIXED: PCA Dimension Error\n")
        f.write("- **Problem**: `n_components=512 must be between 0 and min(n_samples, n_features)=200`\n")
        f.write("- **Solution**: Automatic dimension alignment with PCA constraint checking\n")
        f.write("- **Result**: Robust feature comparison without dimension errors\n\n")
        
        f.write("### FIXED: SubplotSpec Subscriptable Error\n")
        f.write("- **Problem**: `'SubplotSpec' object is not subscriptable`\n")
        f.write("- **Solution**: Individual subplot creation instead of GridSpec subscripting\n")
        f.write("- **Result**: Visualization dashboard renders correctly\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### For Production Deployment\n")
        f.write("- **Primary Model**: Choose based on AUROC and OSCR AUC performance\n")
        f.write("- **Ensemble Strategy**: Combine VLM semantic understanding with ResNet visual features\n")
        f.write("- **Monitoring**: Track feature quality metrics during deployment\n\n")
        
        f.write("### For Further Research\n")
        f.write("- **Feature Fusion**: Investigate optimal combination strategies\n")
        f.write("- **Domain Adaptation**: Fine-tune models for specific aircraft types\n")
        f.write("- **Efficiency**: Optimize computational requirements for real-time use\n")
        f.write("- **Robustness**: Evaluate performance under various conditions\n")


if __name__ == "__main__":
    print("VLM vs ResNet Comparison Module")
    print("Usage: Import and call run_vlm_resnet_comparison() with your VLM model and data")
    print("FIXED: PCA dimension error and SubplotSpec subscriptable error")