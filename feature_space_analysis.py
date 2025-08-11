"""
Feature Space Analysis Protocol
==============================

Focused analysis on features before and after VLM embedding with emphasis on:
1. Detailed feature space characterization
2. Similarity analysis across feature spaces
3. Open-set scenario evaluation
4. ResNet baseline comparison
5. Classification boundary analysis
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import wasserstein_distance
from typing import Dict, List, Tuple, Optional
import pickle
from collections import defaultdict


class FeatureSpaceAnalyzer:
    """Comprehensive analysis of feature spaces before and after VLM embedding."""
    
    def __init__(self):
        self.feature_spaces = {}
        self.analysis_results = {}
        
    def add_feature_space(self, 
                         name: str, 
                         features: np.ndarray, 
                         labels: np.ndarray,
                         metadata: Dict = None):
        """Add a feature space for analysis."""
        self.feature_spaces[name] = {
            'features': features,
            'labels': labels,
            'metadata': metadata or {}
        }
    
    def compute_intrinsic_dimensionality(self, features: np.ndarray, method: str = 'pca') -> Dict:
        """Estimate intrinsic dimensionality of feature space."""
        
        if method == 'pca':
            pca = PCA()
            pca.fit(features)
            
            # Find number of components explaining 95% variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = np.argmax(cumvar >= 0.95) + 1
            
            return {
                'method': 'pca',
                'intrinsic_dimension': intrinsic_dim,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': cumvar
            }
        
        elif method == 'mle':
            # Maximum Likelihood Estimation (simplified)
            # This is a basic implementation
            from sklearn.neighbors import NearestNeighbors
            
            k = min(20, len(features) // 10)  # Adaptive k
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(features)
            distances, _ = nbrs.kneighbors(features)
            
            # Remove self-distance (first column)
            distances = distances[:, 1:]
            
            # MLE estimate
            log_ratios = np.log(distances[:, -1:] / distances[:, :-1])
            dimension_estimates = 1 / np.mean(log_ratios, axis=1)
            
            return {
                'method': 'mle',
                'intrinsic_dimension': np.mean(dimension_estimates),
                'dimension_std': np.std(dimension_estimates),
                'per_sample_estimates': dimension_estimates
            }
    
    def analyze_class_separability(self, 
                                  features: np.ndarray, 
                                  labels: np.ndarray) -> Dict:
        """Analyze how well classes are separated in the feature space."""
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Compute class centroids
        centroids = {}
        class_features = {}
        
        for label in unique_labels:
            mask = labels == label
            class_feat = features[mask]
            centroids[label] = np.mean(class_feat, axis=0)
            class_features[label] = class_feat
        
        # Inter-class distances
        inter_distances = []
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j:
                    dist = euclidean_distances([centroids[label1]], [centroids[label2]])[0, 0]
                    inter_distances.append(dist)
        
        # Intra-class distances
        intra_distances = []
        for label in unique_labels:
            class_feat = class_features[label]
            centroid = centroids[label]
            
            distances = euclidean_distances(class_feat, [centroid])[:, 0]
            intra_distances.extend(distances)
        
        # Silhouette-like measure
        inter_mean = np.mean(inter_distances)
        intra_mean = np.mean(intra_distances)
        separability_ratio = inter_mean / intra_mean if intra_mean > 0 else np.inf
        
        # Linear Discriminant Analysis (if applicable)
        lda_score = None
        if n_classes > 1 and len(features) > n_classes:
            try:
                lda = LinearDiscriminantAnalysis()
                lda.fit(features, labels)
                lda_score = lda.score(features, labels)
            except:
                lda_score = None
        
        return {
            'n_classes': n_classes,
            'inter_class_distances': inter_distances,
            'intra_class_distances': intra_distances,
            'inter_class_mean': inter_mean,
            'intra_class_mean': intra_mean,
            'separability_ratio': separability_ratio,
            'lda_accuracy': lda_score,
            'centroids': centroids
        }
    
    def compute_feature_space_similarity(self, 
                                       space1_name: str, 
                                       space2_name: str,
                                       method: str = 'cka') -> Dict:
        """Compute similarity between two feature spaces."""
        
        features1 = self.feature_spaces[space1_name]['features']
        features2 = self.feature_spaces[space2_name]['features']
        labels1 = self.feature_spaces[space1_name]['labels']
        labels2 = self.feature_spaces[space2_name]['labels']
        
        # Ensure same number of samples
        min_samples = min(len(features1), len(features2))
        features1 = features1[:min_samples]
        features2 = features2[:min_samples]
        labels1 = labels1[:min_samples]
        labels2 = labels2[:min_samples]
        
        if method == 'cka':
            # Centered Kernel Alignment
            similarity = self._compute_cka(features1, features2)
            
        elif method == 'procrustes':
            # Procrustes analysis
            from scipy.spatial.distance import procrustes
            _, _, disparity = procrustes(features1, features2)
            similarity = 1 - disparity  # Convert disparity to similarity
            
        elif method == 'canonical_correlation':
            # Canonical Correlation Analysis
            from sklearn.cross_decomposition import CCA
            
            min_components = min(features1.shape[1], features2.shape[1], len(features1))
            n_components = min(10, min_components)  # Limit to prevent overfitting
            
            cca = CCA(n_components=n_components)
            try:
                cca.fit(features1, features2)
                X_c, Y_c = cca.transform(features1, features2)
                
                # Compute correlations
                correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] 
                               for i in range(n_components)]
                similarity = np.mean(correlations)
            except:
                similarity = 0.0
        
        elif method == 'mutual_information':
            # Mutual information between class predictions
            from sklearn.metrics import mutual_info_score
            from sklearn.neighbors import KNeighborsClassifier
            
            # Train classifiers on each feature space
            knn1 = KNeighborsClassifier(n_neighbors=5)
            knn2 = KNeighborsClassifier(n_neighbors=5)
            
            knn1.fit(features1, labels1)
            knn2.fit(features2, labels2)
            
            # Get predictions
            pred1 = knn1.predict(features1)
            pred2 = knn2.predict(features2)
            
            # Compute mutual information
            similarity = mutual_info_score(pred1, pred2)
        
        return {
            'method': method,
            'similarity': similarity,
            'space1': space1_name,
            'space2': space2_name
        }
    
    def _compute_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Centered Kernel Alignment between two feature matrices."""
        
        def linear_kernel(X):
            return X @ X.T
        
        def center_kernel(K):
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return H @ K @ H
        
        K_X = center_kernel(linear_kernel(X))
        K_Y = center_kernel(linear_kernel(Y))
        
        numerator = np.trace(K_X @ K_Y)
        denominator = np.sqrt(np.trace(K_X @ K_X) * np.trace(K_Y @ K_Y))
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def analyze_open_set_discriminability(self, 
                                        features: np.ndarray,
                                        labels: np.ndarray,
                                        known_classes: List[str]) -> Dict:
        """Analyze how well open-set scenarios can be handled."""
        
        # Separate known and unknown samples
        known_mask = np.array([label in known_classes for label in labels])
        unknown_mask = ~known_mask
        
        known_features = features[known_mask]
        unknown_features = features[unknown_mask]
        known_labels = labels[known_mask]
        
        if len(unknown_features) == 0:
            return {'error': 'No unknown samples found'}
        
        # Compute known class statistics
        known_stats = self.analyze_class_separability(known_features, known_labels)
        
        # Distance from unknown samples to known class centroids
        known_centroids = known_stats['centroids']
        
        unknown_to_known_distances = []
        for unknown_feat in unknown_features:
            min_dist = np.inf
            for centroid in known_centroids.values():
                dist = euclidean_distances([unknown_feat], [centroid])[0, 0]
                min_dist = min(min_dist, dist)
            unknown_to_known_distances.append(min_dist)
        
        # Compare with known intra-class distances
        known_intra_distances = known_stats['intra_class_distances']
        
        # Compute separation metrics
        unknown_mean_dist = np.mean(unknown_to_known_distances)
        known_intra_mean = np.mean(known_intra_distances)
        
        # Overlap analysis
        unknown_percentile_95 = np.percentile(unknown_to_known_distances, 5)  # Closest unknowns
        known_percentile_95 = np.percentile(known_intra_distances, 95)  # Farthest knowns
        
        separability = unknown_percentile_95 > known_percentile_95
        
        return {
            'known_classes_stats': known_stats,
            'unknown_to_known_distances': unknown_to_known_distances,
            'unknown_mean_distance': unknown_mean_dist,
            'known_intra_mean_distance': known_intra_mean,
            'separation_ratio': unknown_mean_dist / known_intra_mean,
            'is_separable': separability,
            'overlap_margin': unknown_percentile_95 - known_percentile_95
        }
    
    def compute_feature_importance_analysis(self, 
                                          features: np.ndarray,
                                          labels: np.ndarray) -> Dict:
        """Analyze which feature dimensions are most important."""
        
        from sklearn.feature_selection import mutual_info_classif, f_classif
        from sklearn.ensemble import RandomForestClassifier
        
        # Mutual information
        mi_scores = mutual_info_classif(features, labels, random_state=42)
        
        # F-statistic
        f_scores, f_pvals = f_classif(features, labels)
        
        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        rf_importance = rf.feature_importances_
        
        # Variance analysis
        feature_variances = np.var(features, axis=0)
        
        return {
            'mutual_information': mi_scores,
            'f_statistics': f_scores,
            'f_pvalues': f_pvals,
            'random_forest_importance': rf_importance,
            'feature_variances': feature_variances,
            'top_mi_features': np.argsort(mi_scores)[-10:],
            'top_rf_features': np.argsort(rf_importance)[-10:]
        }
    
    def generate_feature_space_comparison_report(self, 
                                                space_names: List[str],
                                                known_classes: List[str] = None) -> Dict:
        """Generate comprehensive comparison report for multiple feature spaces."""
        
        report = {
            'spaces_analyzed': space_names,
            'intrinsic_dimensionality': {},
            'class_separability': {},
            'feature_importance': {},
            'pairwise_similarities': {},
            'open_set_analysis': {}
        }
        
        # Analyze each space individually
        for space_name in space_names:
            features = self.feature_spaces[space_name]['features']
            labels = self.feature_spaces[space_name]['labels']
            
            print(f"Analyzing {space_name}...")
            
            # Intrinsic dimensionality
            report['intrinsic_dimensionality'][space_name] = {
                'pca': self.compute_intrinsic_dimensionality(features, 'pca'),
                'mle': self.compute_intrinsic_dimensionality(features, 'mle')
            }
            
            # Class separability
            report['class_separability'][space_name] = self.analyze_class_separability(features, labels)
            
            # Feature importance
            report['feature_importance'][space_name] = self.compute_feature_importance_analysis(features, labels)
            
            # Open-set analysis (if known classes provided)
            if known_classes:
                report['open_set_analysis'][space_name] = self.analyze_open_set_discriminability(
                    features, labels, known_classes
                )
        
        # Pairwise similarities between spaces
        for i, space1 in enumerate(space_names):
            for j, space2 in enumerate(space_names):
                if i < j:
                    pair_key = f"{space1}_vs_{space2}"
                    
                    # Multiple similarity measures
                    similarities = {}
                    for method in ['cka', 'procrustes', 'canonical_correlation', 'mutual_information']:
                        try:
                            sim_result = self.compute_feature_space_similarity(space1, space2, method)
                            similarities[method] = sim_result['similarity']
                        except Exception as e:
                            print(f"Error computing {method} for {pair_key}: {e}")
                            similarities[method] = None
                    
                    report['pairwise_similarities'][pair_key] = similarities
        
        return report
    
    def visualize_feature_spaces(self, 
                                space_names: List[str],
                                max_samples: int = 1000,
                                save_path: str = None) -> plt.Figure:
        """Create comprehensive visualization of feature spaces."""
        
        n_spaces = len(space_names)
        fig, axes = plt.subplots(2, n_spaces, figsize=(5*n_spaces, 10))
        
        if n_spaces == 1:
            axes = axes.reshape(-1, 1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, space_name in enumerate(space_names):
            features = self.feature_spaces[space_name]['features']
            labels = self.feature_spaces[space_name]['labels']
            
            # Subsample if too many points
            if len(features) > max_samples:
                indices = np.random.choice(len(features), max_samples, replace=False)
                features = features[indices]
                labels = labels[indices]
            
            # PCA visualization
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features)
            
            unique_labels = np.unique(labels)
            for j, label in enumerate(unique_labels):
                mask = labels == label
                axes[0, i].scatter(features_pca[mask, 0], features_pca[mask, 1], 
                                 c=[colors[j % len(colors)]], label=label, alpha=0.6, s=20)
            
            axes[0, i].set_title(f'{space_name} - PCA\n'
                               f'Var explained: {pca.explained_variance_ratio_[:2].sum():.3f}')
            axes[0, i].set_xlabel('PC1')
            axes[0, i].set_ylabel('PC2')
            axes[0, i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # t-SNE visualization
            if len(features) > 50:  # t-SNE needs sufficient samples
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
                features_tsne = tsne.fit_transform(features)
                
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    axes[1, i].scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                                     c=[colors[j % len(colors)]], label=label, alpha=0.6, s=20)
                
                axes[1, i].set_title(f'{space_name} - t-SNE')
                axes[1, i].set_xlabel('t-SNE 1')
                axes[1, i].set_ylabel('t-SNE 2')
            else:
                axes[1, i].text(0.5, 0.5, 'Insufficient samples\nfor t-SNE', 
                              ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{space_name} - t-SNE (N/A)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_similarity_heatmap(self, 
                                 space_names: List[str],
                                 similarity_method: str = 'cka',
                                 save_path: str = None) -> plt.Figure:
        """Create heatmap of similarities between feature spaces."""
        
        n_spaces = len(space_names)
        similarity_matrix = np.eye(n_spaces)
        
        # Compute pairwise similarities
        for i, space1 in enumerate(space_names):
            for j, space2 in enumerate(space_names):
                if i != j:
                    try:
                        sim_result = self.compute_feature_space_similarity(space1, space2, similarity_method)
                        similarity_matrix[i, j] = sim_result['similarity']
                    except:
                        similarity_matrix[i, j] = 0.0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(similarity_matrix, 
                   xticklabels=space_names,
                   yticklabels=space_names,
                   annot=True,
                   fmt='.3f',
                   cmap='viridis',
                   vmin=0, vmax=1,
                   ax=ax)
        
        ax.set_title(f'Feature Space Similarity ({similarity_method.upper()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_classification_boundaries(self, 
                                        space_name: str,
                                        classifier_type: str = 'svm') -> Dict:
        """Analyze decision boundaries in feature space."""
        
        features = self.feature_spaces[space_name]['features']
        labels = self.feature_spaces[space_name]['labels']
        
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        
        # Choose classifier
        if classifier_type == 'svm':
            clf = SVC(kernel='linear', random_state=42)
        elif classifier_type == 'logistic':
            clf = LogisticRegression(random_state=42)
        elif classifier_type == 'tree':
            clf = DecisionTreeClassifier(random_state=42)
        
        # Fit classifier
        clf.fit(features, labels)
        
        # Get decision boundary complexity
        predictions = clf.predict(features)
        decision_values = None
        
        if hasattr(clf, 'decision_function'):
            decision_values = clf.decision_function(features)
        elif hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(features)
            decision_values = np.max(probs, axis=1)
        
        # Analyze boundary complexity through confidence distribution
        confidence_stats = {
            'mean_confidence': np.mean(decision_values) if decision_values is not None else None,
            'std_confidence': np.std(decision_values) if decision_values is not None else None,
            'training_accuracy': np.mean(predictions == labels)
        }
        
        # Analyze support vectors (for SVM)
        support_info = {}
        if classifier_type == 'svm' and hasattr(clf, 'support_'):
            support_info = {
                'n_support_vectors': len(clf.support_),
                'support_ratio': len(clf.support_) / len(features),
                'support_indices': clf.support_
            }
        
        return {
            'classifier_type': classifier_type,
            'confidence_stats': confidence_stats,
            'support_info': support_info,
            'decision_values': decision_values,
            'predictions': predictions
        }


class ComparisonProtocol:
    """Protocol for systematic comparison between feature spaces."""
    
    def __init__(self):
        self.analyzer = FeatureSpaceAnalyzer()
        self.results = {}
    
    def setup_vlm_comparison(self, 
                           clip_pre_features: np.ndarray,
                           clip_embeddings: np.ndarray, 
                           resnet_features: np.ndarray,
                           labels: np.ndarray,
                           known_classes: List[str] = None):
        """Set up comparison between VLM pre/post embedding and ResNet features."""
        
        # Add feature spaces
        self.analyzer.add_feature_space('CLIP_Pre_Embedding', clip_pre_features, labels,
                                      {'description': 'CLIP features before final projection'})
        
        self.analyzer.add_feature_space('CLIP_Embeddings', clip_embeddings, labels,
                                      {'description': 'CLIP features after projection/normalization'})
        
        self.analyzer.add_feature_space('ResNet_Features', resnet_features, labels,
                                      {'description': 'ResNet CNN features'})
        
        # Store known classes for open-set analysis
        self.known_classes = known_classes or []
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive analysis across all feature spaces."""
        
        space_names = ['CLIP_Pre_Embedding', 'CLIP_Embeddings', 'ResNet_Features']
        
        # Generate comparison report
        self.results['comparison_report'] = self.analyzer.generate_feature_space_comparison_report(
            space_names, self.known_classes
        )
        
        # Analyze classification boundaries for each space
        self.results['boundary_analysis'] = {}
        for space_name in space_names:
            self.results['boundary_analysis'][space_name] = {}
            for clf_type in ['svm', 'logistic']:
                self.results['boundary_analysis'][space_name][clf_type] = \
                    self.analyzer.analyze_classification_boundaries(space_name, clf_type)
        
        return self.results
    
    def generate_visualizations(self, output_dir: str = 'feature_analysis_results'):
        """Generate all visualization plots."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        space_names = ['CLIP_Pre_Embedding', 'CLIP_Embeddings', 'ResNet_Features']
        
        # Feature space visualization
        self.analyzer.visualize_feature_spaces(
            space_names, 
            save_path=os.path.join(output_dir, 'feature_spaces_visualization.png')
        )
        
        # Similarity heatmaps for different methods
        for method in ['cka', 'procrustes', 'canonical_correlation']:
            try:
                self.analyzer.create_similarity_heatmap(
                    space_names, 
                    similarity_method=method,
                    save_path=os.path.join(output_dir, f'similarity_heatmap_{method}.png')
                )
            except Exception as e:
                print(f"Error creating {method} heatmap: {e}")
    
    def create_summary_plots(self, output_dir: str = 'feature_analysis_results'):
        """Create summary comparison plots."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report = self.results['comparison_report']
        space_names = ['CLIP_Pre_Embedding', 'CLIP_Embeddings', 'ResNet_Features']
        
        # 1. Intrinsic dimensionality comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        pca_dims = [report['intrinsic_dimensionality'][space]['pca']['intrinsic_dimension'] 
                   for space in space_names]
        mle_dims = [report['intrinsic_dimensionality'][space]['mle']['intrinsic_dimension'] 
                   for space in space_names]
        
        x = np.arange(len(space_names))
        ax1.bar(x - 0.2, pca_dims, 0.4, label='PCA (95% variance)', alpha=0.8)
        ax1.bar(x + 0.2, mle_dims, 0.4, label='MLE estimate', alpha=0.8)
        ax1.set_xlabel('Feature Space')
        ax1.set_ylabel('Intrinsic Dimension')
        ax1.set_title('Intrinsic Dimensionality Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(space_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Separability comparison
        sep_ratios = [report['class_separability'][space]['separability_ratio'] 
                     for space in space_names]
        lda_scores = [report['class_separability'][space]['lda_accuracy'] or 0
                     for space in space_names]
        
        ax2.bar(x - 0.2, sep_ratios, 0.4, label='Inter/Intra distance ratio', alpha=0.8)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + 0.2, lda_scores, 0.4, label='LDA accuracy', alpha=0.8, color='orange')
        
        ax2.set_xlabel('Feature Space')
        ax2.set_ylabel('Separability Ratio', color='blue')
        ax2_twin.set_ylabel('LDA Accuracy', color='orange')
        ax2.set_title('Class Separability Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(space_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dimensionality_separability_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Open-set discriminability (if available)
        if 'open_set_analysis' in report and any(report['open_set_analysis']):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ['separation_ratio', 'overlap_margin']
            metric_data = {metric: [] for metric in metrics}
            valid_spaces = []
            
            for space in space_names:
                if space in report['open_set_analysis'] and 'error' not in report['open_set_analysis'][space]:
                    analysis = report['open_set_analysis'][space]
                    valid_spaces.append(space)
                    metric_data['separation_ratio'].append(analysis['separation_ratio'])
                    metric_data['overlap_margin'].append(analysis['overlap_margin'])
            
            if valid_spaces:
                x = np.arange(len(valid_spaces))
                width = 0.35
                
                ax.bar(x - width/2, metric_data['separation_ratio'], width, 
                      label='Unknown/Known distance ratio', alpha=0.8)
                
                ax_twin = ax.twinx()
                ax_twin.bar(x + width/2, metric_data['overlap_margin'], width,
                           label='Overlap margin', alpha=0.8, color='red')
                
                ax.set_xlabel('Feature Space')
                ax.set_ylabel('Distance Ratio', color='blue')
                ax_twin.set_ylabel('Overlap Margin', color='red')
                ax.set_title('Open-Set Discriminability')
                ax.set_xticks(x)
                ax.set_xticklabels(valid_spaces, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'open_set_discriminability.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def save_results(self, output_dir: str = 'feature_analysis_results'):
        """Save analysis results."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as pickle
        with open(os.path.join(output_dir, 'feature_analysis_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save as JSON (excluding non-serializable objects)
        import json
        json_results = {}
        
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        json_results = make_serializable(self.results)
        
        with open(os.path.join(output_dir, 'feature_analysis_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {output_dir}")


def run_feature_space_analysis_example():
    """Example of how to run the feature space analysis."""
    
    # This is a template - replace with your actual data loading
    
    # Simulated data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Generate simulated features (replace with your actual feature extraction)
    clip_pre_features = np.random.randn(n_samples, 768)  # CLIP pre-embedding features
    clip_embeddings = np.random.randn(n_samples, 512)   # CLIP final embeddings
    resnet_features = np.random.randn(n_samples, 2048)  # ResNet features
    
    # Generate labels
    labels = np.random.choice(['Boeing', 'Airbus', 'Cessna', 'Bombardier', 'Embraer'], 
                             n_samples)
    
    # Define known classes for open-set analysis
    known_classes = ['Boeing', 'Airbus']
    
    # Set up comparison protocol
    protocol = ComparisonProtocol()
    protocol.setup_vlm_comparison(
        clip_pre_features, clip_embeddings, resnet_features, 
        labels, known_classes
    )
    
    # Run analysis
    print("Running comprehensive feature space analysis...")
    results = protocol.run_comprehensive_analysis()
    
    # Generate visualizations
    print("Generating visualizations...")
    protocol.generate_visualizations()
    protocol.create_summary_plots()
    
    # Save results
    print("Saving results...")
    protocol.save_results()
    
    print("Analysis complete! Check 'feature_analysis_results' directory for outputs.")
    
    return results


if __name__ == "__main__":
    # Run example analysis
    results = run_feature_space_analysis_example()
    
    # Print summary
    print("\n=== FEATURE SPACE ANALYSIS SUMMARY ===")
    
    if 'comparison_report' in results:
        report = results['comparison_report']
        
        print("\nIntrinsic Dimensionality:")
        for space in report['intrinsic_dimensionality']:
            pca_dim = report['intrinsic_dimensionality'][space]['pca']['intrinsic_dimension']
            mle_dim = report['intrinsic_dimensionality'][space]['mle']['intrinsic_dimension']
            print(f"  {space}: PCA={pca_dim}, MLE={mle_dim:.1f}")
        
        print("\nClass Separability (Inter/Intra distance ratio):")
        for space in report['class_separability']:
            ratio = report['class_separability'][space]['separability_ratio']
            lda_acc = report['class_separability'][space]['lda_accuracy']
            print(f"  {space}: {ratio:.3f} (LDA accuracy: {lda_acc:.3f})")
        
        print("\nPairwise Similarities (CKA):")
        for pair, similarities in report['pairwise_similarities'].items():
            cka_sim = similarities.get('cka', 'N/A')
            print(f"  {pair}: {cka_sim:.3f}")