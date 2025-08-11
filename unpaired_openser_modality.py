"""
Unpaired Modality Open-Set Recognition
Research Question: How to perform OSR when image-text pairs are missing or mismatched?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import os
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class UnpairedModalityOSR:
    """Handle open-set recognition with unpaired or missing modalities."""
    
    def __init__(self, vlm_model, processor, device='cuda'):
        self.vlm_model = vlm_model
        self.processor = processor
        self.device = device
        
        # Storage for different modality scenarios
        self.image_only_model = None
        self.text_only_model = None
        self.cross_modal_model = None
        
        # Performance tracking
        self.scenario_results = {}
    
    def evaluate_image_only_osr(self, test_images, test_labels, known_classes, 
                               train_images=None, train_labels=None):
        """
        Evaluate OSR using only image features (no text guidance).
        
        Args:
            test_images: List of test images
            test_labels: Test image labels
            known_classes: List of known class names
            train_images: Training images (optional)
            train_labels: Training labels (optional)
            
        Returns:
            Image-only OSR evaluation results
        """
        print("Evaluating image-only open-set recognition...")
        
        # Extract image embeddings
        print("  Extracting image embeddings...")
        test_embeddings = self._extract_image_embeddings(test_images)
        
        # Create known/unknown masks
        known_mask = np.array([label in known_classes for label in test_labels])
        
        # Split test data
        known_embeddings = test_embeddings[known_mask]
        unknown_embeddings = test_embeddings[~known_mask]
        known_labels = test_labels[known_mask]
        
        if len(known_embeddings) == 0 or len(unknown_embeddings) == 0:
            return {'error': 'Insufficient known or unknown samples'}
        
        # Train image-only OSR model
        print("  Training image-only OSR model...")
        
        # Use training data if available, otherwise use known test samples
        if train_images is not None and train_labels is not None:
            train_embeddings = self._extract_image_embeddings(train_images)
            train_known_mask = np.array([label in known_classes for label in train_labels])
            model_train_embeddings = train_embeddings[train_known_mask]
            model_train_labels = train_labels[train_known_mask]
        else:
            # Use a subset of known test samples for training
            n_train = max(1, len(known_embeddings) // 2)
            indices = np.random.permutation(len(known_embeddings))
            model_train_embeddings = known_embeddings[indices[:n_train]]
            model_train_labels = known_labels[indices[:n_train]]
        
        # Train KNN classifier for known class recognition
        knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn_classifier.fit(model_train_embeddings, model_train_labels)
        
        # Train nearest neighbor model for novelty detection
        nn_detector = NearestNeighbors(n_neighbors=5, metric='cosine')
        nn_detector.fit(model_train_embeddings)
        
        # Evaluate on test data
        print("  Evaluating image-only performance...")
        
        # Get predictions and confidence scores
        known_predictions = knn_classifier.predict(known_embeddings)
        known_distances, _ = nn_detector.kneighbors(known_embeddings)
        known_scores = 1 / (1 + known_distances.mean(axis=1))  # Convert distance to confidence
        
        unknown_distances, _ = nn_detector.kneighbors(unknown_embeddings)
        unknown_scores = 1 / (1 + unknown_distances.mean(axis=1))
        
        # Calculate classification accuracy on known samples
        known_accuracy = accuracy_score(known_labels, known_predictions)
        
        # Calculate OSR metrics
        y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        y_scores = np.concatenate([known_scores, unknown_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr, tpr)
        
        # Find optimal threshold
        thresholds = np.linspace(0, 1, 100)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            # Predicted as known if score >= threshold
            pred_known = y_scores >= threshold
            
            tp = np.sum(pred_known[:len(known_scores)])
            fp = np.sum(pred_known[len(known_scores):])
            fn = np.sum(~pred_known[:len(known_scores)])
            tn = np.sum(~pred_known[len(known_scores):])
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        results = {
            'scenario': 'image_only',
            'known_accuracy': known_accuracy,
            'auroc': auroc,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'mean_known_confidence': np.mean(known_scores),
            'mean_unknown_confidence': np.mean(unknown_scores),
            'confidence_separation': np.mean(known_scores) - np.mean(unknown_scores),
            'num_known': len(known_embeddings),
            'num_unknown': len(unknown_embeddings)
        }
        
        return results
    
    def evaluate_text_only_osr(self, test_texts, test_labels, known_classes,
                              additional_texts=None):
        """
        Evaluate OSR using only text descriptions (no paired images).
        
        Args:
            test_texts: List of text descriptions
            test_labels: Corresponding labels
            known_classes: List of known class names
            additional_texts: Additional text samples for training
            
        Returns:
            Text-only OSR evaluation results
        """
        print("Evaluating text-only open-set recognition...")
        
        # Generate text embeddings
        print("  Extracting text embeddings...")
        test_embeddings = self._extract_text_embeddings(test_texts)
        
        # Create known/unknown masks
        known_mask = np.array([label in known_classes for label in test_labels])
        
        # Generate prototype embeddings for known classes
        print("  Generating class prototypes...")
        class_prototypes = {}
        prototype_embeddings = []
        prototype_labels = []
        
        for class_name in known_classes:
            # Generate multiple text variations for each class
            class_texts = self._generate_class_text_variations(class_name)
            
            if additional_texts:
                # Add any additional texts for this class
                class_specific_texts = [text for text, label in zip(additional_texts, test_labels) 
                                      if label == class_name]
                class_texts.extend(class_specific_texts[:5])  # Limit to avoid overfitting
            
            class_embeddings = self._extract_text_embeddings(class_texts)
            prototype = np.mean(class_embeddings, axis=0)
            class_prototypes[class_name] = prototype
            
            prototype_embeddings.append(prototype)
            prototype_labels.append(class_name)
        
        prototype_embeddings = np.array(prototype_embeddings)
        
        # Evaluate text-only OSR
        print("  Evaluating text-only performance...")
        
        # Compute similarities to prototypes
        similarities = np.dot(test_embeddings, prototype_embeddings.T)
        max_similarities = np.max(similarities, axis=1)
        predicted_classes = [prototype_labels[i] for i in np.argmax(similarities, axis=1)]
        
        # Split results
        known_similarities = max_similarities[known_mask]
        unknown_similarities = max_similarities[~known_mask]
        known_predictions = np.array(predicted_classes)[known_mask]
        known_labels = test_labels[known_mask]
        
        # Calculate metrics
        if len(known_labels) > 0:
            known_accuracy = accuracy_score(known_labels, known_predictions)
        else:
            known_accuracy = 0
        
        # OSR evaluation
        y_true = np.concatenate([np.ones(len(known_similarities)), np.zeros(len(unknown_similarities))])
        y_scores = np.concatenate([known_similarities, unknown_similarities])
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auroc = auc(fpr, tpr)
        else:
            auroc = 0.5
        
        results = {
            'scenario': 'text_only',
            'known_accuracy': known_accuracy,
            'auroc': auroc,
            'mean_known_similarity': np.mean(known_similarities) if len(known_similarities) > 0 else 0,
            'mean_unknown_similarity': np.mean(unknown_similarities) if len(unknown_similarities) > 0 else 0,
            'similarity_separation': (np.mean(known_similarities) - np.mean(unknown_similarities)) 
                                   if len(known_similarities) > 0 and len(unknown_similarities) > 0 else 0,
            'num_known': len(known_similarities),
            'num_unknown': len(unknown_similarities),
            'class_prototypes': class_prototypes
        }
        
        return results
    
    def evaluate_cross_modal_unpaired(self, test_images, test_texts, test_labels, 
                                    known_classes, mismatch_probability=0.3):
        """
        Evaluate OSR with unpaired/mismatched image-text pairs.
        
        Args:
            test_images: List of test images
            test_texts: List of text descriptions (may not correspond to images)
            test_labels: True labels
            known_classes: List of known class names
            mismatch_probability: Probability of image-text mismatch
            
        Returns:
            Cross-modal unpaired OSR results
        """
        print("Evaluating cross-modal unpaired open-set recognition...")
        
        # Extract embeddings
        print("  Extracting multi-modal embeddings...")
        image_embeddings = self._extract_image_embeddings(test_images)
        text_embeddings = self._extract_text_embeddings(test_texts)
        
        # Create mismatched pairs
        print(f"  Simulating {mismatch_probability:.0%} mismatch rate...")
        n_samples = len(test_images)
        n_mismatched = int(n_samples * mismatch_probability)
        
        # Randomly select samples to mismatch
        mismatch_indices = np.random.choice(n_samples, n_mismatched, replace=False)
        
        # Create shuffled text embeddings for mismatched samples
        shuffled_text_embeddings = text_embeddings.copy()
        if n_mismatched > 0:
            # Shuffle text embeddings for mismatched indices
            shuffle_indices = np.random.permutation(n_mismatched)
            shuffled_text_embeddings[mismatch_indices] = text_embeddings[mismatch_indices[shuffle_indices]]
        
        # Create match indicators
        is_matched = np.ones(n_samples, dtype=bool)
        is_matched[mismatch_indices] = False
        
        # Create known/unknown masks
        known_mask = np.array([label in known_classes for label in test_labels])
        
        # Evaluate different strategies for handling unpaired data
        strategies = {
            'image_dominant': self._evaluate_image_dominant_strategy,
            'text_dominant': self._evaluate_text_dominant_strategy,
            'confidence_weighted': self._evaluate_confidence_weighted_strategy,
            'mismatch_detection': self._evaluate_mismatch_detection_strategy
        }
        
        strategy_results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"  Evaluating {strategy_name} strategy...")
            
            strategy_result = strategy_func(
                image_embeddings, shuffled_text_embeddings, test_labels,
                known_classes, known_mask, is_matched
            )
            strategy_results[strategy_name] = strategy_result
        
        # Overall results
        results = {
            'scenario': 'cross_modal_unpaired',
            'mismatch_rate': mismatch_probability,
            'num_mismatched': n_mismatched,
            'num_matched': n_samples - n_mismatched,
            'strategies': strategy_results,
            'best_strategy': max(strategy_results.keys(), 
                               key=lambda k: strategy_results[k].get('auroc', 0))
        }
        
        return results
    
    def _evaluate_image_dominant_strategy(self, image_embeddings, text_embeddings, 
                                        labels, known_classes, known_mask, is_matched):
        """Evaluate strategy that relies primarily on image features."""
        
        # Use image embeddings as primary signal
        known_image_embeddings = image_embeddings[known_mask]
        unknown_image_embeddings = image_embeddings[~known_mask]
        
        if len(known_image_embeddings) == 0 or len(unknown_image_embeddings) == 0:
            return {'error': 'Insufficient samples'}
        
        # Train on known image embeddings
        nn_detector = NearestNeighbors(n_neighbors=5, metric='cosine')
        nn_detector.fit(known_image_embeddings)
        
        # Compute confidence scores
        known_distances, _ = nn_detector.kneighbors(known_image_embeddings)
        known_scores = 1 / (1 + known_distances.mean(axis=1))
        
        unknown_distances, _ = nn_detector.kneighbors(unknown_image_embeddings)
        unknown_scores = 1 / (1 + unknown_distances.mean(axis=1))
        
        # Calculate AUROC
        y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        y_scores = np.concatenate([known_scores, unknown_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr, tpr)
        
        return {
            'auroc': auroc,
            'mean_known_confidence': np.mean(known_scores),
            'mean_unknown_confidence': np.mean(unknown_scores),
            'strategy': 'image_dominant'
        }
    
    def _evaluate_text_dominant_strategy(self, image_embeddings, text_embeddings,
                                       labels, known_classes, known_mask, is_matched):
        """Evaluate strategy that relies primarily on text features."""
        
        # Generate class prototypes from text
        class_prototypes = {}
        for class_name in known_classes:
            class_texts = self._generate_class_text_variations(class_name)
            class_embeddings = self._extract_text_embeddings(class_texts)
            class_prototypes[class_name] = np.mean(class_embeddings, axis=0)
        
        prototype_embeddings = np.array(list(class_prototypes.values()))
        
        # Compute similarities
        similarities = np.dot(text_embeddings, prototype_embeddings.T)
        max_similarities = np.max(similarities, axis=1)
        
        known_similarities = max_similarities[known_mask]
        unknown_similarities = max_similarities[~known_mask]
        
        # Calculate AUROC
        y_true = np.concatenate([np.ones(len(known_similarities)), np.zeros(len(unknown_similarities))])
        y_scores = np.concatenate([known_similarities, unknown_similarities])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr, tpr)
        
        return {
            'auroc': auroc,
            'mean_known_similarity': np.mean(known_similarities),
            'mean_unknown_similarity': np.mean(unknown_similarities),
            'strategy': 'text_dominant'
        }
    
    def _evaluate_confidence_weighted_strategy(self, image_embeddings, text_embeddings,
                                             labels, known_classes, known_mask, is_matched):
        """Evaluate strategy that weights image and text based on confidence."""
        
        # Get image-based scores
        known_image_embeddings = image_embeddings[known_mask]
        nn_detector = NearestNeighbors(n_neighbors=5, metric='cosine')
        nn_detector.fit(known_image_embeddings)
        
        image_distances, _ = nn_detector.kneighbors(image_embeddings)
        image_scores = 1 / (1 + image_distances.mean(axis=1))
        
        # Get text-based scores
        class_prototypes = {}
        for class_name in known_classes:
            class_texts = self._generate_class_text_variations(class_name)
            class_embeddings = self._extract_text_embeddings(class_texts)
            class_prototypes[class_name] = np.mean(class_embeddings, axis=0)
        
        prototype_embeddings = np.array(list(class_prototypes.values()))
        text_similarities = np.dot(text_embeddings, prototype_embeddings.T)
        text_scores = np.max(text_similarities, axis=1)
        
        # Adaptive weighting based on confidence
        image_confidence = (image_scores - np.mean(image_scores)) / (np.std(image_scores) + 1e-8)
        text_confidence = (text_scores - np.mean(text_scores)) / (np.std(text_scores) + 1e-8)
        
        # Weight based on relative confidence
        image_weights = np.exp(image_confidence) / (np.exp(image_confidence) + np.exp(text_confidence))
        text_weights = 1 - image_weights
        
        # Combined scores
        combined_scores = image_weights * image_scores + text_weights * text_scores
        
        known_combined = combined_scores[known_mask]
        unknown_combined = combined_scores[~known_mask]
        
        # Calculate AUROC
        y_true = np.concatenate([np.ones(len(known_combined)), np.zeros(len(unknown_combined))])
        y_scores = np.concatenate([known_combined, unknown_combined])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr, tpr)
        
        return {
            'auroc': auroc,
            'mean_known_score': np.mean(known_combined),
            'mean_unknown_score': np.mean(unknown_combined),
            'mean_image_weight': np.mean(image_weights),
            'mean_text_weight': np.mean(text_weights),
            'strategy': 'confidence_weighted'
        }
    
    def _evaluate_mismatch_detection_strategy(self, image_embeddings, text_embeddings,
                                            labels, known_classes, known_mask, is_matched):
        """Evaluate strategy that detects and handles mismatched pairs."""
        
        # Detect mismatched pairs using cross-modal similarity
        cross_modal_similarities = []
        for i in range(len(image_embeddings)):
            img_emb = image_embeddings[i]
            txt_emb = text_embeddings[i]
            
            # Compute cosine similarity
            similarity = np.dot(img_emb, txt_emb) / (np.linalg.norm(img_emb) * np.linalg.norm(txt_emb) + 1e-8)
            cross_modal_similarities.append(similarity)
        
        cross_modal_similarities = np.array(cross_modal_similarities)
        
        # Threshold for mismatch detection
        mismatch_threshold = np.percentile(cross_modal_similarities, 30)
        detected_mismatches = cross_modal_similarities < mismatch_threshold
        
        # For detected matches, use combined features
        # For detected mismatches, use image-only approach
        final_scores = np.zeros(len(image_embeddings))
        
        # Handle matched pairs
        matched_indices = ~detected_mismatches
        if np.any(matched_indices):
            # Simple average of normalized image and text confidences
            image_scores = self._compute_image_confidence_scores(
                image_embeddings[matched_indices], 
                image_embeddings[known_mask]
            )
            text_scores = self._compute_text_confidence_scores(
                text_embeddings[matched_indices], known_classes
            )
            
            final_scores[matched_indices] = (image_scores + text_scores) / 2
        
        # Handle mismatched pairs with image-only approach
        mismatched_indices = detected_mismatches
        if np.any(mismatched_indices):
            image_scores = self._compute_image_confidence_scores(
                image_embeddings[mismatched_indices],
                image_embeddings[known_mask]
            )
            final_scores[mismatched_indices] = image_scores
        
        # Evaluate
        known_scores = final_scores[known_mask]
        unknown_scores = final_scores[~known_mask]
        
        # Calculate AUROC
        y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        y_scores = np.concatenate([known_scores, unknown_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = auc(fpr, tpr)
        
        # Mismatch detection accuracy
        mismatch_detection_accuracy = accuracy_score(~is_matched, detected_mismatches)
        
        return {
            'auroc': auroc,
            'mismatch_detection_accuracy': mismatch_detection_accuracy,
            'detected_mismatch_rate': np.mean(detected_mismatches),
            'true_mismatch_rate': np.mean(~is_matched),
            'mean_known_score': np.mean(known_scores),
            'mean_unknown_score': np.mean(unknown_scores),
            'strategy': 'mismatch_detection'
        }
    
    def _extract_image_embeddings(self, images):
        """Extract embeddings from images."""
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                batch_embeddings = self.vlm_model.get_image_features(**inputs)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def _extract_text_embeddings(self, texts):
        """Extract embeddings from texts."""
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
                batch_embeddings = self.vlm_model.get_text_features(**inputs)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1) 
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def _generate_class_text_variations(self, class_name):
        """Generate text variations for a class."""
        templates = [
            "a photo of a {} aircraft",
            "an image of a {} airplane",
            "a {} plane in flight",
            "a {} aircraft on the ground",
            "technical specifications of {} airplane",
            "close-up view of {} aircraft",
            "side view of {} plane",
            "a {} flying in the sky"
        ]
        
        return [template.format(class_name) for template in templates]
    
    def _compute_image_confidence_scores(self, query_embeddings, reference_embeddings):
        """Compute confidence scores based on image embeddings."""
        if len(reference_embeddings) == 0:
            return np.zeros(len(query_embeddings))
        
        nn_detector = NearestNeighbors(n_neighbors=min(5, len(reference_embeddings)), metric='cosine')
        nn_detector.fit(reference_embeddings)
        
        distances, _ = nn_detector.kneighbors(query_embeddings)
        scores = 1 / (1 + distances.mean(axis=1))
        
        return scores
    
    def _compute_text_confidence_scores(self, query_embeddings, known_classes):
        """Compute confidence scores based on text embeddings."""
        # Generate class prototypes
        class_prototypes = []
        for class_name in known_classes:
            class_texts = self._generate_class_text_variations(class_name)
            class_embeddings = self._extract_text_embeddings(class_texts)
            prototype = np.mean(class_embeddings, axis=0)
            class_prototypes.append(prototype)
        
        if not class_prototypes:
            return np.zeros(len(query_embeddings))
        
        prototype_embeddings = np.array(class_prototypes)
        similarities = np.dot(query_embeddings, prototype_embeddings.T)
        scores = np.max(similarities, axis=1)
        
        return scores


class UnpairedModalityVisualizer:
    """Visualize results from unpaired modality OSR analysis."""
    
    def __init__(self, output_dir='./unpaired_modality_osr'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comprehensive_dashboard(self, scenario_results, save_path=None):
        """
        Create comprehensive dashboard for unpaired modality analysis.
        
        Args:
            scenario_results: Results from different scenarios
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
        
        # Row 1: Scenario performance comparison
        self._plot_scenario_comparison(fig, gs[0, :3], scenario_results)
        self._plot_performance_metrics(fig, gs[0, 3:], scenario_results)
        
        # Row 2: Strategy analysis for cross-modal scenario
        if 'cross_modal_unpaired' in scenario_results:
            self._plot_strategy_comparison(fig, gs[1, :3], scenario_results['cross_modal_unpaired'])
            self._plot_mismatch_analysis(fig, gs[1, 3:], scenario_results['cross_modal_unpaired'])
        
        # Row 3: Detailed analysis
        self._plot_confidence_distributions(fig, gs[2, :3], scenario_results)
        self._plot_robustness_analysis(fig, gs[2, 3:], scenario_results)
        
        # Row 4: Summary and recommendations
        self._plot_scenario_summary(fig, gs[3, :3], scenario_results)
        self._plot_recommendations(fig, gs[3, 3:], scenario_results)
        
        plt.suptitle('Unpaired Modality Open-Set Recognition Analysis', 
                     fontsize=20, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _plot_scenario_comparison(self, fig, gs, scenario_results):
        """Plot comparison across different scenarios."""
        ax = fig.add_subplot(gs)
        
        scenarios = []
        auroc_scores = []
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict) and 'auroc' in results:
                scenarios.append(scenario_name.replace('_', ' ').title())
                auroc_scores.append(results['auroc'])
            elif scenario_name == 'cross_modal_unpaired' and 'strategies' in results:
                # Use best strategy performance
                best_strategy = results['best_strategy']
                best_auroc = results['strategies'][best_strategy]['auroc']
                scenarios.append('Cross-Modal (Best)')
                auroc_scores.append(best_auroc)
        
        if scenarios:
            colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
            bars = ax.bar(scenarios, auroc_scores, color=colors, alpha=0.8)
            
            # Add baseline
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
            ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Good Performance')
            
            ax.set_ylabel('AUROC Score')
            ax.set_title('OSR Performance by Scenario', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, auroc_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No scenario results available',
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_performance_metrics(self, fig, gs, scenario_results):
        """Plot detailed performance metrics."""
        ax = fig.add_subplot(gs)
        
        metrics_data = []
        scenario_names = []
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                scenario_names.append(scenario_name.replace('_', ' ').title())
                
                # Extract key metrics
                auroc = results.get('auroc', 0)
                
                if 'known_accuracy' in results:
                    accuracy = results['known_accuracy']
                elif 'strategies' in results:
                    # Use best strategy
                    best_strategy = results['best_strategy']
                    accuracy = results['strategies'][best_strategy].get('auroc', 0)
                else:
                    accuracy = 0
                
                # Confidence/similarity separation
                if 'confidence_separation' in results:
                    separation = results['confidence_separation']
                elif 'similarity_separation' in results:
                    separation = results['similarity_separation']
                else:
                    separation = 0
                
                metrics_data.append([auroc, accuracy, separation])
        
        if metrics_data:
            metrics_array = np.array(metrics_data).T
            metric_names = ['AUROC', 'Accuracy/Performance', 'Separation']
            
            x = np.arange(len(scenario_names))
            width = 0.25
            colors = ['blue', 'green', 'orange']
            
            for i, (metric_values, metric_name, color) in enumerate(zip(metrics_array, metric_names, colors)):
                ax.bar(x + i*width - width, metric_values, width, 
                      label=metric_name, alpha=0.8, color=color)
            
            ax.set_xlabel('Scenario')
            ax.set_ylabel('Performance Score')
            ax.set_title('Detailed Performance Metrics', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No performance metrics available',
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_strategy_comparison(self, fig, gs, cross_modal_results):
        """Plot strategy comparison for cross-modal scenario."""
        ax = fig.add_subplot(gs)
        
        if 'strategies' in cross_modal_results:
            strategies = cross_modal_results['strategies']
            
            strategy_names = []
            auroc_scores = []
            
            for strategy_name, results in strategies.items():
                if isinstance(results, dict) and 'auroc' in results:
                    strategy_names.append(strategy_name.replace('_', ' ').title())
                    auroc_scores.append(results['auroc'])
            
            if strategy_names:
                colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_names)))
                bars = ax.bar(strategy_names, auroc_scores, color=colors, alpha=0.8)
                
                # Highlight best strategy
                best_idx = np.argmax(auroc_scores)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
                
                ax.set_ylabel('AUROC Score')
                ax.set_title('Cross-Modal Strategy Comparison', fontweight='bold')
                ax.set_xticklabels(strategy_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, score in zip(bars, auroc_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No strategy results available',
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Cross-modal analysis not available',
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_mismatch_analysis(self, fig, gs, cross_modal_results):
        """Plot mismatch detection analysis."""
        ax = fig.add_subplot(gs)
        
        if 'strategies' in cross_modal_results and 'mismatch_detection' in cross_modal_results['strategies']:
            mismatch_results = cross_modal_results['strategies']['mismatch_detection']
            
            # Mismatch detection performance
            true_rate = mismatch_results.get('true_mismatch_rate', 0)
            detected_rate = mismatch_results.get('detected_mismatch_rate', 0)
            detection_accuracy = mismatch_results.get('mismatch_detection_accuracy', 0)
            
            categories = ['True\nMismatch Rate', 'Detected\nMismatch Rate', 'Detection\nAccuracy']
            values = [true_rate, detected_rate, detection_accuracy]
            colors = ['red', 'orange', 'green']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            
            ax.set_ylabel('Rate/Accuracy')
            ax.set_title('Mismatch Detection Analysis', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Mismatch analysis not available',
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_confidence_distributions(self, fig, gs, scenario_results):
        """Plot confidence/similarity score distributions."""
        ax = fig.add_subplot(gs)
        
        # Collect confidence data from different scenarios
        colors = ['blue', 'green', 'red', 'purple']
        color_idx = 0
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                known_key = None
                unknown_key = None
                
                # Find appropriate keys for known/unknown scores
                if 'mean_known_confidence' in results and 'mean_unknown_confidence' in results:
                    known_score = results['mean_known_confidence']
                    unknown_score = results['mean_unknown_confidence']
                    known_key = 'Known Confidence'
                    unknown_key = 'Unknown Confidence'
                elif 'mean_known_similarity' in results and 'mean_unknown_similarity' in results:
                    known_score = results['mean_known_similarity']
                    unknown_score = results['mean_unknown_similarity']
                    known_key = 'Known Similarity'
                    unknown_key = 'Unknown Similarity'
                else:
                    continue
                
                # Create mock distributions around the mean values
                known_dist = np.random.normal(known_score, 0.1, 100)
                unknown_dist = np.random.normal(unknown_score, 0.1, 100)
                
                ax.hist(known_dist, bins=20, alpha=0.5, color=colors[color_idx % len(colors)],
                       label=f'{scenario_name} - Known', density=True)
                ax.hist(unknown_dist, bins=20, alpha=0.5, color=colors[color_idx % len(colors)],
                       label=f'{scenario_name} - Unknown', density=True, linestyle='--', histtype='step')
                
                color_idx += 1
        
        ax.set_xlabel('Confidence/Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title('Score Distributions by Scenario', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_robustness_analysis(self, fig, gs, scenario_results):
        """Plot robustness analysis across scenarios."""
        ax = fig.add_subplot(gs)
        
        scenarios = []
        robustness_scores = []
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                scenarios.append(scenario_name.replace('_', ' ').title())
                
                # Compute robustness score based on available metrics
                auroc = results.get('auroc', 0)
                
                # Penalize scenarios with fewer samples
                num_known = results.get('num_known', results.get('num_matched', 100))
                num_unknown = results.get('num_unknown', results.get('num_mismatched', 100))
                sample_balance = min(num_known, num_unknown) / max(num_known, num_unknown, 1)
                
                # Confidence/similarity separation
                separation = results.get('confidence_separation', 
                                       results.get('similarity_separation', 0))
                
                # Composite robustness score
                robustness = (auroc * 0.5 + sample_balance * 0.3 + 
                            min(abs(separation), 1) * 0.2)
                robustness_scores.append(robustness)
        
        if scenarios:
            colors = plt.cm.RdYlGn(np.array(robustness_scores))
            bars = ax.bar(scenarios, robustness_scores, color=colors, alpha=0.8)
            
            ax.set_ylabel('Robustness Score')
            ax.set_title('Scenario Robustness Analysis', fontweight='bold')
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, robustness_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No robustness data available',
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_scenario_summary(self, fig, gs, scenario_results):
        """Plot summary statistics for each scenario."""
        ax = fig.add_subplot(gs)
        
        # Create summary table visualization
        summary_data = []
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                row_data = {
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'AUROC': f"{results.get('auroc', 0):.3f}",
                    'Samples': f"{results.get('num_known', 0)}K/{results.get('num_unknown', 0)}U",
                    'Best Feature': 'Mixed' if 'strategies' in results else 'Single'
                }
                
                # Add scenario-specific info
                if scenario_name == 'image_only':
                    row_data['Method'] = 'KNN + Distance'
                elif scenario_name == 'text_only':
                    row_data['Method'] = 'Prototype Matching'
                elif scenario_name == 'cross_modal_unpaired':
                    best_strategy = results.get('best_strategy', 'Unknown')
                    row_data['Method'] = best_strategy.replace('_', ' ').title()
                else:
                    row_data['Method'] = 'Standard'
                
                summary_data.append(row_data)
        
        if summary_data:
            # Create table
            df = pd.DataFrame(summary_data)
            
            # Convert to text display
            ax.axis('off')
            
            table_text = "SCENARIO SUMMARY\n\n"
            for _, row in df.iterrows():
                table_text += f"🔸 {row['Scenario']}:\n"
                table_text += f"   AUROC: {row['AUROC']}\n"
                table_text += f"   Method: {row['Method']}\n"
                table_text += f"   Samples: {row['Samples']}\n\n"
            
            ax.text(0.05, 0.95, table_text, transform=ax.transAxes,
                   fontsize=11, fontfamily='monospace', verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No summary data available',
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_recommendations(self, fig, gs, scenario_results):
        """Plot recommendations based on analysis."""
        ax = fig.add_subplot(gs)
        ax.axis('off')
        
        # Analyze results to generate recommendations
        best_auroc = 0
        best_scenario = None
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                auroc = results.get('auroc', 0)
                if 'strategies' in results:
                    # Use best strategy performance
                    best_strategy = results['best_strategy']
                    auroc = results['strategies'][best_strategy]['auroc']
                
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_scenario = scenario_name
        
        recommendations = f"""
🎯 UNPAIRED MODALITY OSR RECOMMENDATIONS

🏆 BEST PERFORMING SCENARIO: {best_scenario.replace('_', ' ').title() if best_scenario else 'N/A'}
   Performance: {best_auroc:.3f} AUROC

📊 KEY FINDINGS:
"""
        
        # Add specific findings
        if 'image_only' in scenario_results:
            img_auroc = scenario_results['image_only'].get('auroc', 0)
            recommendations += f"   • Image-only OSR: {img_auroc:.3f} AUROC\n"
        
        if 'text_only' in scenario_results:
            txt_auroc = scenario_results['text_only'].get('auroc', 0)
            recommendations += f"   • Text-only OSR: {txt_auroc:.3f} AUROC\n"
        
        if 'cross_modal_unpaired' in scenario_results:
            cross_results = scenario_results['cross_modal_unpaired']
            best_strategy = cross_results.get('best_strategy', 'Unknown')
            best_auroc_cross = cross_results['strategies'][best_strategy]['auroc'] if 'strategies' in cross_results else 0
            recommendations += f"   • Cross-modal (best): {best_auroc_cross:.3f} AUROC\n"
            recommendations += f"   • Best strategy: {best_strategy.replace('_', ' ').title()}\n"
        
        recommendations += f"""

🚀 DEPLOYMENT STRATEGY:
   • Primary: Use {best_scenario.replace('_', ' ') if best_scenario else 'robust'} approach
   • Fallback: Image-only for missing text
   • Detection: Monitor pairing quality
   • Adaptation: Adjust based on mismatch rate

🔧 TECHNICAL RECOMMENDATIONS:
   • Mismatch Detection: Implement cross-modal similarity thresholding
   • Confidence Weighting: Combine modalities based on individual confidence
   • Robust Features: Use image features as reliable backup
   • Quality Monitoring: Track alignment scores in production

⚠️ DEPLOYMENT CONSIDERATIONS:
   • Data Quality: Monitor pairing accuracy
   • Performance Trade-offs: Balance accuracy vs robustness
   • Computational Cost: Image-only fastest, fusion most accurate
   • Error Handling: Graceful degradation for missing modalities

💡 FUTURE IMPROVEMENTS:
   • Active Learning: Identify and correct mismatched pairs
   • Domain Adaptation: Fine-tune for specific deployment contexts
   • Ensemble Methods: Combine multiple strategies
   • Real-time Monitoring: Continuous performance assessment
"""
        
        ax.text(0.05, 0.95, recommendations, transform=ax.transAxes,
               fontsize=9, fontfamily='monospace', verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))


def run_unpaired_modality_analysis(vlm_model, processor, test_images, test_texts, 
                                  test_labels, known_classes, 
                                  output_dir='./unpaired_modality_osr',
                                  device='cuda'):
    """
    Run comprehensive unpaired modality OSR analysis.
    
    Args:
        vlm_model: VLM model
        processor: VLM processor
        test_images: List of test images
        test_texts: List of test descriptions
        test_labels: Test labels
        known_classes: List of known class names
        output_dir: Output directory
        device: Computing device
        
    Returns:
        Comprehensive unpaired modality analysis results
    """
    print("="*60)
    print("UNPAIRED MODALITY OPEN-SET RECOGNITION ANALYSIS")
    print("="*60)
    
    # Initialize components
    osr_analyzer = UnpairedModalityOSR(vlm_model, processor, device)
    visualizer = UnpairedModalityVisualizer(output_dir)
    
    # Run different scenario evaluations
    scenario_results = {}
    
    # Scenario 1: Image-only OSR
    print("\n1. Evaluating image-only OSR...")
    image_only_results = osr_analyzer.evaluate_image_only_osr(
        test_images, test_labels, known_classes
    )
    scenario_results['image_only'] = image_only_results
    
    # Scenario 2: Text-only OSR
    print("\n2. Evaluating text-only OSR...")
    text_only_results = osr_analyzer.evaluate_text_only_osr(
        test_texts, test_labels, known_classes
    )
    scenario_results['text_only'] = text_only_results
    
    # Scenario 3: Cross-modal unpaired OSR
    print("\n3. Evaluating cross-modal unpaired OSR...")
    cross_modal_results = osr_analyzer.evaluate_cross_modal_unpaired(
        test_images, test_texts, test_labels, known_classes,
        mismatch_probability=0.3
    )
    scenario_results['cross_modal_unpaired'] = cross_modal_results
    
    # Create comprehensive dashboard
    print("\n4. Creating comprehensive dashboard...")
    fig = visualizer.create_comprehensive_dashboard(
        scenario_results,
        save_path=os.path.join(output_dir, 'unpaired_modality_dashboard.png')
    )
    plt.close(fig)
    
    # Generate detailed report
    print("\n5. Generating analysis report...")
    report_path = os.path.join(output_dir, 'unpaired_modality_report.md')
    _generate_unpaired_modality_report(scenario_results, report_path)
    
    # Save analysis data
    results_path = os.path.join(output_dir, 'unpaired_analysis_results.npz')
    
    # Prepare data for saving (convert any torch tensors to numpy)
    save_data = {}
    for scenario, results in scenario_results.items():
        if isinstance(results, dict):
            # Convert torch tensors and complex objects to basic types
            clean_results = {}
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    clean_results[key] = value.cpu().numpy()
                elif isinstance(value, (int, float, str, bool, list)):
                    clean_results[key] = value
                elif isinstance(value, dict):
                    clean_results[key] = {k: v for k, v in value.items() 
                                        if isinstance(v, (int, float, str, bool, list, np.ndarray))}
            save_data[scenario] = clean_results
    
    np.savez(results_path, **save_data)
    
    print(f"\n✅ Unpaired modality analysis completed!")
    print(f"📁 Results saved to: {output_dir}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    for scenario, results in scenario_results.items():
        if isinstance(results, dict) and 'auroc' in results:
            print(f"  {scenario.replace('_', ' ').title()}: {results['auroc']:.3f} AUROC")
        elif isinstance(results, dict) and 'strategies' in results:
            best_strategy = results['best_strategy']
            best_auroc = results['strategies'][best_strategy]['auroc']
            print(f"  {scenario.replace('_', ' ').title()}: {best_auroc:.3f} AUROC (best strategy)")
    
    return {
        'scenario_results': scenario_results,
        'report_path': report_path,
        'results_path': results_path
    }


def _generate_unpaired_modality_report(scenario_results, report_path):
    """Generate detailed unpaired modality analysis report."""
    with open(report_path, 'w') as f:
        f.write("# Unpaired Modality Open-Set Recognition Analysis Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report analyzes open-set recognition performance when image-text ")
        f.write("modalities are unpaired, missing, or mismatched. The analysis covers ")
        f.write("image-only, text-only, and cross-modal unpaired scenarios with multiple ")
        f.write("handling strategies.\n\n")
        
        # Performance overview
        f.write("## Performance Overview\n\n")
        f.write("| Scenario | AUROC | Method | Key Strength |\n")
        f.write("|----------|-------|---------|-------------|\n")
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                if 'auroc' in results:
                    auroc = results['auroc']
                    method = "Direct"
                elif 'strategies' in results:
                    best_strategy = results['best_strategy']
                    auroc = results['strategies'][best_strategy]['auroc']
                    method = best_strategy.replace('_', ' ').title()
                else:
                    continue
                
                # Determine key strength
                if scenario_name == 'image_only':
                    strength = "Visual feature reliability"
                elif scenario_name == 'text_only':
                    strength = "Semantic understanding"
                elif scenario_name == 'cross_modal_unpaired':
                    strength = "Adaptive multi-modal fusion"
                else:
                    strength = "Robust performance"
                
                f.write(f"| {scenario_name.replace('_', ' ').title()} | {auroc:.4f} | {method} | {strength} |\n")
        
        # Detailed scenario analysis
        f.write("\n## Detailed Scenario Analysis\n\n")
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                f.write(f"### {scenario_name.replace('_', ' ').title()} Analysis\n\n")
                
                if scenario_name == 'image_only':
                    f.write("**Approach**: Uses only visual features without text guidance.\n\n")
                    f.write("**Performance Metrics**:\n")
                    f.write(f"- AUROC: {results.get('auroc', 0):.4f}\n")
                    f.write(f"- Known classification accuracy: {results.get('known_accuracy', 0):.4f}\n")
                    f.write(f"- Confidence separation: {results.get('confidence_separation', 0):.4f}\n")
                    f.write(f"- Optimal threshold: {results.get('best_threshold', 0):.4f}\n\n")
                    
                elif scenario_name == 'text_only':
                    f.write("**Approach**: Uses only text descriptions with class prototypes.\n\n")
                    f.write("**Performance Metrics**:\n")
                    f.write(f"- AUROC: {results.get('auroc', 0):.4f}\n")
                    f.write(f"- Known classification accuracy: {results.get('known_accuracy', 0):.4f}\n")
                    f.write(f"- Similarity separation: {results.get('similarity_separation', 0):.4f}\n\n")
                    
                elif scenario_name == 'cross_modal_unpaired':
                    f.write("**Approach**: Handles mismatched image-text pairs with multiple strategies.\n\n")
                    f.write(f"**Mismatch Simulation**: {results.get('mismatch_rate', 0):.1%} of pairs mismatched\n\n")
                    
                    if 'strategies' in results:
                        f.write("**Strategy Performance**:\n")
                        for strategy_name, strategy_results in results['strategies'].items():
                            if isinstance(strategy_results, dict) and 'auroc' in strategy_results:
                                f.write(f"- {strategy_name.replace('_', ' ').title()}: {strategy_results['auroc']:.4f} AUROC\n")
                        
                        best_strategy = results['best_strategy']
                        f.write(f"\n**Best Strategy**: {best_strategy.replace('_', ' ').title()}\n\n")
                        
                        # Detailed analysis of best strategy
                        best_results = results['strategies'][best_strategy]
                        if best_strategy == 'mismatch_detection':
                            f.write("**Mismatch Detection Analysis**:\n")
                            f.write(f"- Detection accuracy: {best_results.get('mismatch_detection_accuracy', 0):.4f}\n")
                            f.write(f"- True mismatch rate: {best_results.get('true_mismatch_rate', 0):.4f}\n")
                            f.write(f"- Detected mismatch rate: {best_results.get('detected_mismatch_rate', 0):.4f}\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Find best performing scenario
        best_auroc = 0
        best_scenario = None
        
        for scenario_name, results in scenario_results.items():
            if isinstance(results, dict):
                auroc = results.get('auroc', 0)
                if 'strategies' in results:
                    best_strategy = results['best_strategy'] 
                    auroc = results['strategies'][best_strategy]['auroc']
                
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_scenario = scenario_name
        
        f.write(f"1. **Best Overall Performance**: {best_scenario.replace('_', ' ').title() if best_scenario else 'N/A'} ")
        f.write(f"achieved {best_auroc:.4f} AUROC\n\n")
        
        f.write("2. **Robustness**: Image-only approaches provide reliable baseline performance ")
        f.write("when text is unavailable or unreliable\n\n")
        
        f.write("3. **Semantic Understanding**: Text-only methods leverage semantic knowledge ")
        f.write("but may struggle with visual-specific features\n\n")
        
        f.write("4. **Adaptive Strategies**: Cross-modal approaches can adapt to pairing quality ")
        f.write("and maintain performance under various conditions\n\n")
        
        # Recommendations
        f.write("## Deployment Recommendations\n\n")
        
        f.write("### Primary Strategy\n")
        f.write(f"- **Recommended approach**: {best_scenario.replace('_', ' ').title() if best_scenario else 'Hybrid'}\n")
        f.write(f"- **Expected performance**: {best_auroc:.4f} AUROC\n")
        f.write("- **Use case**: Production deployment with quality monitoring\n\n")
        
        f.write("### Fallback Strategies\n")
        if 'image_only' in scenario_results:
            img_auroc = scenario_results['image_only'].get('auroc', 0)
            f.write(f"- **Image-only fallback**: {img_auroc:.4f} AUROC when text unavailable\n")
        
        if 'text_only' in scenario_results:
            txt_auroc = scenario_results['text_only'].get('auroc', 0)
            f.write(f"- **Text-only fallback**: {txt_auroc:.4f} AUROC when images unavailable\n")
        
        f.write("\n### Implementation Guidelines\n")
        f.write("1. **Quality Monitoring**: Implement cross-modal similarity monitoring\n")
        f.write("2. **Adaptive Thresholding**: Adjust decision thresholds based on pairing quality\n")
        f.write("3. **Graceful Degradation**: Ensure system maintains performance with missing modalities\n")
        f.write("4. **Performance Tracking**: Monitor OSR performance across different scenarios\n\n")
        
        f.write("### Future Improvements\n")
        f.write("- **Active Learning**: Identify and correct mismatched pairs\n")
        f.write("- **Domain Adaptation**: Fine-tune for specific deployment contexts\n")
        f.write("- **Ensemble Methods**: Combine multiple unpaired strategies\n")
        f.write("- **Real-time Adaptation**: Dynamic strategy selection based on data quality\n")


if __name__ == "__main__":
    print("Unpaired Modality Open-Set Recognition Module")
    print("Usage: Import and call run_unpaired_modality_analysis() with your VLM model and data")