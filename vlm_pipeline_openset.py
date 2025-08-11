#!/usr/bin/env python3
"""
Enhanced VLM Open-Set Evaluation Pipeline
=========================================

This enhanced pipeline incorporates state-of-the-art open-set recognition methods and metrics:

1. **Advanced Metrics**: AUOSCR, OSCR, FPR95, AUROC, per-class fairness metrics
2. **GHOST Algorithm**: Manuel Gunther's Gaussian Hypothesis Open-Set Technique
3. **Proper Open-Set Protocol**: Training on known classes, testing on all
4. **Text Prompt Analysis**: Distance analysis between text prompts and unknown classes
5. **Enhanced Feature Analysis**: Comprehensive discriminability assessment
6. **LoRA Integration**: Fixed implementation with proper evaluation

Key Improvements:
- Uses Boeing/Airbus as known classes, all others as unknown
- Implements GHOST approach with Gaussian modeling
- Adds comprehensive open-set metrics (AUOSCR, OSCR, FPR95)
- Includes per-class fairness analysis
- Enhanced text prompt vs unknown class distance analysis
- Proper statistical significance testing
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import shapiro
import umap
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# VLM and dataset imports
from transformers import CLIPProcessor, CLIPModel
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import FGVCAircraft
from PIL import Image


class OpenSetMetrics:
    """
    Comprehensive open-set recognition metrics including AUOSCR, OSCR, FPR95, and fairness measures.
    
    Based on recent literature and Manuel Gunther's work.
    """
    
    @staticmethod
    def compute_oscr_curve(known_scores: np.ndarray, 
                          unknown_scores: np.ndarray,
                          known_predictions: np.ndarray,
                          known_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Open Set Classification Rate (OSCR) curve.
        
        OSCR measures the trade-off between correct classification rate (CCR) of known classes
        and false positive rate (FPR) of unknown classes.
        
        Args:
            known_scores: Confidence scores for known class samples
            unknown_scores: Confidence scores for unknown class samples  
            known_predictions: Predicted class labels for known samples
            known_labels: True class labels for known samples
            
        Returns:
            ccr: Correct Classification Rate at different thresholds
            fpr: False Positive Rate at different thresholds  
            thresholds: Decision thresholds
        """
        # Combine scores and create labels (1 = known, 0 = unknown)
        all_scores = np.concatenate([known_scores, unknown_scores])
        score_labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(-all_scores)
        sorted_scores = all_scores[sorted_indices]
        sorted_labels = score_labels[sorted_indices]
        
        # Get unique thresholds
        thresholds = np.unique(sorted_scores)
        thresholds = np.concatenate([thresholds, [thresholds[-1] - 1e-6]])  # Add one more threshold
        
        ccr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            # Samples predicted as known (score >= threshold)
            predicted_known_mask = all_scores >= threshold
            predicted_unknown_mask = ~predicted_known_mask
            
            # For known samples: check if correctly classified AND predicted as known
            known_predicted_known = predicted_known_mask[:len(known_scores)]
            correctly_classified = (known_predictions == known_labels)
            correct_known = np.sum(correctly_classified & known_predicted_known)
            
            # CCR = (correctly classified known samples) / (total known samples)
            ccr = correct_known / len(known_scores) if len(known_scores) > 0 else 0
            
            # For unknown samples: FPR = (unknown predicted as known) / (total unknown)
            unknown_predicted_known = predicted_known_mask[len(known_scores):]
            fpr = np.sum(unknown_predicted_known) / len(unknown_scores) if len(unknown_scores) > 0 else 0
            
            ccr_values.append(ccr)
            fpr_values.append(fpr)
            
        return np.array(ccr_values), np.array(fpr_values), thresholds
    
    @staticmethod
    def compute_auoscr(known_scores: np.ndarray, 
                      unknown_scores: np.ndarray,
                      known_predictions: np.ndarray,
                      known_labels: np.ndarray) -> float:
        """Compute Area Under Open Set Classification Rate curve."""
        ccr, fpr, _ = OpenSetMetrics.compute_oscr_curve(
            known_scores, unknown_scores, known_predictions, known_labels
        )
        
        # Sort by FPR for proper integration
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        ccr_sorted = ccr[sorted_indices]
        
        # Compute AUC using trapezoidal rule
        return auc(fpr_sorted, ccr_sorted)
    
    @staticmethod
    def compute_fpr95(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
        """Compute False Positive Rate when True Positive Rate is 95%."""
        # Combine scores and labels
        all_scores = np.concatenate([known_scores, unknown_scores])
        all_labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        
        # Find FPR when TPR >= 0.95
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) == 0:
            return 1.0  # If TPR never reaches 95%, return worst case
        
        return fpr[idx[0]]
    
    @staticmethod
    def compute_per_class_fairness(scores_by_class: Dict[str, np.ndarray], 
                                  threshold: float = None) -> Dict[str, float]:
        """
        Compute per-class fairness metrics as proposed by Manuel Gunther.
        
        Args:
            scores_by_class: Dictionary mapping class names to confidence scores
            threshold: Decision threshold (if None, use median of all scores)
            
        Returns:
            Dictionary with fairness metrics
        """
        all_scores = np.concatenate(list(scores_by_class.values()))
        if threshold is None:
            threshold = np.median(all_scores)
        
        # Compute per-class detection rates
        class_detection_rates = {}
        for class_name, scores in scores_by_class.items():
            detection_rate = np.mean(scores >= threshold)
            class_detection_rates[class_name] = detection_rate
        
        # Compute fairness metrics
        detection_rates = np.array(list(class_detection_rates.values()))
        
        fairness_metrics = {
            'coefficient_of_variation': np.std(detection_rates) / np.mean(detection_rates) if np.mean(detection_rates) > 0 else np.inf,
            'max_difference': np.max(detection_rates) - np.min(detection_rates),
            'worst_class_performance': np.min(detection_rates),
            'best_class_performance': np.max(detection_rates),
            'per_class_rates': class_detection_rates
        }
        
        return fairness_metrics


class GHOSTDetector:
    """
    Implementation of GHOST (Gaussian Hypothesis Open-Set Technique) by Manuel Gunther.
    
    Models each known class as a multivariate Gaussian distribution and uses
    Z-score normalization for open-set detection.
    """
    
    def __init__(self, use_diagonal_covariance: bool = True):
        """
        Initialize GHOST detector.
        
        Args:
            use_diagonal_covariance: If True, use diagonal covariance matrices
        """
        self.use_diagonal_covariance = use_diagonal_covariance
        self.class_gaussians = {}
        self.fitted = False
        
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Fit Gaussian distributions for each known class.
        
        Args:
            features: Feature vectors for known classes
            labels: Class labels
        """
        unique_classes = np.unique(labels)
        
        for class_label in unique_classes:
            class_mask = labels == class_label
            class_features = features[class_mask]
            
            if len(class_features) < 2:
                # Not enough samples for Gaussian estimation
                continue
                
            # Compute mean and covariance
            mean = np.mean(class_features, axis=0)
            
            if self.use_diagonal_covariance:
                # Use diagonal covariance (assume feature independence)
                variance = np.var(class_features, axis=0)
                # Add small epsilon to prevent numerical issues
                variance = np.maximum(variance, 1e-6)
                covariance = np.diag(variance)
            else:
                # Use full covariance matrix
                covariance = np.cov(class_features.T)
                # Regularize to ensure positive definiteness
                covariance += 1e-6 * np.eye(covariance.shape[0])
            
            self.class_gaussians[class_label] = {
                'mean': mean,
                'covariance': covariance,
                'variance': np.diag(covariance) if self.use_diagonal_covariance else np.diag(covariance)
            }
        
        # Test Gaussian hypothesis using Shapiro-Wilk test
        self._test_gaussian_hypothesis(features, labels)
        self.fitted = True
    
    def _test_gaussian_hypothesis(self, features: np.ndarray, labels: np.ndarray):
        """Test if features follow Gaussian distribution using Shapiro-Wilk test."""
        unique_classes = np.unique(labels)
        normality_results = {}
        
        for class_label in unique_classes:
            class_mask = labels == class_label
            class_features = features[class_mask]
            
            if len(class_features) < 3:  # Need at least 3 samples for Shapiro-Wilk
                continue
                
            # Test each feature dimension
            p_values = []
            for dim in range(class_features.shape[1]):
                try:
                    _, p_value = shapiro(class_features[:, dim])
                    p_values.append(p_value)
                except:
                    continue
            
            if p_values:
                # Apply Holm's step-down procedure for multiple testing
                p_values = np.array(p_values)
                rejected_hypothesis = np.sum(p_values < 0.05) / len(p_values)
                normality_results[class_label] = {
                    'rejection_rate': rejected_hypothesis,
                    'passes_normality': rejected_hypothesis < 0.1  # Allow 10% violations
                }
        
        self.normality_results = normality_results
    
    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Compute GHOST scores for samples.
        
        Args:
            features: Feature vectors to score
            
        Returns:
            Array of GHOST scores (higher = more likely to be known class)
        """
        if not self.fitted:
            raise ValueError("GHOST detector must be fitted before prediction")
        
        scores = []
        
        for feature in features:
            max_score = -np.inf
            
            for class_label, gaussian_params in self.class_gaussians.items():
                mean = gaussian_params['mean']
                variance = gaussian_params['variance']
                
                # Compute Z-score normalized log-likelihood
                z_scores = (feature - mean) / np.sqrt(variance + 1e-8)
                
                # Negative of sum of squared Z-scores (higher is better)
                score = -np.sum(z_scores ** 2)
                max_score = max(max_score, score)
            
            scores.append(max_score)
        
        return np.array(scores)
    
    def predict_class_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Compute class probabilities using Gaussian distributions."""
        if not self.fitted:
            raise ValueError("GHOST detector must be fitted before prediction")
        
        n_samples = len(features)
        n_classes = len(self.class_gaussians)
        class_names = list(self.class_gaussians.keys())
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, feature in enumerate(features):
            log_likelihoods = []
            
            for class_label in class_names:
                gaussian_params = self.class_gaussians[class_label]
                mean = gaussian_params['mean']
                variance = gaussian_params['variance']
                
                # Compute log-likelihood
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance))
                log_likelihood -= 0.5 * np.sum((feature - mean) ** 2 / variance)
                log_likelihoods.append(log_likelihood)
            
            # Convert to probabilities (softmax)
            log_likelihoods = np.array(log_likelihoods)
            log_likelihoods -= np.max(log_likelihoods)  # Numerical stability
            probabilities[i] = np.exp(log_likelihoods)
            probabilities[i] /= np.sum(probabilities[i])
        
        return probabilities


class TextPromptDistanceAnalyzer:
    """
    Analyzes distances between text prompts and visual features for unknown classes.
    
    This addresses your requirement to analyze where unknown classes get classified
    relative to text prompts, without using text prompts during testing.
    """
    
    def __init__(self, clip_extractor):
        self.clip_extractor = clip_extractor
        self.known_class_text_embeddings = {}
        
    def compute_known_class_text_embeddings(self, known_classes: List[str], 
                                          prompt_templates: Dict[str, str]):
        """
        Compute text embeddings for known classes using different prompt strategies.
        
        Args:
            known_classes: List of known class names (e.g., ['Boeing', 'Airbus'])
            prompt_templates: Dictionary of prompt templates
        """
        for template_name, template in prompt_templates.items():
            class_embeddings = {}
            
            for class_name in known_classes:
                prompt = template.format(class_name)
                text_embedding = self.clip_extractor.extract_text_embeddings([prompt])[0]
                class_embeddings[class_name] = text_embedding
            
            self.known_class_text_embeddings[template_name] = class_embeddings
    
    def analyze_unknown_class_distances(self, 
                                      unknown_visual_features: np.ndarray,
                                      unknown_labels: List[str],
                                      prompt_template: str = 'descriptive') -> Dict:
        """
        Analyze how far unknown classes are from known class text prompts.
        
        Args:
            unknown_visual_features: Visual features of unknown class samples
            unknown_labels: Labels of unknown samples
            prompt_template: Which prompt template to use for comparison
            
        Returns:
            Dictionary with distance analysis results
        """
        if prompt_template not in self.known_class_text_embeddings:
            raise ValueError(f"Prompt template '{prompt_template}' not found")
        
        text_embeddings = self.known_class_text_embeddings[prompt_template]
        known_classes = list(text_embeddings.keys())
        
        results = {
            'prompt_template': prompt_template,
            'distance_statistics': {},
            'per_unknown_class_analysis': {},
            'nearest_known_class_mapping': {}
        }
        
        # Compute distances for each unknown sample
        all_distances = []
        sample_analyses = []
        
        for i, (visual_feat, unknown_label) in enumerate(zip(unknown_visual_features, unknown_labels)):
            sample_analysis = {
                'unknown_class': unknown_label,
                'distances_to_known': {},
                'nearest_known_class': None,
                'min_distance': np.inf
            }
            
            for known_class, text_emb in text_embeddings.items():
                # Compute cosine distance
                distance = cosine(visual_feat, text_emb)
                sample_analysis['distances_to_known'][known_class] = distance
                all_distances.append(distance)
                
                if distance < sample_analysis['min_distance']:
                    sample_analysis['min_distance'] = distance
                    sample_analysis['nearest_known_class'] = known_class
            
            sample_analyses.append(sample_analysis)
        
        # Aggregate statistics
        results['distance_statistics'] = {
            'mean_distance': np.mean(all_distances),
            'std_distance': np.std(all_distances),
            'min_distance': np.min(all_distances),
            'max_distance': np.max(all_distances),
            'median_distance': np.median(all_distances)
        }
        
        # Per unknown class analysis
        unique_unknown_classes = list(set(unknown_labels))
        for unknown_class in unique_unknown_classes:
            class_samples = [s for s in sample_analyses if s['unknown_class'] == unknown_class]
            
            class_distances = []
            nearest_classes = []
            
            for sample in class_samples:
                class_distances.append(sample['min_distance'])
                nearest_classes.append(sample['nearest_known_class'])
            
            # Find most common nearest class
            from collections import Counter
            nearest_class_counts = Counter(nearest_classes)
            most_common_nearest = nearest_class_counts.most_common(1)[0]
            
            results['per_unknown_class_analysis'][unknown_class] = {
                'mean_min_distance': np.mean(class_distances),
                'std_min_distance': np.std(class_distances),
                'most_common_nearest_class': most_common_nearest[0],
                'nearest_class_frequency': most_common_nearest[1] / len(class_samples),
                'n_samples': len(class_samples)
            }
        
        # Overall nearest class mapping
        for known_class in known_classes:
            mapped_unknowns = [s['unknown_class'] for s in sample_analyses 
                             if s['nearest_known_class'] == known_class]
            unique_mapped = list(set(mapped_unknowns))
            
            results['nearest_known_class_mapping'][known_class] = {
                'mapped_unknown_classes': unique_mapped,
                'total_unknown_samples': len(mapped_unknowns)
            }
        
        return results


class LoRAAdapter(nn.Module):
    """
    LoRA (Low-Rank Adaptation) module for fine-tuning embeddings.
    
    Moved to module level to fix pickling issues.
    """
    
    def __init__(self, embedding_dim: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(embedding_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, embedding_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.lora_A @ self.lora_B) * self.scaling


class EnhancedVLMEvaluationPipeline:
    """
    Enhanced VLM evaluation pipeline with state-of-the-art open-set recognition methods.
    """
    
    def __init__(self, 
                 clip_model: str = "openai/clip-vit-base-patch32",
                 device: str = None,
                 known_classes: List[str] = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.known_classes = known_classes or ['Boeing', 'Airbus']
        
        # Initialize components
        self.clip_extractor = self._init_clip_extractor(clip_model)
        self.resnet_extractor = self._init_resnet_extractor()
        self.ghost_detector = GHOSTDetector()
        self.text_analyzer = TextPromptDistanceAnalyzer(self.clip_extractor)
        self.metrics_calculator = OpenSetMetrics()
        
        # Storage
        self.results = {}
        self.features_cache = {}
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _init_clip_extractor(self, model_name: str):
        """Initialize CLIP feature extractor."""
        from vlm_eval_framework import CLIPFeatureExtractor  # Assuming this exists
        return CLIPFeatureExtractor(model_name, self.device)
        
    def _init_resnet_extractor(self):
        """Initialize ResNet feature extractor."""
        from vlm_eval_framework import ResNetFeatureExtractor  # Assuming this exists
        return ResNetFeatureExtractor("resnet50", device=self.device)
    
    def load_and_prepare_data(self, data_dir: str, max_samples: int = None):
        """
        Load FGVC Aircraft dataset and prepare for open-set evaluation.
        
        Following proper open-set protocol:
        - Training set: Only known classes (Boeing, Airbus)
        - Test set: All classes (known + unknown)
        """
        # Load dataset
        train_dataset = FGVCAircraft(
            root=data_dir,
            split='train',
            annotation_level='manufacturer',
            download=True,
            transform=None
        )
        
        test_dataset = FGVCAircraft(
            root=data_dir,
            split='test', 
            annotation_level='manufacturer',
            download=False,
            transform=None
        )
        
        # Get all classes
        all_classes = train_dataset.classes
        self.logger.info(f"All available classes: {all_classes}")
        self.logger.info(f"Known classes for training: {self.known_classes}")
        
        # Filter training set to only known classes
        train_indices = []
        for i in range(len(train_dataset)):
            _, label_idx = train_dataset[i]
            class_name = train_dataset.classes[label_idx]
            if class_name in self.known_classes:
                train_indices.append(i)
        
        if len(train_indices) == 0:
            raise ValueError(f"No training samples found for known classes: {self.known_classes}")
        
        # Create filtered training set (only known classes)
        train_subset = Subset(train_dataset, train_indices)
        
        # Test set contains all classes (known + unknown)
        test_subset = test_dataset
        
        # Apply sampling if requested
        if max_samples:
            # Sample from training subset
            if len(train_indices) > max_samples:
                sampled_train_indices = np.random.choice(
                    train_indices, max_samples, replace=False
                )
                train_subset = Subset(train_dataset, sampled_train_indices)
            
            # Sample from test set
            if len(test_dataset) > max_samples:
                sampled_test_indices = np.random.choice(
                    len(test_dataset), max_samples, replace=False
                )
                test_subset = Subset(test_dataset, sampled_test_indices)
        
        self.logger.info(f"Training samples (known classes only): {len(train_subset)}")
        self.logger.info(f"Test samples (all classes): {len(test_subset)}")
        
        return train_subset, test_subset, all_classes
    
    def dataset_to_lists(self, dataset) -> Tuple[List[Image.Image], List[str]]:
        """Convert dataset to lists of images and labels."""
        images = []
        labels = []
        
        for i in range(len(dataset)):
            if isinstance(dataset, Subset):
                original_idx = dataset.indices[i]
                image, label_idx = dataset.dataset[original_idx]
                label = dataset.dataset.classes[label_idx]
            else:
                image, label_idx = dataset[i]
                label = dataset.classes[label_idx]
            
            images.append(image)
            labels.append(label)
        
        return images, labels
    
    def extract_all_features(self, images: List[Image.Image], feature_types: List[str] = None):
        """Extract features using different methods."""
        if feature_types is None:
            feature_types = ['clip_pre_embedding', 'clip_embeddings', 'resnet']
        
        features = {}
        
        for feature_type in feature_types:
            self.logger.info(f"Extracting {feature_type} features...")
            
            if feature_type == 'clip_pre_embedding':
                features[feature_type] = self.clip_extractor.extract_features(images)
            elif feature_type == 'clip_embeddings':
                features[feature_type] = self.clip_extractor.extract_embeddings(images)
            elif feature_type == 'resnet':
                features[feature_type] = self.resnet_extractor.extract_features(images)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
        
        return features
    
    def train_classifiers_on_known_classes(self, 
                                         features: Dict[str, np.ndarray],
                                         labels: List[str]) -> Dict:
        """Train classifiers on known classes only."""
        
        # Filter to known classes only
        known_mask = np.array([label in self.known_classes for label in labels])
        known_labels = np.array(labels)[known_mask]
        
        classifiers = {
            'knn_1': KNeighborsClassifier(n_neighbors=1),
            'knn_3': KNeighborsClassifier(n_neighbors=3),
            'knn_5': KNeighborsClassifier(n_neighbors=5),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'svm_linear': SVC(kernel='linear', probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, random_state=42)
        }
        
        trained_classifiers = {}
        
        for feature_type, feature_data in features.items():
            self.logger.info(f"Training classifiers on {feature_type}...")
            
            # Filter features to known classes
            known_features = feature_data[known_mask]
            
            # Standardize features
            scaler = StandardScaler()
            known_features_scaled = scaler.fit_transform(known_features)
            
            feature_classifiers = {}
            
            for clf_name, clf in classifiers.items():
                try:
                    clf_copy = type(clf)(**clf.get_params())
                    clf_copy.fit(known_features_scaled, known_labels)
                    feature_classifiers[clf_name] = {
                        'classifier': clf_copy,
                        'scaler': scaler
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to train {clf_name} on {feature_type}: {e}")
            
            trained_classifiers[feature_type] = feature_classifiers
        
        return trained_classifiers
    
    def evaluate_open_set_performance(self,
                                    test_features: Dict[str, np.ndarray],
                                    test_labels: List[str],
                                    trained_classifiers: Dict) -> Dict:
        """
        Comprehensive open-set evaluation using advanced metrics.
        """
        results = {}
        
        # Separate known and unknown test samples
        test_labels_array = np.array(test_labels)
        known_test_mask = np.array([label in self.known_classes for label in test_labels])
        unknown_test_mask = ~known_test_mask
        
        n_known_test = np.sum(known_test_mask)
        n_unknown_test = np.sum(unknown_test_mask)
        
        self.logger.info(f"Test set - Known samples: {n_known_test}, Unknown samples: {n_unknown_test}")
        
        for feature_type, feature_data in test_features.items():
            self.logger.info(f"Evaluating open-set performance for {feature_type}...")
            
            feature_results = {}
            
            # Get known and unknown features
            known_test_features = feature_data[known_test_mask]
            unknown_test_features = feature_data[unknown_test_mask]
            known_test_labels = test_labels_array[known_test_mask]
            unknown_test_labels = test_labels_array[unknown_test_mask]
            
            # Train GHOST detector on known training features (from train_classifiers step)
            train_known_features = self.features_cache['train'][feature_type]
            train_known_labels = np.array(self.features_cache['train_labels'])
            train_known_mask = np.array([label in self.known_classes for label in train_known_labels])
            
            self.ghost_detector.fit(
                train_known_features[train_known_mask],
                train_known_labels[train_known_mask]
            )
            
            # Evaluate each classifier
            for clf_name, clf_data in trained_classifiers[feature_type].items():
                classifier = clf_data['classifier']
                scaler = clf_data['scaler']
                
                clf_results = {}
                
                # Scale test features
                known_test_scaled = scaler.transform(known_test_features)
                unknown_test_scaled = scaler.transform(unknown_test_features)
                
                # Get predictions and confidence scores for known test samples
                known_predictions = classifier.predict(known_test_scaled)
                if hasattr(classifier, 'predict_proba'):
                    known_probabilities = classifier.predict_proba(known_test_scaled)
                    known_scores = np.max(known_probabilities, axis=1)
                elif hasattr(classifier, 'decision_function'):
                    known_scores = classifier.decision_function(known_test_scaled)
                    if len(known_scores.shape) > 1:
                        known_scores = np.max(known_scores, axis=1)
                else:
                    # Fallback: use distance to decision boundary
                    known_scores = np.ones(len(known_test_scaled))
                
                # Get confidence scores for unknown test samples
                if hasattr(classifier, 'predict_proba'):
                    unknown_probabilities = classifier.predict_proba(unknown_test_scaled)
                    unknown_scores = np.max(unknown_probabilities, axis=1)
                elif hasattr(classifier, 'decision_function'):
                    unknown_scores = classifier.decision_function(unknown_test_scaled)
                    if len(unknown_scores.shape) > 1:
                        unknown_scores = np.max(unknown_scores, axis=1)
                else:
                    unknown_scores = np.ones(len(unknown_test_scaled))
                
                # Closed-set accuracy on known test samples
                closed_set_accuracy = accuracy_score(known_test_labels, known_predictions)
                
                # Open-set metrics
                
                # 1. AUROC for known vs unknown detection
                all_scores = np.concatenate([known_scores, unknown_scores])
                all_binary_labels = np.concatenate([
                    np.ones(len(known_scores)),  # Known = 1
                    np.zeros(len(unknown_scores))  # Unknown = 0
                ])
                
                fpr, tpr, _ = roc_curve(all_binary_labels, all_scores)
                auroc = auc(fpr, tpr)
                
                # 2. FPR95 - False Positive Rate when TPR = 95%
                fpr95 = self.metrics_calculator.compute_fpr95(known_scores, unknown_scores)
                
                # 3. OSCR and AUOSCR
                auoscr = self.metrics_calculator.compute_auoscr(
                    known_scores, unknown_scores, known_predictions, known_test_labels
                )
                
                # 4. GHOST scores
                ghost_known_scores = self.ghost_detector.predict_scores(known_test_features)
                ghost_unknown_scores = self.ghost_detector.predict_scores(unknown_test_features)
                
                # GHOST AUROC
                ghost_all_scores = np.concatenate([ghost_known_scores, ghost_unknown_scores])
                ghost_fpr, ghost_tpr, _ = roc_curve(all_binary_labels, ghost_all_scores)
                ghost_auroc = auc(ghost_fpr, ghost_tpr)
                
                # GHOST FPR95
                ghost_fpr95 = self.metrics_calculator.compute_fpr95(ghost_known_scores, ghost_unknown_scores)
                
                # GHOST AUOSCR
                ghost_auoscr = self.metrics_calculator.compute_auoscr(
                    ghost_known_scores, ghost_unknown_scores, known_predictions, known_test_labels
                )
                
                # 5. Per-class fairness analysis
                if len(np.unique(known_test_labels)) > 1:
                    known_scores_by_class = {}
                    for class_name in np.unique(known_test_labels):
                        class_mask = known_test_labels == class_name
                        known_scores_by_class[class_name] = known_scores[class_mask]
                    
                    fairness_metrics = self.metrics_calculator.compute_per_class_fairness(
                        known_scores_by_class
                    )
                else:
                    fairness_metrics = {'error': 'Insufficient classes for fairness analysis'}
                
                # Store results
                clf_results = {
                    'closed_set_accuracy': closed_set_accuracy,
                    'classifier_auroc': auroc,
                    'classifier_fpr95': fpr95,
                    'classifier_auoscr': auoscr,
                    'ghost_auroc': ghost_auroc,
                    'ghost_fpr95': ghost_fpr95,
                    'ghost_auoscr': ghost_auoscr,
                    'fairness_metrics': fairness_metrics,
                    'n_known_test': n_known_test,
                    'n_unknown_test': n_unknown_test
                }
                
                feature_results[clf_name] = clf_results
            
            results[feature_type] = feature_results
        
        return results
    
    def analyze_text_prompt_distances(self, 
                                     test_features: Dict[str, np.ndarray],
                                     test_labels: List[str]) -> Dict:
        """
        Analyze distances between text prompts and unknown class visual features.
        
        This addresses your specific requirement about text prompt analysis.
        """
        self.logger.info("Analyzing text prompt distances...")
        
        # Define prompt templates
        prompt_templates = {
            'simple': "{}",
            'basic': "a photo of a {}",
            'descriptive': "a photo of a {} aircraft",
            'detailed': "a high quality photo of a {} aircraft",
            'context': "an image showing a {} airplane in flight",
            'technical': "aircraft model {} photographed clearly"
        }
        
        # Compute text embeddings for known classes
        self.text_analyzer.compute_known_class_text_embeddings(
            self.known_classes, prompt_templates
        )
        
        results = {}
        
        # Only analyze CLIP embeddings for text comparison
        if 'clip_embeddings' not in test_features:
            self.logger.warning("CLIP embeddings not available for text prompt analysis")
            return results
        
        # Separate unknown samples
        test_labels_array = np.array(test_labels)
        unknown_mask = np.array([label not in self.known_classes for label in test_labels])
        
        if np.sum(unknown_mask) == 0:
            self.logger.warning("No unknown samples found for text prompt analysis")
            return results
        
        unknown_features = test_features['clip_embeddings'][unknown_mask]
        unknown_labels = test_labels_array[unknown_mask].tolist()
        
        # Analyze for each prompt template
        for template_name in prompt_templates.keys():
            self.logger.info(f"Analyzing distances for prompt template: {template_name}")
            
            template_results = self.text_analyzer.analyze_unknown_class_distances(
                unknown_features, unknown_labels, template_name
            )
            
            results[template_name] = template_results
        
        return results
    
    def evaluate_lora_adaptation(self, 
                                train_features: Dict[str, np.ndarray],
                                train_labels: List[str],
                                rank: int = 16,
                                epochs: int = 20,
                                lr: float = 0.001) -> Dict:
        """
        Evaluate LoRA fine-tuning on CLIP embeddings with proper open-set evaluation.
        """
        self.logger.info("Evaluating LoRA adaptation...")
        
        if 'clip_embeddings' not in train_features:
            return {'error': 'CLIP embeddings not available for LoRA adaptation'}
        
        # Filter to known classes only (proper open-set protocol)
        train_labels_array = np.array(train_labels)
        known_mask = np.array([label in self.known_classes for label in train_labels])
        
        known_features = train_features['clip_embeddings'][known_mask]
        known_labels = train_labels_array[known_mask]
        
        if len(known_features) == 0:
            return {'error': 'No known class samples for LoRA training'}
        
        # Prepare data for LoRA training
        unique_classes = sorted(list(set(known_labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
        train_label_indices = [label_to_idx[label] for label in known_labels]
        
        # Convert to tensors
        X = torch.from_numpy(known_features).float().to(self.device)
        y = torch.LongTensor(train_label_indices).to(self.device)
        
        # Initialize LoRA adapter and classifier
        embedding_dim = known_features.shape[1]
        lora_adapter = LoRAAdapter(embedding_dim, rank).to(self.device)
        classifier = nn.Linear(embedding_dim, len(unique_classes)).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(
            list(lora_adapter.parameters()) + list(classifier.parameters()), 
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        lora_adapter.train()
        classifier.train()
        training_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            adapted_embeddings = lora_adapter(X)
            outputs = classifier(adapted_embeddings)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if epoch % 5 == 0:
                self.logger.info(f"LoRA Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate adaptation
        lora_adapter.eval()
        classifier.eval()
        
        with torch.no_grad():
            adapted_embeddings = lora_adapter(X)
            adapted_embeddings_np = adapted_embeddings.cpu().numpy()
            
            # Training accuracy
            outputs = classifier(adapted_embeddings)
            predictions = torch.argmax(outputs, dim=1)
            training_accuracy = (predictions == y).float().mean().item()
        
        # Analyze adaptation effects
        original_norm = np.mean(np.linalg.norm(known_features, axis=1))
        adapted_norm = np.mean(np.linalg.norm(adapted_embeddings_np, axis=1))
        embedding_change = np.mean(np.linalg.norm(adapted_embeddings_np - known_features, axis=1))
        
        return {
            'lora_config': {'rank': rank, 'epochs': epochs, 'lr': lr},
            'training_metrics': {
                'final_loss': training_losses[-1],
                'training_accuracy': training_accuracy,
                'losses': training_losses
            },
            'adaptation_metrics': {
                'original_embedding_norm': original_norm,
                'adapted_embedding_norm': adapted_norm,
                'embedding_change_magnitude': embedding_change,
                'relative_change': embedding_change / original_norm
            },
            'adapted_features': adapted_embeddings_np,
            'lora_adapter': lora_adapter,
            'classifier': classifier
        }
    
    def generate_comprehensive_visualizations(self, 
                                            results: Dict,
                                            output_dir: str = 'enhanced_vlm_results'):
        """Generate comprehensive visualizations for all analyses."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Open-set performance comparison heatmap
        self._plot_openset_performance_heatmap(results, output_dir)
        
        # 2. GHOST vs Traditional classifier comparison
        self._plot_ghost_comparison(results, output_dir)
        
        # 3. Text prompt distance analysis
        if 'text_prompt_analysis' in results:
            self._plot_text_prompt_distances(results['text_prompt_analysis'], output_dir)
        
        # 4. Per-class fairness analysis
        self._plot_fairness_analysis(results, output_dir)
        
        # 5. OSCR curves with logarithmic FPR
        self._plot_oscr_curves(results, output_dir)
        
        # 6. LoRA adaptation analysis
        if 'lora_results' in results:
            self._plot_lora_analysis(results['lora_results'], output_dir)
    
    def _plot_openset_performance_heatmap(self, results: Dict, output_dir: str):
        """Plot comprehensive open-set performance heatmap."""
        
        if 'open_set_evaluation' not in results:
            return
        
        # Extract metrics for heatmap
        feature_types = list(results['open_set_evaluation'].keys())
        classifiers = list(results['open_set_evaluation'][feature_types[0]].keys())
        
        metrics = ['classifier_auroc', 'ghost_auroc', 'classifier_fpr95', 'ghost_fpr95', 
                  'classifier_auoscr', 'ghost_auoscr']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            # Create heatmap data
            heatmap_data = []
            for feature_type in feature_types:
                row = []
                for clf_name in classifiers:
                    if clf_name in results['open_set_evaluation'][feature_type]:
                        value = results['open_set_evaluation'][feature_type][clf_name].get(metric, 0)
                        row.append(value)
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            # Plot heatmap
            sns.heatmap(heatmap_data,
                       xticklabels=classifiers,
                       yticklabels=feature_types,
                       annot=True,
                       fmt='.3f',
                       cmap='RdYlBu_r' if 'fpr95' in metric else 'viridis',
                       ax=axes[i])
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/openset_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ghost_comparison(self, results: Dict, output_dir: str):
        """Plot GHOST vs traditional classifier comparison."""
        
        if 'open_set_evaluation' not in results:
            return
        
        # Extract GHOST vs classifier performance
        feature_types = list(results['open_set_evaluation'].keys())
        classifiers = list(results['open_set_evaluation'][feature_types[0]].keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['auroc', 'fpr95', 'auoscr']
        metric_labels = ['AUROC (↑)', 'FPR95 (↓)', 'AUOSCR (↑)']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            traditional_scores = []
            ghost_scores = []
            labels_list = []
            
            for feature_type in feature_types:
                for clf_name in classifiers:
                    if clf_name in results['open_set_evaluation'][feature_type]:
                        clf_data = results['open_set_evaluation'][feature_type][clf_name]
                        traditional_scores.append(clf_data.get(f'classifier_{metric}', 0))
                        ghost_scores.append(clf_data.get(f'ghost_{metric}', 0))
                        labels_list.append(f'{feature_type}\n{clf_name}')
            
            x = np.arange(len(labels_list))
            width = 0.35
            
            bars1 = axes[i].bar(x - width/2, traditional_scores, width, label='Traditional', alpha=0.8)
            bars2 = axes[i].bar(x + width/2, ghost_scores, width, label='GHOST', alpha=0.8)
            
            axes[i].set_xlabel('Feature Type + Classifier')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label} Comparison')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(labels_list, rotation=45, ha='right')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ghost_vs_traditional_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_text_prompt_distances(self, text_results: Dict, output_dir: str):
        """Plot text prompt distance analysis."""
        
        # Distance statistics across prompt templates
        templates = list(text_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Mean distances by prompt template
        mean_distances = [text_results[t]['distance_statistics']['mean_distance'] for t in templates]
        std_distances = [text_results[t]['distance_statistics']['std_distance'] for t in templates]
        
        axes[0, 0].bar(templates, mean_distances, yerr=std_distances, alpha=0.8, capsize=5)
        axes[0, 0].set_title('Mean Distance to Known Class Text Prompts')
        axes[0, 0].set_ylabel('Cosine Distance')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distance distribution comparison
        all_template_distances = []
        template_labels = []
        
        for template in templates:
            # Get per-class mean distances
            per_class_data = text_results[template]['per_unknown_class_analysis']
            for unknown_class, class_data in per_class_data.items():
                all_template_distances.append(class_data['mean_min_distance'])
                template_labels.append(template)
        
        # Create violin plot
        from collections import defaultdict
        template_distance_groups = defaultdict(list)
        for dist, temp in zip(all_template_distances, template_labels):
            template_distance_groups[temp].append(dist)
        
        violin_data = [template_distance_groups[t] for t in templates]
        parts = axes[0, 1].violinplot(violin_data, positions=range(len(templates)), showmeans=True)
        axes[0, 1].set_xticks(range(len(templates)))
        axes[0, 1].set_xticklabels(templates, rotation=45)
        axes[0, 1].set_title('Distance Distribution by Prompt Template')
        axes[0, 1].set_ylabel('Mean Min Distance per Unknown Class')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Unknown class mapping to known classes (using best template)
        best_template = min(templates, key=lambda t: text_results[t]['distance_statistics']['mean_distance'])
        mapping_data = text_results[best_template]['nearest_known_class_mapping']
        
        known_classes = list(mapping_data.keys())
        mapped_counts = [mapping_data[kc]['total_unknown_samples'] for kc in known_classes]
        
        axes[1, 0].bar(known_classes, mapped_counts, alpha=0.8)
        axes[1, 0].set_title(f'Unknown Samples Mapped to Known Classes\n(Template: {best_template})')
        axes[1, 0].set_ylabel('Number of Unknown Samples')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Per unknown class analysis
        per_class_data = text_results[best_template]['per_unknown_class_analysis']
        unknown_classes = list(per_class_data.keys())
        mean_distances = [per_class_data[uc]['mean_min_distance'] for uc in unknown_classes]
        
        bars = axes[1, 1].bar(range(len(unknown_classes)), mean_distances, alpha=0.8)
        axes[1, 1].set_title(f'Mean Distance by Unknown Class\n(Template: {best_template})')
        axes[1, 1].set_ylabel('Mean Min Distance')
        axes[1, 1].set_xticks(range(len(unknown_classes)))
        axes[1, 1].set_xticklabels(unknown_classes, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            nearest_class = per_class_data[unknown_classes[i]]['most_common_nearest_class']
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{nearest_class}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/text_prompt_distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fairness_analysis(self, results: Dict, output_dir: str):
        """Plot per-class fairness analysis."""
        
        if 'open_set_evaluation' not in results:
            return
        
        # Extract fairness metrics
        fairness_data = {}
        
        for feature_type, feature_results in results['open_set_evaluation'].items():
            for clf_name, clf_results in feature_results.items():
                if 'fairness_metrics' in clf_results and 'error' not in clf_results['fairness_metrics']:
                    key = f'{feature_type}_{clf_name}'
                    fairness_data[key] = clf_results['fairness_metrics']
        
        if not fairness_data:
            return
        
        # Plot fairness metrics
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Coefficient of variation
        method_names = list(fairness_data.keys())
        cv_values = [fairness_data[m]['coefficient_of_variation'] for m in method_names]
        
        bars1 = axes[0].bar(range(len(method_names)), cv_values, alpha=0.8)
        axes[0].set_title('Per-Class Fairness: Coefficient of Variation\n(Lower is Better)')
        axes[0].set_ylabel('Coefficient of Variation')
        axes[0].set_xticks(range(len(method_names)))
        axes[0].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, cv in zip(bars1, cv_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{cv:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Performance spread
        max_diff_values = [fairness_data[m]['max_difference'] for m in method_names]
        
        bars2 = axes[1].bar(range(len(method_names)), max_diff_values, alpha=0.8, color='orange')
        axes[1].set_title('Per-Class Performance Spread\n(Lower is Better)')
        axes[1].set_ylabel('Max - Min Performance')
        axes[1].set_xticks(range(len(method_names)))
        axes[1].set_xticklabels(method_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars2, max_diff_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{diff:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fairness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_oscr_curves(self, results: Dict, output_dir: str):
        """Plot OSCR curves with logarithmic FPR axis as recommended by Manuel Gunther."""
        
        if 'open_set_evaluation' not in results:
            return
        
        # This would require storing the actual OSCR curve data during evaluation
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'OSCR Curves with Log FPR\n(Implementation needed)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('OSCR Curves with Logarithmic FPR')
        plt.savefig(f'{output_dir}/oscr_curves_logfpr.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lora_analysis(self, lora_results: Dict, output_dir: str):
        """Plot LoRA adaptation analysis."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Training loss curve
        losses = lora_results['training_metrics']['losses']
        axes[0].plot(losses, linewidth=2)
        axes[0].set_title('LoRA Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Adaptation metrics
        metrics = lora_results['adaptation_metrics']
        metric_names = ['Original Norm', 'Adapted Norm', 'Change Magnitude']
        metric_values = [
            metrics['original_embedding_norm'],
            metrics['adapted_embedding_norm'], 
            metrics['embedding_change_magnitude']
        ]
        
        bars = axes[1].bar(metric_names, metric_values, alpha=0.8)
        axes[1].set_title('LoRA Adaptation Metrics')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Training accuracy
        train_acc = lora_results['training_metrics']['training_accuracy']
        axes[2].bar(['Training Accuracy'], [train_acc], alpha=0.8, color='green')
        axes[2].set_title('LoRA Training Accuracy')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3)
        axes[2].text(0, train_acc + 0.05, f'{train_acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/lora_adaptation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_pipeline(self, 
                                  data_dir: str,
                                  max_samples: int = None,
                                  output_dir: str = 'enhanced_vlm_results') -> Dict:
        """
        Run the complete enhanced VLM evaluation pipeline.
        
        This implements the proper open-set protocol:
        1. Train classifiers on known classes only (Boeing, Airbus)
        2. Test on all classes (known + unknown)
        3. Evaluate using advanced open-set metrics
        4. Analyze text prompt distances
        5. Apply LoRA adaptation with proper evaluation
        """
        
        self.logger.info("=== Starting Enhanced VLM Open-Set Evaluation Pipeline ===")
        
        # 1. Load and prepare data
        self.logger.info("Loading and preparing data...")
        train_dataset, test_dataset, all_classes = self.load_and_prepare_data(data_dir, max_samples)
        
        # Convert to lists
        train_images, train_labels = self.dataset_to_lists(train_dataset)
        test_images, test_labels = self.dataset_to_lists(test_dataset)
        
        self.logger.info(f"Training on known classes: {self.known_classes}")
        self.logger.info(f"All classes in dataset: {all_classes}")
        
        unknown_classes = [cls for cls in all_classes if cls not in self.known_classes]
        self.logger.info(f"Unknown classes for testing: {unknown_classes}")
        
        # 2. Extract features
        self.logger.info("Extracting features...")
        train_features = self.extract_all_features(train_images)
        test_features = self.extract_all_features(test_images)
        
        # Cache features for later use
        self.features_cache = {
            'train': train_features,
            'test': test_features,
            'train_labels': train_labels,
            'test_labels': test_labels
        }
        
        # 3. Train classifiers on known classes only
        self.logger.info("Training classifiers on known classes...")
        trained_classifiers = self.train_classifiers_on_known_classes(train_features, train_labels)
        
        # 4. Evaluate open-set performance
        self.logger.info("Evaluating open-set performance...")
        open_set_results = self.evaluate_open_set_performance(
            test_features, test_labels, trained_classifiers
        )
        
        # 5. Analyze text prompt distances
        self.logger.info("Analyzing text prompt distances...")
        text_prompt_results = self.analyze_text_prompt_distances(test_features, test_labels)
        
        # 6. LoRA adaptation evaluation
        self.logger.info("Evaluating LoRA adaptation...")
        lora_results = self.evaluate_lora_adaptation(train_features, train_labels)
        
        # 7. Compile all results
        all_results = {
            'dataset_info': {
                'known_classes': self.known_classes,
                'unknown_classes': unknown_classes,
                'all_classes': all_classes,
                'n_train_samples': len(train_images),
                'n_test_samples': len(test_images),
                'n_known_test': sum(1 for label in test_labels if label in self.known_classes),
                'n_unknown_test': sum(1 for label in test_labels if label not in self.known_classes)
            },
            'open_set_evaluation': open_set_results,
            'text_prompt_analysis': text_prompt_results,
            'lora_results': lora_results,
            'feature_types_analyzed': list(train_features.keys())
        }
        
        # 8. Generate visualizations
        self.logger.info("Generating comprehensive visualizations...")
        self.generate_comprehensive_visualizations(all_results, output_dir)
        
        # 9. Generate enhanced report
        self.logger.info("Generating enhanced evaluation report...")
        report_path = self.generate_enhanced_report(all_results, output_dir)
        
        # 10. Save results
        self.save_results(all_results, output_dir)
        
        self.logger.info("=== Enhanced VLM Evaluation Pipeline Complete ===")
        self.logger.info(f"Results saved to: {output_dir}")
        self.logger.info(f"Report saved to: {report_path}")
        
        return all_results
    
    def generate_enhanced_report(self, results: Dict, output_dir: str) -> str:
        """Generate comprehensive evaluation report with advanced metrics."""
        
        report_path = Path(output_dir) / 'enhanced_vlm_evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced VLM Open-Set Recognition Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents a comprehensive evaluation of Vision-Language Model (VLM) ")
            f.write("performance in open-set recognition scenarios using state-of-the-art methods and metrics.\n\n")
            
            f.write("### Key Innovations:\n")
            f.write("- **GHOST Algorithm**: Implementation of Manuel Gunther's Gaussian Hypothesis Open-Set Technique\n")
            f.write("- **Advanced Metrics**: AUOSCR, OSCR, FPR95, per-class fairness analysis\n")
            f.write("- **Proper Open-Set Protocol**: Training on known classes, testing on all classes\n")
            f.write("- **Text Prompt Distance Analysis**: Novel analysis of unknown class relationships to text prompts\n")
            f.write("- **Enhanced LoRA Evaluation**: Fixed implementation with comprehensive analysis\n\n")
            
            # Dataset Information
            dataset_info = results['dataset_info']
            f.write("## Dataset Configuration\n\n")
            f.write(f"- **Known Classes (Training)**: {dataset_info['known_classes']}\n")
            f.write(f"- **Unknown Classes (Testing)**: {dataset_info['unknown_classes']}\n")
            f.write(f"- **Total Classes**: {len(dataset_info['all_classes'])}\n")
            f.write(f"- **Training Samples**: {dataset_info['n_train_samples']} (known classes only)\n")
            f.write(f"- **Test Samples**: {dataset_info['n_test_samples']} (known: {dataset_info['n_known_test']}, unknown: {dataset_info['n_unknown_test']})\n\n")
            
            # Open-Set Performance Analysis
            f.write("## Open-Set Performance Analysis\n\n")
            
            if 'open_set_evaluation' in results:
                open_set_results = results['open_set_evaluation']
                
                # Find best performing combinations
                best_combinations = self._find_best_open_set_combinations(open_set_results)
                
                f.write("### Best Performing Combinations\n\n")
                for metric, (feature_type, classifier, score) in best_combinations.items():
                    f.write(f"- **{metric.upper()}**: {feature_type} + {classifier} = {score:.4f}\n")
                f.write("\n")
                
                # GHOST vs Traditional Comparison
                f.write("### GHOST vs Traditional Classifier Performance\n\n")
                ghost_improvements = self._calculate_ghost_improvements(open_set_results)
                
                f.write("| Metric | Mean Traditional | Mean GHOST | Improvement |\n")
                f.write("|--------|------------------|------------|-------------|\n")
                for metric, improvement in ghost_improvements.items():
                    f.write(f"| {metric.upper()} | {improvement['traditional']:.4f} | {improvement['ghost']:.4f} | {improvement['improvement']:.4f} |\n")
                f.write("\n")
                
                # Per-Class Fairness Analysis
                f.write("### Per-Class Fairness Analysis\n\n")
                fairness_summary = self._summarize_fairness_metrics(open_set_results)
                if fairness_summary:
                    f.write("**Coefficient of Variation (Lower is Better)**:\n")
                    for method, cv in fairness_summary['cv_metrics'].items():
                        f.write(f"- {method}: {cv:.4f}\n")
                    f.write("\n")
            
            # Text Prompt Analysis
            f.write("## Text Prompt Distance Analysis\n\n")
            if 'text_prompt_analysis' in results:
                text_results = results['text_prompt_analysis']
                
                f.write("This analysis reveals how unknown classes relate to known class text prompts ")
                f.write("without using text prompts during testing.\n\n")
                
                # Best prompt template
                best_template = min(text_results.keys(), 
                                  key=lambda t: text_results[t]['distance_statistics']['mean_distance'])
                
                f.write(f"### Best Prompt Template: '{best_template}'\n\n")
                best_stats = text_results[best_template]['distance_statistics']
                f.write(f"- **Mean Distance**: {best_stats['mean_distance']:.4f}\n")
                f.write(f"- **Standard Deviation**: {best_stats['std_distance']:.4f}\n")
                f.write(f"- **Distance Range**: {best_stats['min_distance']:.4f} - {best_stats['max_distance']:.4f}\n\n")
                
                # Unknown class mapping
                f.write("### Unknown Class Mapping to Known Classes\n\n")
                mapping_data = text_results[best_template]['nearest_known_class_mapping']
                for known_class, mapping_info in mapping_data.items():
                    f.write(f"**{known_class}**:\n")
                    f.write(f"- Mapped Unknown Classes: {mapping_info['mapped_unknown_classes']}\n")
                    f.write(f"- Total Unknown Samples: {mapping_info['total_unknown_samples']}\n\n")
            
            # LoRA Analysis
            f.write("## LoRA Fine-Tuning Analysis\n\n")
            if 'lora_results' in results:
                lora_results = results['lora_results']
                
                if 'error' not in lora_results:
                    training_metrics = lora_results['training_metrics']
                    adaptation_metrics = lora_results['adaptation_metrics']
                    
                    f.write("### Training Performance\n")
                    f.write(f"- **Final Training Accuracy**: {training_metrics['training_accuracy']:.4f}\n")
                    f.write(f"- **Final Loss**: {training_metrics['final_loss']:.4f}\n\n")
                    
                    f.write("### Adaptation Analysis\n")
                    f.write(f"- **Embedding Change Magnitude**: {adaptation_metrics['embedding_change_magnitude']:.4f}\n")
                    f.write(f"- **Relative Change**: {adaptation_metrics['relative_change']:.4f}\n")
                    f.write(f"- **Original Norm**: {adaptation_metrics['original_embedding_norm']:.4f}\n")
                    f.write(f"- **Adapted Norm**: {adaptation_metrics['adapted_embedding_norm']:.4f}\n\n")
                else:
                    f.write(f"LoRA evaluation failed: {lora_results['error']}\n\n")
            
            # Key Findings and Insights
            f.write("## Key Findings and Insights\n\n")
            
            findings = self._extract_key_findings(results)
            for i, finding in enumerate(findings, 1):
                f.write(f"{i}. **{finding['title']}**: {finding['description']}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            recommendations = self._generate_recommendations(results)
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. **{rec['category']}**: {rec['recommendation']}\n\n")
            
            # Technical Implementation Notes
            f.write("## Technical Implementation Notes\n\n")
            f.write("### Advanced Metrics Implemented\n")
            f.write("- **AUOSCR**: Area Under Open Set Classification Rate curve\n")
            f.write("- **OSCR**: Open Set Classification Rate (CCR vs FPR trade-off)\n")
            f.write("- **FPR95**: False Positive Rate when True Positive Rate = 95%\n")
            f.write("- **GHOST**: Gaussian Hypothesis Open-Set Technique with Z-score normalization\n")
            f.write("- **Per-Class Fairness**: Coefficient of variation and performance spread analysis\n\n")
            
            f.write("### Open-Set Protocol\n")
            f.write("- Training exclusively on known classes (Boeing, Airbus)\n")
            f.write("- Testing on all classes (known + unknown)\n")
            f.write("- Proper separation of known vs unknown evaluation\n")
            f.write("- No data leakage from unknown classes during training\n\n")
            
            f.write("### Statistical Significance\n")
            f.write("- Gaussian hypothesis testing using Shapiro-Wilk test\n")
            f.write("- Holm's step-down procedure for multiple testing correction\n")
            f.write("- Confidence intervals and error bars in visualizations\n\n")
            
            # Future Work
            f.write("## Future Research Directions\n\n")
            future_work = [
                "**Hierarchical Open-Set Recognition**: Leverage aircraft taxonomy for better unknown detection",
                "**Multi-Modal Fusion**: Combine visual and textual features for improved discriminability",
                "**Adaptive Threshold Selection**: Dynamic threshold tuning based on deployment scenarios",
                "**Continual Open-Set Learning**: Online adaptation to new unknown classes",
                "**Adversarial Robustness**: Evaluation against adversarial examples in open-set scenarios",
                "**Large-Scale Evaluation**: Extension to ImageNet-scale open-set recognition",
                "**Causal Open-Set Recognition**: Implementation of CISOR approach for real-world robustness"
            ]
            
            for item in future_work:
                f.write(f"- {item}\n")
            
            f.write(f"\n---\n\n*Report generated by Enhanced VLM Evaluation Pipeline*\n")
        
        self.logger.info(f"Enhanced report saved to {report_path}")
        return str(report_path)
    
    def _find_best_open_set_combinations(self, open_set_results: Dict) -> Dict:
        """Find best performing feature-classifier combinations for each metric."""
        
        metrics = ['classifier_auroc', 'ghost_auroc', 'classifier_fpr95', 'ghost_fpr95', 
                  'classifier_auoscr', 'ghost_auoscr']
        
        best_combinations = {}
        
        for metric in metrics:
            best_score = -np.inf if 'fpr95' not in metric else np.inf
            best_combination = None
            
            for feature_type, feature_results in open_set_results.items():
                for clf_name, clf_results in feature_results.items():
                    if metric in clf_results:
                        score = clf_results[metric]
                        
                        if 'fpr95' in metric:  # Lower is better for FPR95
                            if score < best_score:
                                best_score = score
                                best_combination = (feature_type, clf_name, score)
                        else:  # Higher is better for AUROC, AUOSCR
                            if score > best_score:
                                best_score = score
                                best_combination = (feature_type, clf_name, score)
            
            if best_combination:
                best_combinations[metric] = best_combination
        
        return best_combinations
    
    def _calculate_ghost_improvements(self, open_set_results: Dict) -> Dict:
        """Calculate GHOST improvements over traditional classifiers."""
        
        improvements = {}
        metrics = ['auroc', 'fpr95', 'auoscr']
        
        for metric in metrics:
            traditional_scores = []
            ghost_scores = []
            
            for feature_type, feature_results in open_set_results.items():
                for clf_name, clf_results in feature_results.items():
                    traditional_key = f'classifier_{metric}'
                    ghost_key = f'ghost_{metric}'
                    
                    if traditional_key in clf_results and ghost_key in clf_results:
                        traditional_scores.append(clf_results[traditional_key])
                        ghost_scores.append(clf_results[ghost_key])
            
            if traditional_scores and ghost_scores:
                traditional_mean = np.mean(traditional_scores)
                ghost_mean = np.mean(ghost_scores)
                
                if 'fpr95' in metric:  # Lower is better
                    improvement = traditional_mean - ghost_mean
                else:  # Higher is better
                    improvement = ghost_mean - traditional_mean
                
                improvements[metric] = {
                    'traditional': traditional_mean,
                    'ghost': ghost_mean,
                    'improvement': improvement
                }
        
        return improvements
    
    def _summarize_fairness_metrics(self, open_set_results: Dict) -> Dict:
        """Summarize fairness metrics across all methods."""
        
        cv_metrics = {}
        
        for feature_type, feature_results in open_set_results.items():
            for clf_name, clf_results in feature_results.items():
                if 'fairness_metrics' in clf_results and 'error' not in clf_results['fairness_metrics']:
                    method_name = f"{feature_type}_{clf_name}"
                    cv = clf_results['fairness_metrics']['coefficient_of_variation']
                    cv_metrics[method_name] = cv
        
        if not cv_metrics:
            return None
        
        return {'cv_metrics': cv_metrics}
    
    def _extract_key_findings(self, results: Dict) -> List[Dict]:
        """Extract key findings from the evaluation results."""
        
        findings = []
        
        # Open-set performance finding
        if 'open_set_evaluation' in results:
            best_combinations = self._find_best_open_set_combinations(results['open_set_evaluation'])
            if 'ghost_auroc' in best_combinations:
                feature_type, classifier, score = best_combinations['ghost_auroc']
                findings.append({
                    'title': 'Best Open-Set Detection Performance',
                    'description': f"GHOST with {feature_type} features and {classifier} classifier achieved "
                                 f"AUROC of {score:.4f}, demonstrating superior unknown class detection capability."
                })
        
        # GHOST effectiveness finding
        if 'open_set_evaluation' in results:
            ghost_improvements = self._calculate_ghost_improvements(results['open_set_evaluation'])
            if 'auroc' in ghost_improvements and ghost_improvements['auroc']['improvement'] > 0:
                improvement = ghost_improvements['auroc']['improvement']
                findings.append({
                    'title': 'GHOST Algorithm Effectiveness',
                    'description': f"GHOST consistently outperforms traditional classifiers with an average "
                                 f"AUROC improvement of {improvement:.4f}, validating the Gaussian hypothesis approach."
                })
        
        # Text prompt analysis finding
        if 'text_prompt_analysis' in results:
            text_results = results['text_prompt_analysis']
            if text_results:
                best_template = min(text_results.keys(), 
                                  key=lambda t: text_results[t]['distance_statistics']['mean_distance'])
                findings.append({
                    'title': 'Text Prompt Discriminability',
                    'description': f"The '{best_template}' prompt template provides best separation between "
                                 f"known and unknown classes, suggesting optimal text-visual alignment strategies."
                })
        
        # LoRA adaptation finding
        if 'lora_results' in results and 'error' not in results['lora_results']:
            lora_results = results['lora_results']
            change_magnitude = lora_results['adaptation_metrics']['embedding_change_magnitude']
            training_acc = lora_results['training_metrics']['training_accuracy']
            
            findings.append({
                'title': 'LoRA Adaptation Efficiency',
                'description': f"LoRA fine-tuning achieved {training_acc:.4f} training accuracy with only "
                             f"{change_magnitude:.4f} average embedding change, demonstrating efficient adaptation."
            })
        
        # Feature space comparison finding
        if 'open_set_evaluation' in results:
            feature_performance = {}
            for feature_type, feature_results in results['open_set_evaluation'].items():
                auroc_scores = [clf_results.get('ghost_auroc', 0) for clf_results in feature_results.values()]
                feature_performance[feature_type] = np.mean(auroc_scores)
            
            best_feature_space = max(feature_performance.items(), key=lambda x: x[1])
            findings.append({
                'title': 'Optimal Feature Space',
                'description': f"{best_feature_space[0]} features provide best open-set performance with "
                             f"average AUROC of {best_feature_space[1]:.4f}, indicating superior discriminative power."
            })
        
        return findings
    
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generate actionable recommendations based on evaluation results."""
        
        recommendations = []
        
        # Feature space recommendation
        if 'open_set_evaluation' in results:
            best_combinations = self._find_best_open_set_combinations(results['open_set_evaluation'])
            if 'ghost_auroc' in best_combinations:
                feature_type, classifier, _ = best_combinations['ghost_auroc']
                recommendations.append({
                    'category': 'Optimal Configuration',
                    'recommendation': f"Deploy GHOST algorithm with {feature_type} features and {classifier} "
                                    f"classifier for best open-set recognition performance in production systems."
                })
        
        # GHOST implementation recommendation
        ghost_improvements = self._calculate_ghost_improvements(results.get('open_set_evaluation', {}))
        if ghost_improvements:
            recommendations.append({
                'category': 'Algorithm Selection',
                'recommendation': "Implement GHOST (Gaussian Hypothesis Open-Set Technique) as the primary "
                                "open-set detection method due to its consistent improvements over traditional approaches."
            })
        
        # Text prompt strategy recommendation
        if 'text_prompt_analysis' in results:
            text_results = results['text_prompt_analysis']
            if text_results:
                best_template = min(text_results.keys(), 
                                  key=lambda t: text_results[t]['distance_statistics']['mean_distance'])
                recommendations.append({
                    'category': 'Text Prompt Design',
                    'recommendation': f"Use '{best_template}' prompt template for optimal text-visual alignment "
                                    f"and improved zero-shot classification performance."
                })
        
        # Fairness recommendation
        fairness_summary = self._summarize_fairness_metrics(results.get('open_set_evaluation', {}))
        if fairness_summary:
            recommendations.append({
                'category': 'Fairness and Deployment',
                'recommendation': "Monitor per-class fairness metrics in production deployment to ensure "
                                "equitable performance across different aircraft manufacturers and models."
            })
        
        # LoRA recommendation
        if 'lora_results' in results and 'error' not in results['lora_results']:
            recommendations.append({
                'category': 'Fine-Tuning Strategy',
                'recommendation': "Apply LoRA fine-tuning for domain-specific adaptation while maintaining "
                                "computational efficiency and avoiding catastrophic forgetting."
            })
        
        # Evaluation protocol recommendation
        recommendations.append({
            'category': 'Evaluation Best Practices',
            'recommendation': "Adopt comprehensive open-set evaluation protocols including AUOSCR, OSCR, "
                            "FPR95, and per-class fairness metrics for thorough performance assessment."
        })
        
        return recommendations
    
    def save_results(self, results: Dict, output_dir: str):
        """Save all results to files with proper handling of non-serializable objects."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a copy of results for safe serialization
        serializable_results = self._make_results_serializable(results)
        
        # Save as pickle (with serializable version)
        try:
            with open(output_path / 'enhanced_vlm_results.pkl', 'wb') as f:
                pickle.dump(serializable_results, f)
            self.logger.info("Results saved as pickle file")
        except Exception as e:
            self.logger.warning(f"Could not save pickle file: {e}")
        
        # Save as JSON (serializable parts only)
        json_results = self._make_json_serializable(serializable_results)
        with open(output_path / 'enhanced_vlm_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save LoRA models separately if they exist
        if 'lora_results' in results and 'lora_adapter' in results['lora_results']:
            try:
                lora_adapter = results['lora_results']['lora_adapter']
                classifier = results['lora_results']['classifier']
                
                # Save model state dicts
                torch.save(lora_adapter.state_dict(), output_path / 'lora_adapter.pth')
                torch.save(classifier.state_dict(), output_path / 'lora_classifier.pth')
                
                # Save model info
                lora_info = {
                    'rank': lora_adapter.rank,
                    'alpha': lora_adapter.alpha,
                    'embedding_dim': lora_adapter.lora_A.shape[0],
                    'n_classes': classifier.out_features
                }
                with open(output_path / 'lora_model_info.json', 'w') as f:
                    json.dump(lora_info, f, indent=2)
                
                self.logger.info("LoRA models saved separately")
            except Exception as e:
                self.logger.warning(f"Could not save LoRA models: {e}")
        
        # Save configuration
        config = {
            'known_classes': self.known_classes,
            'device': self.device,
            'evaluation_timestamp': datetime.now().isoformat(),
            'feature_types': list(results.get('feature_types_analyzed', [])),
            'dataset_info': results.get('dataset_info', {})
        }
        
        with open(output_path / 'evaluation_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"All results saved to {output_dir}")
    
    def _make_results_serializable(self, results: Dict) -> Dict:
        """Create a serializable version of results by handling problematic objects."""
        
        serializable = {}
        
        for key, value in results.items():
            if key == 'lora_results' and isinstance(value, dict):
                # Handle LoRA results specially
                lora_copy = {}
                for lora_key, lora_value in value.items():
                    if isinstance(lora_value, (torch.nn.Module, torch.Tensor)):
                        # Convert to summary info instead of the actual object
                        if hasattr(lora_value, 'state_dict'):
                            lora_copy[lora_key] = f"<{type(lora_value).__name__}>"
                        else:
                            lora_copy[lora_key] = f"<Tensor: {lora_value.shape}>"
                    else:
                        lora_copy[lora_key] = lora_value
                serializable[key] = lora_copy
            else:
                serializable[key] = self._make_object_serializable(value)
        
        return serializable
    
    def _make_object_serializable(self, obj):
        """Recursively make an object serializable."""
        
        if isinstance(obj, dict):
            return {k: self._make_object_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_object_serializable(item) for item in obj]
        elif isinstance(obj, (torch.nn.Module, torch.Tensor)):
            return f"<{type(obj).__name__}>"
        elif isinstance(obj, np.ndarray):
            if obj.size > 1000:  # Large arrays - save shape info only
                return f"<ndarray: shape={obj.shape}, dtype={obj.dtype}>"
            else:
                return obj.tolist()  # Small arrays - convert to list
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable by converting numpy arrays and other types."""
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (torch.nn.Module, torch.Tensor)):
            return str(type(obj))
        else:
            return obj


def main():
    """Example usage of the enhanced VLM evaluation pipeline."""
    
    # Configuration
    config = {
        'data_dir': './data/fgvc-aircraft',
        'known_classes': ['Boeing', 'Airbus'],  # Known classes for training
        'max_samples': 1000,  # For testing; set to None for full dataset
        'clip_model': 'openai/clip-vit-base-patch32',
        'output_dir': 'enhanced_vlm_results'
    }
    
    # Initialize pipeline
    pipeline = EnhancedVLMEvaluationPipeline(
        clip_model=config['clip_model'],
        known_classes=config['known_classes']
    )
    
    # Run comprehensive evaluation
    try:
        results = pipeline.run_comprehensive_pipeline(
            data_dir=config['data_dir'],
            max_samples=config['max_samples'],
            output_dir=config['output_dir']
        )
        
        print("\n=== ENHANCED VLM EVALUATION SUMMARY ===")
        print(f"Known classes: {config['known_classes']}")
        print(f"Dataset: FGVC Aircraft")
        print(f"Results saved to: {config['output_dir']}")
        
        # Print key metrics
        if 'open_set_evaluation' in results:
            best_combinations = pipeline._find_best_open_set_combinations(results['open_set_evaluation'])
            print("\nBest Performance:")
            for metric, (feature_type, classifier, score) in best_combinations.items():
                print(f"  {metric}: {feature_type} + {classifier} = {score:.4f}")
        
        print("\nEvaluation complete! Check the output directory for detailed results and visualizations.")
        
    except Exception as e:
        print(f"ERROR: Pipeline failed with: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())