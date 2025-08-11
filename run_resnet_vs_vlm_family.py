#!/usr/bin/env python
"""
ResNet vs VLM Comparison for Aircraft Classification

This script compares ResNet (ImageNet) and VLM (CLIP) embeddings for aircraft
classification, with a focus on:
1. Binary classification (Boeing vs. Airbus)
2. Aircraft model classification
3. OOD detection performance
4. Embedding space analysis

Usage:
    python run_resnet_vs_vlm.py --data_dir ./data/fgvc-aircraft --analyze_all
    python run_resnet_vs_vlm.py --analyze_knn --analyze_embeddings
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from aircraft_family_protocol import AircraftProtocol
from aircraft_family_analysis import (
    setup_output_dir,
    load_and_prepare_data
)
from enhanced_ood_detection_family import (
    knn_ood_detection_optimized,
    visualize_ood_comparison
)
from scipy.spatial.distance import cosine

class ResNetEmbeddingExtractor:
    """ResNet embedding extractor for aircraft images."""
    
    def __init__(self, model_name='resnet50', device=None):
        """
        Initialize ResNet embedding extractor.
        
        Args:
            model_name: Name of the ResNet model to use ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pretrained ResNet model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=True)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
        elif model_name == 'resnet152':
            self.model = models.resnet152(pretrained=True)
        else:  # default to resnet50
            self.model = models.resnet50(pretrained=True)
        
        # Remove the final classification layer to get embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store embeddings
        self.image_embeddings = {}
        
        print(f"Initialized ResNet embedding extractor with {model_name}")
    
    def extract_embeddings(self, df, image_dir, batch_size=32, max_samples=None):
        """
        Extract embeddings for images in the dataframe.
        
        Args:
            df: DataFrame with image filenames
            image_dir: Directory containing the images
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary mapping image filenames to embeddings
        """
        if max_samples:
            df = df.head(max_samples)
        
        # Process images in batches
        with torch.no_grad():
            batch_images = []
            batch_filenames = []
            
            for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Extracting ResNet embeddings")):
                try:
                    img_path = os.path.join(image_dir, row['filename'])
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = self.transform(image).unsqueeze(0)
                    batch_images.append(image_tensor)
                    batch_filenames.append(row['filename'])
                except Exception as e:
                    print(f"Error processing {row['filename']}: {e}")
                
                # Process batch if it's full or at the end
                if len(batch_images) == batch_size or i == len(df) - 1:
                    if batch_images:
                        # Stack images into a batch
                        batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                        
                        # Extract embeddings
                        batch_embeddings = self.model(batch_tensor).squeeze(-1).squeeze(-1)
                        
                        # Normalize embeddings for fair comparison with VLM
                        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
                        
                        # Store embeddings
                        for filename, embedding in zip(batch_filenames, batch_embeddings):
                            self.image_embeddings[filename] = embedding.cpu()
                    
                    # Reset batch
                    batch_images = []
                    batch_filenames = []
        
        print(f"Extracted ResNet embeddings for {len(self.image_embeddings)} images")
        return self.image_embeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ResNet vs VLM Comparison for Aircraft Classification")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="./data/fgvc-aircraft",
                        help="Path to the FGVC Aircraft dataset directory")
    parser.add_argument("--train_csv", type=str, default="train.csv",
                        help="CSV file with training data")
    parser.add_argument("--val_csv", type=str, default="val.csv",
                        help="CSV file with validation data")
    parser.add_argument("--test_csv", type=str, default="test.csv",
                        help="CSV file with test data")
    
    # Model options
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model to use")
    parser.add_argument("--resnet_model", type=str, default="resnet50",
                        help="ResNet model to use (resnet18, resnet34, resnet50, resnet101, resnet152)")
    
    # Analysis options
    parser.add_argument("--analyze_knn", action="store_true",
                        help="Run KNN classification analysis")
    parser.add_argument("--analyze_open_set", action="store_true",
                        help="Run open-set evaluation analysis")
    parser.add_argument("--analyze_embeddings", action="store_true",
                        help="Run embedding distribution analysis")
    parser.add_argument("--analyze_all", action="store_true",
                        help="Run all analyses")
    
    # Analysis specifics
    parser.add_argument("--classification_target", type=str, default="both",
                        help="Classification target: manufacturer, model, or both")
    parser.add_argument("--k_values", type=str, default="1,3,5,10",
                        help="Comma-separated list of k values for KNN")
    
    # UMAP options
    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="resnet_vs_vlm_results",
                        help="Directory to save analysis results")
    parser.add_argument("--timestamp", action="store_true",
                        help="Add timestamp to output directory")
    
    # Execution options
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # If no analysis is specified, run all
    if not any([args.analyze_knn, args.analyze_open_set, args.analyze_embeddings, 
                args.analyze_all]):
        args.analyze_all = True
        print("No specific analysis requested. Running all analyses.")
    
    return args


def knn_classification(
    embeddings, 
    train_df, 
    test_df, 
    k=5, 
    classification_target='manufacturer'
):
    """
    Perform KNN classification using the provided embeddings.
    
    Args:
        embeddings: Dictionary mapping image filenames to embeddings.
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        k: Number of neighbors to use.
        classification_target: Target for classification ('manufacturer' or 'Classes')
        
    Returns:
        DataFrame with classification results and metrics dict.
    """
    # Extract embeddings for training samples
    train_embeddings = []
    train_labels = []
    train_filenames = []
    
    for _, row in train_df.iterrows():
        filename = row['filename']
        if filename in embeddings:
            # Convert torch tensor to numpy if needed
            embedding = embeddings[filename].numpy() if isinstance(embeddings[filename], torch.Tensor) else embeddings[filename]
            
            # Get label based on classification target
            if classification_target in row:
                train_embeddings.append(embedding)
                train_labels.append(row[classification_target])
                train_filenames.append(filename)
            elif classification_target == 'manufacturer' and 'Classes' in row:
                # Derive manufacturer from class if needed
                if row['Classes'].startswith('7') or row['Classes'].startswith('Boeing'):
                    manufacturer = 'Boeing'
                elif row['Classes'].startswith('A3') or row['Classes'].startswith('Airbus'):
                    manufacturer = 'Airbus'
                else:
                    manufacturer = 'Unknown'
                
                train_embeddings.append(embedding)
                train_labels.append(manufacturer)
                train_filenames.append(filename)
    
    train_embeddings = np.array(train_embeddings)
    
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=min(k, len(train_embeddings)))
    knn.fit(train_embeddings)
    
    # Classify test samples
    results = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"KNN classification (k={k})"):
        filename = row['filename']
        if filename not in embeddings:
            continue
            
        # Get image embedding
        img_embedding = embeddings[filename]
        if isinstance(img_embedding, torch.Tensor):
            img_embedding = img_embedding.numpy()
        img_embedding = img_embedding.reshape(1, -1)
        
        # Find k nearest neighbors
        distances, indices = knn.kneighbors(img_embedding)
        
        # Get neighbor labels
        neighbor_labels = [train_labels[i] for i in indices[0]]
        
        # Get most common label (majority vote)
        from collections import Counter
        label_counts = Counter(neighbor_labels)
        predicted = label_counts.most_common(1)[0][0]
        confidence = label_counts[predicted] / k
        
        # Get true label based on classification target
        if classification_target in row:
            true_label = row[classification_target]
        elif classification_target == 'manufacturer' and 'Classes' in row:
            # Derive manufacturer from class
            if row['Classes'].startswith('7') or row['Classes'].startswith('Boeing'):
                true_label = 'Boeing'
            elif row['Classes'].startswith('A3') or row['Classes'].startswith('Airbus'):
                true_label = 'Airbus'
            else:
                true_label = 'Unknown'
        else:
            true_label = None
        
        # Store result
        results.append({
            'filename': filename,
            f'true_{classification_target}': true_label,
            'predicted': predicted,
            'confidence': confidence,
            'is_correct': predicted == true_label,
            'mean_distance': distances[0].mean()
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    metrics = {}
    
    if len(results_df) > 0 and f'true_{classification_target}' in results_df.columns:
        y_true = results_df[f'true_{classification_target}'].values
        y_pred = results_df['predicted'].values
        
        # Calculate metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        unique_labels = sorted(set(y_true) | set(y_pred))
        metrics['confusion_matrix'] = confusion_matrix(
            y_true, y_pred, labels=unique_labels
        )
        metrics['classes'] = unique_labels
    else:
        metrics['accuracy'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
        metrics['confusion_matrix'] = np.zeros((1, 1))
        metrics['classes'] = ["Unknown"]
    
    return results_df, metrics


def compare_knn_classification(
    resnet_embeddings, 
    vlm_embeddings, 
    data, 
    k_values, 
    output_dir, 
    classification_target='manufacturer'
):
    """
    Compare ResNet and VLM embeddings for KNN classification.
    
    Args:
        resnet_embeddings: Dictionary with ResNet embeddings.
        vlm_embeddings: Dictionary with VLM embeddings.
        data: Dictionary with data.
        k_values: List of k values to test.
        output_dir: Output directory.
        classification_target: Target for classification ('manufacturer', 'Classes', or 'model')
        
    Returns:
        Dictionary with metrics for both models.
    """
    print(f"\nComparing KNN classification for {classification_target}...")
    
    # Create output directory
    knn_dir = os.path.join(output_dir, f"knn_{classification_target}")
    os.makedirs(knn_dir, exist_ok=True)
    
    # Determine which DataFrames to use
    if classification_target == 'manufacturer':
        # For manufacturer, use filtered data (Boeing/Airbus only)
        train_df = data['train_df_filtered'] if 'train_df_filtered' in data else data['train_df']
        test_df = data['test_df_close'] if 'test_df_close' in data else data['test_df']
    else:
        # For model/class, use all data
        train_df = data['train_df']
        test_df = data['test_df']
    
    # Run KNN classification for different k values
    resnet_metrics = {}
    vlm_metrics = {}
    
    for k in k_values:
        print(f"\nRunning KNN classification with k={k}...")
        
        # ResNet classification
        print("ResNet classification...")
        resnet_results, resnet_k_metrics = knn_classification(
            resnet_embeddings,
            train_df,
            test_df,
            k=k,
            classification_target=classification_target
        )
        
        # Save ResNet results
        resnet_results.to_csv(
            os.path.join(knn_dir, f"resnet_k{k}_results.csv"),
            index=False
        )
        
        # VLM classification
        print("VLM classification...")
        vlm_results, vlm_k_metrics = knn_classification(
            vlm_embeddings,
            train_df,
            test_df,
            k=k,
            classification_target=classification_target
        )
        
        # Save VLM results
        vlm_results.to_csv(
            os.path.join(knn_dir, f"vlm_k{k}_results.csv"),
            index=False
        )
        
        # Store metrics
        resnet_metrics[f"k{k}"] = resnet_k_metrics
        vlm_metrics[f"k{k}"] = vlm_k_metrics
        
        # Print comparison
        print(f"\nClassification Comparison (k={k}):")
        print(f"ResNet Accuracy: {resnet_k_metrics['accuracy']:.4f}")
        print(f"VLM Accuracy: {vlm_k_metrics['accuracy']:.4f}")
        print(f"ResNet F1 Score: {resnet_k_metrics['f1']:.4f}")
        print(f"VLM F1 Score: {vlm_k_metrics['f1']:.4f}")
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(k_values))
    width = 0.35
    
    resnet_acc = [resnet_metrics[f"k{k}"]["accuracy"] for k in k_values]
    vlm_acc = [vlm_metrics[f"k{k}"]["accuracy"] for k in k_values]
    
    plt.bar(x - width/2, resnet_acc, width, label='ResNet')
    plt.bar(x + width/2, vlm_acc, width, label='VLM')
    
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.title(f'ResNet vs VLM {classification_target.capitalize()} Classification Accuracy')
    plt.xticks(x, k_values)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(knn_dir, f"accuracy_comparison.png"))
    plt.close()
    
    # Create F1 score comparison
    plt.figure(figsize=(12, 8))
    
    resnet_f1 = [resnet_metrics[f"k{k}"]["f1"] for k in k_values]
    vlm_f1 = [vlm_metrics[f"k{k}"]["f1"] for k in k_values]
    
    plt.bar(x - width/2, resnet_f1, width, label='ResNet')
    plt.bar(x + width/2, vlm_f1, width, label='VLM')
    
    plt.xlabel('k value')
    plt.ylabel('F1 Score')
    plt.title(f'ResNet vs VLM {classification_target.capitalize()} Classification F1 Score')
    plt.xticks(x, k_values)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(knn_dir, f"f1_comparison.png"))
    plt.close()
    
    # Return metrics for summary
    return {
        'resnet': resnet_metrics,
        'vlm': vlm_metrics
    }


def compare_ood_detection(
    resnet_embeddings, 
    vlm_embeddings, 
    vlm_protocol,
    data, 
    k_values, 
    output_dir, 
    classification_target='manufacturer'
):
    """
    Compare ResNet and VLM embeddings for OOD detection.
    
    Args:
        resnet_embeddings: Dictionary with ResNet embeddings.
        vlm_embeddings: Dictionary with VLM embeddings.
        vlm_protocol: AircraftProtocol instance (for running OOD detection)
        data: Dictionary with data.
        k_values: List of k values to test.
        output_dir: Output directory.
        classification_target: Target for classification ('manufacturer' or 'model')
        
    Returns:
        Dictionary with metrics for both models.
    """
    print(f"\nComparing OOD detection performance for {classification_target}...")
    
    # Create output directory
    ood_dir = os.path.join(output_dir, f"ood_{classification_target}")
    os.makedirs(ood_dir, exist_ok=True)
    
    # Store metrics for both models
    resnet_ood_metrics = {}
    vlm_ood_metrics = {}
    
    # For each k value
    for k in k_values:
        print(f"\nRunning OOD detection with k={k}...")
        
        # Create a ResNet "protocol" for running OOD detection with ResNet embeddings
        resnet_protocol = type('', (), {'image_embeddings': resnet_embeddings})()
        
        # For ResNet, we need to copy other attributes from the VLM protocol
        resnet_protocol.known_manufacturers = vlm_protocol.known_manufacturers
        resnet_protocol.get_manufacturer = vlm_protocol.get_manufacturer
        
        # ResNet OOD detection
        print("ResNet OOD detection...")
        resnet_results, resnet_ood_k_metrics = knn_ood_detection_optimized(
            resnet_protocol,
            data['test_df'],  # Full test set including Unknown
            k=k,
            optimization_metric='balanced_accuracy',
            class_type=classification_target
        )
        
        # Save ResNet results
        resnet_results.to_csv(
            os.path.join(ood_dir, f"resnet_ood_k{k}_results.csv"),
            index=False
        )
        
        # Generate ROC curve for ResNet
        plt.figure(figsize=(8, 8))
        plt.plot(resnet_ood_k_metrics['fpr'], resnet_ood_k_metrics['tpr'], color='blue', lw=2,
                 label=f'ROC curve (area = {resnet_ood_k_metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ResNet OOD Detection (k={k})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(ood_dir, f"resnet_ood_k{k}_roc.png"))
        plt.close()
        
        # VLM OOD detection
        print("VLM OOD detection...")
        vlm_results, vlm_ood_k_metrics = knn_ood_detection_optimized(
            vlm_protocol,
            data['test_df'],  # Full test set including Unknown
            k=k,
            optimization_metric='balanced_accuracy',
            class_type=classification_target
        )
        
        # Save VLM results
        vlm_results.to_csv(
            os.path.join(ood_dir, f"vlm_ood_k{k}_results.csv"),
            index=False
        )
        
        # Generate ROC curve for VLM
        plt.figure(figsize=(8, 8))
        plt.plot(vlm_ood_k_metrics['fpr'], vlm_ood_k_metrics['tpr'], color='green', lw=2,
                 label=f'ROC curve (area = {vlm_ood_k_metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'VLM OOD Detection (k={k})')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(ood_dir, f"vlm_ood_k{k}_roc.png"))
        plt.close()
        
        # Store metrics
        resnet_ood_metrics[f"ResNet k{k}"] = resnet_ood_k_metrics
        vlm_ood_metrics[f"VLM k{k}"] = vlm_ood_k_metrics
        
        # Print comparison
        print(f"\nOOD Detection Comparison (k={k}):")
        print(f"ResNet ROC AUC: {resnet_ood_k_metrics['roc_auc']:.4f}")
        print(f"VLM ROC AUC: {vlm_ood_k_metrics['roc_auc']:.4f}")
        print(f"ResNet Normalized Accuracy: {resnet_ood_k_metrics.get('normalized_accuracy', 0):.4f}")
        print(f"VLM Normalized Accuracy: {vlm_ood_k_metrics.get('normalized_accuracy', 0):.4f}")
    
    # Combine metrics for visualization
    combined_metrics = {**resnet_ood_metrics, **vlm_ood_metrics}
    visualize_ood_comparison(combined_metrics, ood_dir)
    
    # Create ROC comparison visualization
    plt.figure(figsize=(10, 8))
    
    # Pick the best k value for each model based on ROC AUC
    best_resnet_k = max(resnet_ood_metrics.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_vlm_k = max(vlm_ood_metrics.items(), key=lambda x: x[1]['roc_auc'])[0]
    
    # Plot the ROC curves with best k values
    plt.plot(resnet_ood_metrics[best_resnet_k]['fpr'], 
             resnet_ood_metrics[best_resnet_k]['tpr'], 
             color='blue', lw=2,
             label=f'ResNet {best_resnet_k} (AUC = {resnet_ood_metrics[best_resnet_k]["roc_auc"]:.3f})')
    
    plt.plot(vlm_ood_metrics[best_vlm_k]['fpr'], 
             vlm_ood_metrics[best_vlm_k]['tpr'], 
             color='green', lw=2,
             label=f'VLM {best_vlm_k} (AUC = {vlm_ood_metrics[best_vlm_k]["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ResNet vs VLM OOD Detection ROC Comparison')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(ood_dir, "roc_comparison.png"))
    plt.close()
    
    # Return metrics for summary
    return {
        'resnet': resnet_ood_metrics,
        'vlm': vlm_ood_metrics
    }


def compare_embeddings_umap(
    resnet_embeddings, 
    vlm_embeddings, 
    df, 
    output_dir, 
    umap_params=None, 
    color_by='manufacturer'
):
    """
    Compare ResNet and VLM embeddings using UMAP visualization.
    
    Args:
        resnet_embeddings: Dictionary mapping filenames to ResNet embeddings.
        vlm_embeddings: Dictionary mapping filenames to VLM embeddings.
        df: DataFrame with image data.
        output_dir: Output directory.
        umap_params: Dictionary with UMAP parameters.
        color_by: Column to use for coloring ('manufacturer' or 'Classes').
        
    Returns:
        None
    """
    print(f"\nComparing embeddings with UMAP visualization (color by {color_by})...")
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set default UMAP parameters if not provided
    if umap_params is None:
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1
        }
    
    # Get common filenames between both embedding sets
    common_filenames = set(resnet_embeddings.keys()) & set(vlm_embeddings.keys())
    common_df = df[df['filename'].isin(common_filenames)]
    
    # Extract embeddings and labels
    resnet_data = []
    vlm_data = []
    labels = []
    
    for _, row in common_df.iterrows():
        filename = row['filename']
        if filename in resnet_embeddings and filename in vlm_embeddings:
            resnet_emb = resnet_embeddings[filename].numpy() if isinstance(resnet_embeddings[filename], torch.Tensor) else resnet_embeddings[filename]
            vlm_emb = vlm_embeddings[filename].numpy() if isinstance(vlm_embeddings[filename], torch.Tensor) else vlm_embeddings[filename]
            
            resnet_data.append(resnet_emb)
            vlm_data.append(vlm_emb)
            
            # Get label based on color_by
            if color_by in row:
                labels.append(row[color_by])
            elif color_by == 'manufacturer' and 'Classes' in row:
                # Infer manufacturer from class
                if row['Classes'].startswith('7') or row['Classes'].startswith('Boeing'):
                    labels.append('Boeing')
                elif row['Classes'].startswith('A3') or row['Classes'].startswith('Airbus'):
                    labels.append('Airbus')
                else:
                    labels.append('Unknown')
            else:
                labels.append('Unknown')
    
    resnet_data = np.array(resnet_data)
    vlm_data = np.array(vlm_data)
    labels = np.array(labels)
    
    # Apply UMAP to both embedding sets
    print(f"Running UMAP with n_neighbors={umap_params.get('n_neighbors', 15)}, min_dist={umap_params.get('min_dist', 0.1)}...")
    
    resnet_reducer = UMAP(
        n_neighbors=umap_params.get('n_neighbors', 15),
        min_dist=umap_params.get('min_dist', 0.1),
        n_components=2,
        random_state=42
    )
    resnet_2d = resnet_reducer.fit_transform(resnet_data)
    
    vlm_reducer = UMAP(
        n_neighbors=umap_params.get('n_neighbors', 15),
        min_dist=umap_params.get('min_dist', 0.1),
        n_components=2,
        random_state=42
    )
    vlm_2d = vlm_reducer.fit_transform(vlm_data)
    
    # Limit to top categories if we have too many classes
    if color_by == 'Classes' and len(np.unique(labels)) > 10:
        # Count occurrences of each class
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Get top 10 classes by count
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_class_names = [cls[0] for cls in top_classes]
        
        # Create a mask for top classes
        mask = np.array([label in top_class_names for label in labels])
        
        # Filtered data
        filtered_labels = labels[mask]
        filtered_resnet_2d = resnet_2d[mask]
        filtered_vlm_2d = vlm_2d[mask]
        
        # Update variables for plotting
        labels = filtered_labels
        resnet_2d = filtered_resnet_2d
        vlm_2d = filtered_vlm_2d
    
    # Define colors for labels
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot ResNet embeddings
    for label in unique_labels:
        mask = labels == label
        axes[0].scatter(
            resnet_2d[mask, 0],
            resnet_2d[mask, 1],
            c=[color_dict[label]],
            label=label,
            alpha=0.7,
            edgecolors='none'
        )
    
    axes[0].set_title('ResNet Embeddings (UMAP)')
    axes[0].set_xlabel('UMAP Dimension 1')
    axes[0].set_ylabel('UMAP Dimension 2')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot VLM embeddings
    for label in unique_labels:
        mask = labels == label
        axes[1].scatter(
            vlm_2d[mask, 0],
            vlm_2d[mask, 1],
            c=[color_dict[label]],
            label=label,
            alpha=0.7,
            edgecolors='none'
        )
    
    axes[1].set_title('VLM Embeddings (UMAP)')
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"resnet_vs_vlm_umap_{color_by}.png"))
    plt.close()
    
    print(f"UMAP comparison saved to {os.path.join(vis_dir, f'resnet_vs_vlm_umap_{color_by}.png')}")
    
    # Analyze embedding distances
    print("\nAnalyzing embedding distances...")
    
    # Project embeddings to same dimension for comparison
    print("Projecting embeddings to common dimension space...")
    from sklearn.decomposition import PCA
    
    # Use PCA to project both to a common dimension (e.g., 256)
    common_dim = min(256, resnet_data.shape[1], vlm_data.shape[1])
    
    # Fit PCA on ResNet data and transform
    resnet_pca = PCA(n_components=common_dim, random_state=42)
    resnet_projected = resnet_pca.fit_transform(resnet_data)
    
    # Fit PCA on VLM data and transform
    vlm_pca = PCA(n_components=common_dim, random_state=42)
    vlm_projected = vlm_pca.fit_transform(vlm_data)
    
    print(f"Projected ResNet embeddings: {resnet_data.shape} -> {resnet_projected.shape}")
    print(f"Projected VLM embeddings: {vlm_data.shape} -> {vlm_projected.shape}")
    
    # Calculate cross-model distances
    distances = {
        'resnet-internal': [],  # Distances between ResNet embeddings
        'vlm-internal': [],     # Distances between VLM embeddings
        'cross-model': []       # Distances between ResNet and VLM embeddings of the same image
    }
    
    # Sample to avoid too many computations
    max_pairs = 1000
    random_indices = np.random.choice(len(resnet_projected), min(max_pairs, len(resnet_projected)), replace=False)
    
    for i in random_indices:
        # Calculate distance between the same image's embeddings from different models (using projected embeddings)
        distances['cross-model'].append(
            cosine(resnet_projected[i], vlm_projected[i])
        )
        
        # Sample a few other images to calculate internal distances (using original embeddings)
        other_indices = np.random.choice(len(resnet_data), min(5, len(resnet_data)), replace=False)
        for j in other_indices:
            if i != j:
                distances['resnet-internal'].append(
                    cosine(resnet_data[i], resnet_data[j])
                )
                distances['vlm-internal'].append(
                    cosine(vlm_data[i], vlm_data[j])
                )
    
    # Create distance distribution visualization
    plt.figure(figsize=(10, 6))
    
    for key, vals in distances.items():
        plt.hist(vals, alpha=0.7, bins=30, label=key, density=True)
    
    plt.title('Distribution of Cosine Distances Between Embeddings')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(os.path.join(vis_dir, "embedding_distance_distributions.png"))
    plt.close()
    
    # Calculate statistics
    stats = {}
    for key, vals in distances.items():
        stats[key] = {
            'mean': np.mean(vals),
            'median': np.median(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals)
        }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.to_csv(os.path.join(vis_dir, "embedding_distance_statistics.csv"))
    
    # Print statistics
    print("\nEmbedding Distance Statistics:")
    for key, vals in stats.items():
        print(f"{key}: Mean={vals['mean']:.4f}, Median={vals['median']:.4f}, Std={vals['std']:.4f}")
    
    return stats


def create_summary_report(metrics, output_dir, distance_stats=None):
    """
    Create a comprehensive summary report.
    
    Args:
        metrics: Dictionary with metrics from all analyses.
        output_dir: Output directory.
        distance_stats: Optional distance statistics from embedding analysis.
    """
    print("\nGenerating comparison report...")
    
    report_path = os.path.join(output_dir, "resnet_vs_vlm_comparison.md")
    
    with open(report_path, 'w') as f:
        f.write("# ResNet vs VLM Comparison for Aircraft Classification\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model information
        f.write("## Model Information\n\n")
        f.write(f"- ResNet Model: ImageNet pretrained (ResNet-50)\n")
        f.write(f"- VLM Model: CLIP (ViT-B/32)\n\n")
        
        # Embedding space comparison
        f.write("## Embedding Space Comparison\n\n")
        
        # Add UMAP visualizations
        f.write("### UMAP Visualization\n\n")
        f.write("**By Manufacturer:**\n\n")
        f.write("![ResNet vs VLM UMAP (Manufacturer)](visualizations/resnet_vs_vlm_umap_manufacturer.png)\n\n")
        
        f.write("**By Aircraft Model:**\n\n")
        f.write("![ResNet vs VLM UMAP (Model)](visualizations/resnet_vs_vlm_umap_Classes.png)\n\n")
        
        f.write("### Embedding Distance Analysis\n\n")
        f.write("![Embedding Distance Distributions](visualizations/embedding_distance_distributions.png)\n\n")
        
        # Load statistics if available
        stats_path = os.path.join(output_dir, "visualizations", "embedding_distance_statistics.csv")
        if os.path.exists(stats_path):
            stats_df = pd.read_csv(stats_path, index_col=0)
            
            f.write("#### Distance Statistics\n\n")
            f.write("| Distance Type | Mean | Median | Std | Min | Max |\n")
            f.write("|--------------|------|--------|-----|-----|-----|\n")
            
            for idx in stats_df.index:
                row = stats_df.loc[idx]
                f.write(f"| {idx} | {row['mean']:.4f} | {row['median']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
            
            f.write("\n")
        
        # Classification comparison - manufacturer
        if 'classification_manufacturer' in metrics:
            f.write("## Manufacturer Classification Performance\n\n")
            
            f.write("### Accuracy Comparison\n\n")
            f.write("![Accuracy Comparison](knn_manufacturer/accuracy_comparison.png)\n\n")
            
            f.write("### F1 Score Comparison\n\n")
            f.write("![F1 Score Comparison](knn_manufacturer/f1_comparison.png)\n\n")
            
            # Add table with best results
            f.write("### Best Classification Results\n\n")
            f.write("| Model | Best k | Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|--------|----------|-----------|--------|----------|\n")
            
            # Find best ResNet results
            best_resnet_k = max(metrics['classification_manufacturer']['resnet'].items(), key=lambda x: x[1]['accuracy'])
            best_resnet_metrics = best_resnet_k[1]
            
            # Find best VLM results
            best_vlm_k = max(metrics['classification_manufacturer']['vlm'].items(), key=lambda x: x[1]['accuracy'])
            best_vlm_metrics = best_vlm_k[1]
            
            f.write(f"| ResNet | {best_resnet_k[0][1:]} | {best_resnet_metrics['accuracy']:.4f} | ")
            f.write(f"{best_resnet_metrics['precision']:.4f} | {best_resnet_metrics['recall']:.4f} | ")
            f.write(f"{best_resnet_metrics['f1']:.4f} |\n")
            
            f.write(f"| VLM | {best_vlm_k[0][1:]} | {best_vlm_metrics['accuracy']:.4f} | ")
            f.write(f"{best_vlm_metrics['precision']:.4f} | {best_vlm_metrics['recall']:.4f} | ")
            f.write(f"{best_vlm_metrics['f1']:.4f} |\n\n")
        
        # Classification comparison - model
        if 'classification_model' in metrics:
            f.write("## Aircraft Model Classification Performance\n\n")
            
            f.write("### Accuracy Comparison\n\n")
            f.write("![Accuracy Comparison](knn_Classes/accuracy_comparison.png)\n\n")
            
            f.write("### F1 Score Comparison\n\n")
            f.write("![F1 Score Comparison](knn_Classes/f1_comparison.png)\n\n")
            
            # Add table with best results
            f.write("### Best Classification Results\n\n")
            f.write("| Model | Best k | Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|--------|----------|-----------|--------|----------|\n")
            
            # Find best ResNet results
            best_resnet_k = max(metrics['classification_model']['resnet'].items(), key=lambda x: x[1]['accuracy'])
            best_resnet_metrics = best_resnet_k[1]
            
            # Find best VLM results
            best_vlm_k = max(metrics['classification_model']['vlm'].items(), key=lambda x: x[1]['accuracy'])
            best_vlm_metrics = best_vlm_k[1]
            
            f.write(f"| ResNet | {best_resnet_k[0][1:]} | {best_resnet_metrics['accuracy']:.4f} | ")
            f.write(f"{best_resnet_metrics['precision']:.4f} | {best_resnet_metrics['recall']:.4f} | ")
            f.write(f"{best_resnet_metrics['f1']:.4f} |\n")
            
            f.write(f"| VLM | {best_vlm_k[0][1:]} | {best_vlm_metrics['accuracy']:.4f} | ")
            f.write(f"{best_vlm_metrics['precision']:.4f} | {best_vlm_metrics['recall']:.4f} | ")
            f.write(f"{best_vlm_metrics['f1']:.4f} |\n\n")
        
        # OOD detection comparison - manufacturer
        if 'ood_manufacturer' in metrics:
            f.write("## Open-Set Evaluation Comparison (Manufacturer)\n\n")
            
            f.write("### ROC Curve Comparison\n\n")
            f.write("![ROC Curve Comparison](ood_manufacturer/roc_comparison.png)\n\n")
            
            f.write("### Performance Metrics Comparison\n\n")
            f.write("![Metrics Comparison](ood_manufacturer/ood_metrics_comparison.png)\n\n")
            
            # Add table with best results
            f.write("### Best OOD Detection Results\n\n")
            f.write("| Model | Best k | ROC AUC | Normalized Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|--------|---------|---------------------|-----------|--------|----------|\n")
            
            # Find best ResNet results
            best_resnet_ood_k = max(metrics['ood_manufacturer']['resnet'].items(), key=lambda x: x[1]['roc_auc'])
            best_resnet_ood_metrics = best_resnet_ood_k[1]
            
            # Find best VLM results
            best_vlm_ood_k = max(metrics['ood_manufacturer']['vlm'].items(), key=lambda x: x[1]['roc_auc'])
            best_vlm_ood_metrics = best_vlm_ood_k[1]
            
            f.write(f"| ResNet | {best_resnet_ood_k[0].split('k')[1]} | {best_resnet_ood_metrics['roc_auc']:.4f} | ")
            f.write(f"{best_resnet_ood_metrics.get('normalized_accuracy', 0):.4f} | {best_resnet_ood_metrics['precision']:.4f} | ")
            f.write(f"{best_resnet_ood_metrics['recall']:.4f} | {best_resnet_ood_metrics['f1']:.4f} |\n")
            
            f.write(f"| VLM | {best_vlm_ood_k[0].split('k')[1]} | {best_vlm_ood_metrics['roc_auc']:.4f} | ")
            f.write(f"{best_vlm_ood_metrics.get('normalized_accuracy', 0):.4f} | {best_vlm_ood_metrics['precision']:.4f} | ")
            f.write(f"{best_vlm_ood_metrics['recall']:.4f} | {best_vlm_ood_metrics['f1']:.4f} |\n\n")
        
        # OOD detection comparison - model
        if 'ood_model' in metrics:
            f.write("## Open-Set Evaluation Comparison (Aircraft Model)\n\n")
            
            f.write("### ROC Curve Comparison\n\n")
            f.write("![ROC Curve Comparison](ood_Classes/roc_comparison.png)\n\n")
            
            f.write("### Performance Metrics Comparison\n\n")
            f.write("![Metrics Comparison](ood_Classes/ood_metrics_comparison.png)\n\n")
            
            # Add table with best results
            f.write("### Best OOD Detection Results\n\n")
            f.write("| Model | Best k | ROC AUC | Normalized Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|--------|---------|---------------------|-----------|--------|----------|\n")
            
            # Find best ResNet results
            best_resnet_ood_k = max(metrics['ood_model']['resnet'].items(), key=lambda x: x[1]['roc_auc'])
            best_resnet_ood_metrics = best_resnet_ood_k[1]
            
            # Find best VLM results
            best_vlm_ood_k = max(metrics['ood_model']['vlm'].items(), key=lambda x: x[1]['roc_auc'])
            best_vlm_ood_metrics = best_vlm_ood_k[1]
            
            f.write(f"| ResNet | {best_resnet_ood_k[0].split('k')[1]} | {best_resnet_ood_metrics['roc_auc']:.4f} | ")
            f.write(f"{best_resnet_ood_metrics.get('normalized_accuracy', 0):.4f} | {best_resnet_ood_metrics['precision']:.4f} | ")
            f.write(f"{best_resnet_ood_metrics['recall']:.4f} | {best_resnet_ood_metrics['f1']:.4f} |\n")
            
            f.write(f"| VLM | {best_vlm_ood_k[0].split('k')[1]} | {best_vlm_ood_metrics['roc_auc']:.4f} | ")
            f.write(f"{best_vlm_ood_metrics.get('normalized_accuracy', 0):.4f} | {best_vlm_ood_metrics['precision']:.4f} | ")
            f.write(f"{best_vlm_ood_metrics['recall']:.4f} | {best_vlm_ood_metrics['f1']:.4f} |\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Add model comparison conclusion
        if ('classification_manufacturer' in metrics and 
            'classification_model' in metrics):
            
            # Manufacturer comparison
            best_resnet_manuf = max(metrics['classification_manufacturer']['resnet'].items(), 
                                    key=lambda x: x[1]['accuracy'])[1]['accuracy']
            best_vlm_manuf = max(metrics['classification_manufacturer']['vlm'].items(), 
                                key=lambda x: x[1]['accuracy'])[1]['accuracy']
            manuf_diff = best_vlm_manuf - best_resnet_manuf
            
            # Model comparison
            best_resnet_model = max(metrics['classification_model']['resnet'].items(), 
                                    key=lambda x: x[1]['accuracy'])[1]['accuracy']
            best_vlm_model = max(metrics['classification_model']['vlm'].items(), 
                                key=lambda x: x[1]['accuracy'])[1]['accuracy']
            model_diff = best_vlm_model - best_resnet_model
            
            f.write("### Classification Performance\n\n")
            
            if manuf_diff > 0.05:
                f.write(f"- **Manufacturer Classification**: VLM significantly outperforms ResNet by {manuf_diff*100:.1f}% accuracy\n")
            elif manuf_diff > 0:
                f.write(f"- **Manufacturer Classification**: VLM slightly outperforms ResNet by {manuf_diff*100:.1f}% accuracy\n")
            elif manuf_diff < -0.05:
                f.write(f"- **Manufacturer Classification**: ResNet significantly outperforms VLM by {-manuf_diff*100:.1f}% accuracy\n")
            elif manuf_diff < 0:
                f.write(f"- **Manufacturer Classification**: ResNet slightly outperforms VLM by {-manuf_diff*100:.1f}% accuracy\n")
            else:
                f.write("- **Manufacturer Classification**: ResNet and VLM perform similarly\n")
                
            if model_diff > 0.05:
                f.write(f"- **Aircraft Model Classification**: VLM significantly outperforms ResNet by {model_diff*100:.1f}% accuracy\n")
            elif model_diff > 0:
                f.write(f"- **Aircraft Model Classification**: VLM slightly outperforms ResNet by {model_diff*100:.1f}% accuracy\n")
            elif model_diff < -0.05:
                f.write(f"- **Aircraft Model Classification**: ResNet significantly outperforms VLM by {-model_diff*100:.1f}% accuracy\n")
            elif model_diff < 0:
                f.write(f"- **Aircraft Model Classification**: ResNet slightly outperforms VLM by {-model_diff*100:.1f}% accuracy\n")
            else:
                f.write("- **Aircraft Model Classification**: ResNet and VLM perform similarly\n")
        
        # Add OOD detection conclusion
        if ('ood_manufacturer' in metrics or 
            'ood_model' in metrics):
            
            f.write("\n### OOD Detection Performance\n\n")
            
            if 'ood_manufacturer' in metrics:
                # Manufacturer OOD comparison
                best_resnet_ood = max(metrics['ood_manufacturer']['resnet'].items(), 
                                     key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
                best_vlm_ood = max(metrics['ood_manufacturer']['vlm'].items(), 
                                  key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
                ood_diff = best_vlm_ood - best_resnet_ood
                
                if ood_diff > 0.05:
                    f.write(f"- **Manufacturer OOD Detection**: VLM significantly outperforms ResNet with ROC AUC difference of {ood_diff:.3f}\n")
                elif ood_diff > 0:
                    f.write(f"- **Manufacturer OOD Detection**: VLM slightly outperforms ResNet with ROC AUC difference of {ood_diff:.3f}\n")
                elif ood_diff < -0.05:
                    f.write(f"- **Manufacturer OOD Detection**: ResNet significantly outperforms VLM with ROC AUC difference of {-ood_diff:.3f}\n")
                elif ood_diff < 0:
                    f.write(f"- **Manufacturer OOD Detection**: ResNet slightly outperforms VLM with ROC AUC difference of {-ood_diff:.3f}\n")
                else:
                    f.write("- **Manufacturer OOD Detection**: ResNet and VLM perform similarly\n")
            
            if 'ood_model' in metrics:
                # Model OOD comparison
                best_resnet_model_ood = max(metrics['ood_model']['resnet'].items(), 
                                          key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
                best_vlm_model_ood = max(metrics['ood_model']['vlm'].items(), 
                                       key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
                model_ood_diff = best_vlm_model_ood - best_resnet_model_ood
                
                if model_ood_diff > 0.05:
                    f.write(f"- **Aircraft Model OOD Detection**: VLM significantly outperforms ResNet with ROC AUC difference of {model_ood_diff:.3f}\n")
                elif model_ood_diff > 0:
                    f.write(f"- **Aircraft Model OOD Detection**: VLM slightly outperforms ResNet with ROC AUC difference of {model_ood_diff:.3f}\n")
                elif model_ood_diff < -0.05:
                    f.write(f"- **Aircraft Model OOD Detection**: ResNet significantly outperforms VLM with ROC AUC difference of {-model_ood_diff:.3f}\n")
                elif model_ood_diff < 0:
                    f.write(f"- **Aircraft Model OOD Detection**: ResNet slightly outperforms VLM with ROC AUC difference of {-model_ood_diff:.3f}\n")
                else:
                    f.write("- **Aircraft Model OOD Detection**: ResNet and VLM perform similarly\n")
        
        # Add embedding space conclusion
        if distance_stats:
            f.write("\n### Embedding Space Analysis\n\n")
            
            if 'resnet-internal' in distance_stats and 'vlm-internal' in distance_stats:
                internal_diff = distance_stats['resnet-internal']['mean'] - distance_stats['vlm-internal']['mean']
                
                if internal_diff > 0.05:
                    f.write("- VLM creates more compact and coherent clusters than ResNet\n")
                elif internal_diff < -0.05:
                    f.write("- ResNet creates more compact and coherent clusters than VLM\n")
                else:
                    f.write("- Both models create similarly compact clusters\n")
            
            if 'cross-model' in distance_stats:
                cross_model_mean = distance_stats['cross-model']['mean']
                f.write(f"- The mean cosine distance between ResNet and VLM embeddings is {cross_model_mean:.4f}, indicating ")
                
                if cross_model_mean > 0.7:
                    f.write("significantly different feature representations\n")
                elif cross_model_mean > 0.5:
                    f.write("moderately different feature representations\n")
                else:
                    f.write("similar feature representations\n")
        
        # Final comparison summary
        f.write("\n## Key Differences Between ResNet and VLM\n\n")
        
        # Feature representation
        if distance_stats and 'cross-model' in distance_stats:
            f.write("1. **Feature Representation**: ")
            if distance_stats['cross-model']['mean'] > 0.5:
                f.write("The models learn substantially different feature representations, as evidenced by the high cross-model distances. ")
                f.write("ResNet relies on ImageNet pretraining which focuses on general visual features, while VLM (CLIP) is trained to align images with text descriptions, ")
                f.write("leading to more semantically meaningful representations.\n")
            else:
                f.write("Despite being trained differently, the models learn somewhat similar feature representations. ")
                f.write("Both capture essential visual features needed for aircraft classification.\n")
        
        # Zero-shot capabilities
        f.write("2. **Zero-Shot Capabilities**: VLM models like CLIP can directly compare images to text embeddings, enabling zero-shot classification without labeled training data. ")
        f.write("ResNet lacks this capability since it doesn't have text-image alignment.\n")
        
        # Open-set performance
        if 'ood_manufacturer' in metrics:
            f.write("3. **Open-Set Performance**: ")
            
            # Compare ROC AUC values
            best_resnet_ood = max(metrics['ood_manufacturer']['resnet'].items(), 
                                 key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
            best_vlm_ood = max(metrics['ood_manufacturer']['vlm'].items(), 
                              key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
            ood_diff = best_vlm_ood - best_resnet_ood
            
            if ood_diff > 0.05:
                f.write("VLM demonstrates superior performance in distinguishing between known and unknown aircraft, ")
                f.write("likely due to its broader semantic understanding from text-image pretraining.\n")
            elif ood_diff < -0.05:
                f.write("ResNet demonstrates superior performance in distinguishing between known and unknown aircraft, ")
                f.write("suggesting its visual features are more discriminative for this specific task.\n")
            else:
                f.write("Both models show comparable ability to distinguish between known and unknown aircraft.\n")
        else:
            f.write("3. **Open-Set Performance**: Full comparison requires running open-set evaluation.\n")
            
        # Semantic understanding
        f.write("4. **Semantic Understanding**: VLM embeddings are aligned with text, providing more interpretable features that bridge vision and language. ")
        f.write("This allows VLM to understand aircraft at both the visual and conceptual level, while ResNet relies solely on visual patterns.\n")
        
        # Overall recommendation
        f.write("\n## Overall Recommendation\n\n")
        
        overall_vlm_advantage = 0
        if 'classification_manufacturer' in metrics:
            best_resnet_manuf = max(metrics['classification_manufacturer']['resnet'].items(), 
                                    key=lambda x: x[1]['accuracy'])[1]['accuracy']
            best_vlm_manuf = max(metrics['classification_manufacturer']['vlm'].items(), 
                                key=lambda x: x[1]['accuracy'])[1]['accuracy']
            manuf_diff = best_vlm_manuf - best_resnet_manuf
            overall_vlm_advantage += 1 if manuf_diff > 0 else -1 if manuf_diff < 0 else 0
            
        if 'classification_model' in metrics:
            best_resnet_model = max(metrics['classification_model']['resnet'].items(), 
                                    key=lambda x: x[1]['accuracy'])[1]['accuracy']
            best_vlm_model = max(metrics['classification_model']['vlm'].items(), 
                                key=lambda x: x[1]['accuracy'])[1]['accuracy']
            model_diff = best_vlm_model - best_resnet_model
            overall_vlm_advantage += 1 if model_diff > 0 else -1 if model_diff < 0 else 0
            
        if 'ood_manufacturer' in metrics:
            best_resnet_ood = max(metrics['ood_manufacturer']['resnet'].items(), 
                                 key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
            best_vlm_ood = max(metrics['ood_manufacturer']['vlm'].items(), 
                              key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
            ood_diff = best_vlm_ood - best_resnet_ood
            overall_vlm_advantage += 1 if ood_diff > 0 else -1 if ood_diff < 0 else 0
            
        if 'ood_model' in metrics:
            best_resnet_model_ood = max(metrics['ood_model']['resnet'].items(), 
                                      key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
            best_vlm_model_ood = max(metrics['ood_model']['vlm'].items(), 
                                   key=lambda x: x[1]['roc_auc'])[1]['roc_auc']
            model_ood_diff = best_vlm_model_ood - best_resnet_model_ood
            overall_vlm_advantage += 1 if model_ood_diff > 0 else -1 if model_ood_diff < 0 else 0
        
        # Add zero-shot as a VLM advantage
        overall_vlm_advantage += 1
        
        if overall_vlm_advantage > 0:
            f.write("Based on the comprehensive comparison, **VLM (CLIP) is recommended** for aircraft classification tasks because:\n\n")
            f.write("1. It offers comparable or better accuracy for both manufacturer and model classification\n")
            f.write("2. It enables zero-shot classification without any labeled training data\n")
            f.write("3. It can handle open-set scenarios by effectively detecting unknown aircraft types\n")
            f.write("4. Its semantic understanding provides more interpretable features\n")
        elif overall_vlm_advantage < 0:
            f.write("Based on the comprehensive comparison, **ResNet is recommended** for aircraft classification tasks because:\n\n")
            f.write("1. It provides better accuracy for both manufacturer and model classification\n")
            f.write("2. It creates more discriminative features for this specific domain\n")
            f.write("3. It performs better at detecting unknown aircraft types\n")
            f.write("4. It may be more computationally efficient for deployment\n")
        else:
            f.write("Based on the comprehensive comparison, **both models have strengths** for aircraft classification tasks:\n\n")
            f.write("- **ResNet** provides strong performance with classic computer vision features\n")
            f.write("- **VLM (CLIP)** offers comparable performance with added zero-shot capabilities\n\n")
            f.write("The choice between them should depend on specific requirements such as need for zero-shot classification, computational constraints, and importance of semantic interpretability.\n")
    
    print(f"Comparison report generated: {report_path}")


def main():
    """Main function to run the ResNet vs VLM comparison."""
    args = parse_args()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Setup output directory
        output_dir = setup_output_dir(args)
        print(f"Results will be saved to: {output_dir}")
        
        # Load data
        print("Loading data...")
        
        # Initialize VLM protocol for data loading
        vlm_protocol = AircraftProtocol(
            clip_model_name=args.clip_model,
            data_dir=args.data_dir
        )
        
        # Load and prepare data using the protocol
        data = load_and_prepare_data(args, vlm_protocol)
        
        # Check if we have any data
        if len(data['all_df']) == 0:
            print("No data loaded. Check your data directory and CSV files.")
            sys.exit(1)
        
        # Find images directory
        images_dir = vlm_protocol.image_dir
        
        # Initialize ResNet for embedding extraction
        print(f"Initializing ResNet embedding extractor with {args.resnet_model}...")
        resnet_extractor = ResNetEmbeddingExtractor(model_name=args.resnet_model)
        
        # Extract ResNet embeddings
        print("Extracting ResNet embeddings...")
        resnet_embeddings = resnet_extractor.extract_embeddings(
            data['all_df'],
            images_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        
        # Extract VLM (CLIP) embeddings
        print("Extracting VLM embeddings...")
        # The embeddings are already generated during load_and_prepare_data
        vlm_embeddings = vlm_protocol.image_embeddings
        
        # Track metrics for summary
        all_metrics = {}
        
        # Parse k values
        k_values = [int(k) for k in args.k_values.split(",")]
        
        # Run KNN classification comparison
        if args.analyze_all or args.analyze_knn:
            # Determine which targets to analyze
            if args.classification_target == 'both':
                targets = ['manufacturer', 'Classes']
            else:
                targets = [args.classification_target]
            
            for target in targets:
                print(f"\nRunning KNN classification for {target}...")
                metrics = compare_knn_classification(
                    resnet_embeddings,
                    vlm_embeddings,
                    data,
                    k_values,
                    output_dir,
                    classification_target=target
                )
                
                all_metrics[f'classification_{target}'] = metrics
        
        # Run OOD detection comparison
        if args.analyze_all or args.analyze_open_set:
            # Determine which targets to analyze
            if args.classification_target == 'both':
                targets = ['manufacturer', 'Classes']
            else:
                targets = [args.classification_target]
            
            for target in targets:
                print(f"\nRunning OOD detection for {target}...")
                metrics = compare_ood_detection(
                    resnet_embeddings,
                    vlm_embeddings,
                    vlm_protocol,
                    data,
                    k_values,
                    output_dir,
                    classification_target=target
                )
                
                all_metrics[f'ood_{target}'] = metrics
        
        # Run embedding distribution analysis
        if args.analyze_all or args.analyze_embeddings:
            # Determine which targets to analyze
            if args.classification_target == 'both':
                targets = ['manufacturer', 'Classes']
            else:
                targets = [args.classification_target]
            
            # Set UMAP parameters
            umap_params = {
                'n_neighbors': args.umap_neighbors,
                'min_dist': args.umap_min_dist
            }
            
            # Compare embeddings for each target
            distance_stats = None
            for target in targets:
                stats = compare_embeddings_umap(
                    resnet_embeddings,
                    vlm_embeddings,
                    data['all_df'],
                    output_dir,
                    umap_params=umap_params,
                    color_by=target
                )
                
                # Keep stats from the first run for the report
                if distance_stats is None:
                    distance_stats = stats
        
        # Generate summary report
        create_summary_report(all_metrics, output_dir, distance_stats)
        
        print("\nComparison analysis complete!")
        print(f"See the detailed report at: {os.path.join(output_dir, 'resnet_vs_vlm_comparison.md')}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()