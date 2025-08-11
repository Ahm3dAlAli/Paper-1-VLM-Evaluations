#!/usr/bin/env python
"""
Hierarchical ResNet vs VLM Comparison for Aircraft Classification

This script compares ResNet (ImageNet) and VLM (CLIP) embeddings for hierarchical
aircraft classification, with a focus on:
1. Multi-level classification (Category → Subcategory → Model)
2. Cross-hierarchy performance analysis
3. Embedding space comparison across levels
4. Advanced OOD detection at each hierarchy level
5. Feature representation analysis

Usage:
    python hierarchical_resnet_vs_vlm.py --data_dir ./data/fgvc-aircraft --analyze_all
    python hierarchical_resnet_vs_vlm.py --analyze_hierarchy --analyze_embeddings
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
from aircraft_hierarchy_protocol import HierarchicalAircraftProtocol
from scipy.spatial.distance import cosine
from collections import Counter, defaultdict
import json
import pickle


class HierarchicalResNetExtractor:
    """Enhanced ResNet embedding extractor for hierarchical aircraft classification."""
    
    def __init__(self, model_name='resnet50', device=None, hierarchy_levels=None):
        """
        Initialize hierarchical ResNet embedding extractor.
        
        Args:
            model_name: Name of the ResNet model to use
            device: Device to run the model on ('cuda' or 'cpu')
            hierarchy_levels: Number of hierarchy levels to support
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hierarchy_levels = hierarchy_levels or 3
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
        
        # Store hierarchy information
        self.known_manufacturers = set(["Boeing", "Airbus"])
        
        print(f"Initialized Hierarchical ResNet extractor with {model_name}")
    
    def get_manufacturer(self, aircraft_class):
        """Map aircraft class to manufacturer (for compatibility with VLM protocol)."""
        if pd.isna(aircraft_class):
            return 'Unknown'
        class_str = str(aircraft_class).lower()
        if 'boeing' in class_str or class_str.startswith('7'):
            return 'Boeing'
        elif 'airbus' in class_str or class_str.startswith('a3'):
            return 'Airbus'
        else:
            return 'Unknown'
    
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
    parser = argparse.ArgumentParser(description="Hierarchical ResNet vs VLM Comparison")
    
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
    parser.add_argument("--analyze_hierarchy", action="store_true",
                        help="Run hierarchical classification analysis")
    parser.add_argument("--analyze_cross_level", action="store_true",
                        help="Run cross-level performance analysis")
    parser.add_argument("--analyze_embeddings", action="store_true",
                        help="Run embedding distribution analysis")
    parser.add_argument("--analyze_ood", action="store_true",
                        help="Run comprehensive OOD detection analysis")
    parser.add_argument("--analyze_features", action="store_true",
                        help="Run feature representation analysis")
    parser.add_argument("--analyze_all", action="store_true",
                        help="Run all analyses")
    
    # Hierarchy options
    parser.add_argument("--custom_hierarchy", type=str, default=None,
                        help="Path to custom hierarchy JSON file")
    
    # Analysis specifics
    parser.add_argument("--k_values", type=str, default="1,3,5,10",
                        help="Comma-separated list of k values for KNN")
    parser.add_argument("--ood_methods", type=str, default="knn,isolation_forest,energy,mahalanobis",
                        help="Comma-separated list of OOD methods to compare")
    
    # UMAP options
    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="hierarchical_resnet_vs_vlm_results",
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
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Fraction of data to use for training")
    
    # Save/load options
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save embeddings for future use")
    parser.add_argument("--load_embeddings", type=str, default=None,
                        help="Path to pre-computed embeddings file")
    
    args = parser.parse_args()
    
    # If no analysis is specified, run all
    if not any([args.analyze_hierarchy, args.analyze_cross_level, args.analyze_embeddings, 
                args.analyze_ood, args.analyze_features, args.analyze_all]):
        args.analyze_all = True
        print("No specific analysis requested. Running all analyses.")
    
    return args


def setup_output_dir(args):
    """Setup output directory for analysis results."""
    output_dir = args.output_dir
    
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "hierarchical_comparison"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cross_level_analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "embedding_analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ood_comparison"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "feature_analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    return output_dir


def load_custom_hierarchy(hierarchy_path):
    """Load custom hierarchy from JSON file."""
    try:
        with open(hierarchy_path, 'r') as f:
            hierarchy = json.load(f)
        print(f"Loaded custom hierarchy from {hierarchy_path}")
        return hierarchy
    except Exception as e:
        print(f"Error loading custom hierarchy: {e}")
        return None


def load_and_prepare_data(args, vlm_protocol, resnet_extractor):
    """
    Load and prepare data for hierarchical comparison analysis.
    
    Args:
        args: Command line arguments.
        vlm_protocol: HierarchicalAircraftProtocol instance.
        resnet_extractor: HierarchicalResNetExtractor instance.
        
    Returns:
        Dictionary with hierarchical data and embeddings.
    """
    print("\nLoading and preparing hierarchical data...")
    
    try:
        # Define CSV paths
        train_csv = os.path.join(args.data_dir, args.train_csv)
        val_csv = os.path.join(args.data_dir, args.val_csv)
        test_csv = os.path.join(args.data_dir, args.test_csv)
        
        # Load datasets
        train_df = vlm_protocol.load_data(train_csv)
        val_df = vlm_protocol.load_data(val_csv)
        test_df = vlm_protocol.load_data(test_csv)
        
        # Create hierarchical classifications
        print("Creating hierarchical classifications...")
        train_df_hierarchical = vlm_protocol.create_hierarchical_dataset(train_df)
        val_df_hierarchical = vlm_protocol.create_hierarchical_dataset(val_df)
        test_df_hierarchical = vlm_protocol.create_hierarchical_dataset(test_df)
        
        # Limit number of samples if specified
        if args.max_samples:
            train_df_hierarchical = train_df_hierarchical.sample(
                min(args.max_samples, len(train_df_hierarchical)), random_state=args.seed
            )
            val_df_hierarchical = val_df_hierarchical.sample(
                min(args.max_samples, len(val_df_hierarchical)), random_state=args.seed
            )
            test_df_hierarchical = test_df_hierarchical.sample(
                min(args.max_samples, len(test_df_hierarchical)), random_state=args.seed
            )
        
        print(f"Using {len(train_df_hierarchical)} training samples")
        print(f"Using {len(val_df_hierarchical)} validation samples")
        print(f"Using {len(test_df_hierarchical)} test samples")
        
        # Combine all data for embedding extraction
        all_df = pd.concat([train_df_hierarchical, val_df_hierarchical, test_df_hierarchical])
        
        # Generate embeddings or load them
        if args.load_embeddings and os.path.exists(args.load_embeddings):
            print(f"Loading pre-computed embeddings from {args.load_embeddings}")
            with open(args.load_embeddings, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            vlm_protocol.image_embeddings = embeddings_data['vlm_embeddings']
            vlm_protocol.hierarchy_embeddings = embeddings_data['hierarchy_embeddings']
            resnet_extractor.image_embeddings = embeddings_data['resnet_embeddings']
            
            print(f"Loaded {len(vlm_protocol.image_embeddings)} VLM embeddings")
            print(f"Loaded {len(resnet_extractor.image_embeddings)} ResNet embeddings")
        else:
            # Generate VLM embeddings
            print("Generating VLM hierarchical text embeddings...")
            vlm_protocol.generate_hierarchical_text_embeddings()
            
            print("Generating VLM image embeddings...")
            vlm_protocol.generate_image_embeddings(all_df, batch_size=args.batch_size)
            
            # Generate ResNet embeddings
            print("Generating ResNet image embeddings...")
            resnet_extractor.extract_embeddings(
                all_df, vlm_protocol.image_dir, batch_size=args.batch_size
            )
            
            # Save embeddings if requested
            if args.save_embeddings:
                embeddings_path = os.path.join(args.output_dir, "comparison_embeddings.pkl")
                embeddings_data = {
                    'vlm_embeddings': vlm_protocol.image_embeddings,
                    'hierarchy_embeddings': vlm_protocol.hierarchy_embeddings,
                    'resnet_embeddings': resnet_extractor.image_embeddings,
                    'hierarchy': vlm_protocol.hierarchy
                }
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(embeddings_data, f)
                print(f"Saved embeddings to {embeddings_path}")
        
        data = {
            'train_df': train_df_hierarchical,
            'val_df': val_df_hierarchical,
            'test_df': test_df_hierarchical,
            'all_df': all_df
        }
        
        return data
    except Exception as e:
        print(f"Error in load_and_prepare_data: {e}")
        # Return minimal valid data to avoid crashes
        empty_df = pd.DataFrame(columns=['filename', 'Classes', 'level_1_category', 'level_2_subcategory', 'level_3_model'])
        return {
            'train_df': empty_df,
            'val_df': empty_df,
            'test_df': empty_df,
            'all_df': empty_df
        }


def hierarchical_knn_classification(
    embeddings, 
    train_df, 
    test_df, 
    level=1,
    k=5
):
    """
    Perform hierarchical KNN classification at a specific level.
    
    Args:
        embeddings: Dictionary mapping image filenames to embeddings.
        train_df: Training DataFrame with hierarchical labels.
        test_df: Test DataFrame.
        level: Hierarchy level to classify (1, 2, or 3).
        k: Number of neighbors to use.
        
    Returns:
        DataFrame with classification results and metrics dict.
    """
    # Determine label column based on level
    if level == 1:
        label_column = "level_1_category"
    elif level == 2:
        label_column = "level_2_subcategory"
    else:
        label_column = "level_3_model"
    
    # Extract training embeddings and labels
    train_embeddings = []
    train_labels = []
    train_filenames = []
    
    for _, row in train_df.iterrows():
        filename = row['filename']
        if filename in embeddings and label_column in row:
            # Convert torch tensor to numpy if needed
            embedding = embeddings[filename].numpy() if isinstance(embeddings[filename], torch.Tensor) else embeddings[filename]
            train_embeddings.append(embedding)
            train_labels.append(row[label_column])
            train_filenames.append(filename)
    
    if len(train_embeddings) == 0:
        print(f"No training embeddings found for level {level}")
        return pd.DataFrame(), {}
    
    train_embeddings = np.array(train_embeddings)
    
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=min(k, len(train_embeddings)))
    knn.fit(train_embeddings)
    
    # Classify test samples
    results = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Level {level} KNN classification"):
        filename = row['filename']
        if filename not in embeddings:
            continue
            
        img_embedding = embeddings[filename]
        if isinstance(img_embedding, torch.Tensor):
            img_embedding = img_embedding.numpy()
        img_embedding = img_embedding.reshape(1, -1)
        
        # Find k nearest neighbors
        distances, indices = knn.kneighbors(img_embedding)
        
        # Get neighbor labels and vote
        neighbor_labels = [train_labels[i] for i in indices[0]]
        label_counts = Counter(neighbor_labels)
        predicted = label_counts.most_common(1)[0][0]
        confidence = label_counts[predicted] / k
        
        # Get ground truth
        true_label = row.get(label_column, "Unknown")
        is_correct = predicted == true_label
        
        results.append({
            'filename': filename,
            f'true_{label_column}': true_label,
            'predicted': predicted,
            'confidence': confidence,
            'is_correct': is_correct,
            'mean_distance': distances[0].mean(),
            'level': level
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    metrics = {}
    if len(results_df) > 0 and f'true_{label_column}' in results_df.columns:
        y_true = results_df[f'true_{label_column}'].values
        y_pred = results_df['predicted'].values
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        except:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0
        
        metrics['level'] = level
    
    return results_df, metrics


def compare_hierarchical_classification(
    resnet_extractor,
    vlm_protocol, 
    data, 
    k_values, 
    output_dir
):
    """
    Compare ResNet and VLM embeddings for hierarchical classification.
    
    Args:
        resnet_extractor: HierarchicalResNetExtractor instance.
        vlm_protocol: HierarchicalAircraftProtocol instance.
        data: Dictionary with hierarchical data.
        k_values: List of k values to test.
        output_dir: Output directory.
        
    Returns:
        Dictionary with comprehensive comparison results.
    """
    print("\n" + "="*60)
    print("HIERARCHICAL CLASSIFICATION COMPARISON")
    print("="*60)
    
    comparison_dir = os.path.join(output_dir, "hierarchical_comparison")
    
    # Prepare train/test split
    train_size = int(len(data['test_df']) * 0.7)
    train_df = data['test_df'].sample(n=train_size, random_state=42)
    test_df = data['test_df'].drop(train_df.index)
    
    comparison_results = {}
    
    # Test each hierarchy level
    for level in [1, 2, 3]:
        level_name = ["Category", "Subcategory", "Model"][level-1]
        print(f"\nAnalyzing Level {level} ({level_name}) classification...")
        
        level_results = {
            'resnet': {},
            'vlm': {},
            'level_name': level_name
        }
        
        # Test different k values
        for k in k_values:
            print(f"Testing k={k}...")
            
            # ResNet classification
            resnet_results, resnet_metrics = hierarchical_knn_classification(
                resnet_extractor.image_embeddings,
                train_df,
                test_df,
                level=level,
                k=k
            )
            
            # VLM classification
            vlm_results, vlm_metrics = hierarchical_knn_classification(
                vlm_protocol.image_embeddings,
                train_df,
                test_df,
                level=level,
                k=k
            )
            
            # Store results
            level_results['resnet'][f'k{k}'] = {
                'results': resnet_results,
                'metrics': resnet_metrics
            }
            level_results['vlm'][f'k{k}'] = {
                'results': vlm_results,
                'metrics': vlm_metrics
            }
            
            # Save detailed results
            level_dir = os.path.join(comparison_dir, f"level_{level}")
            os.makedirs(level_dir, exist_ok=True)
            
            resnet_results.to_csv(os.path.join(level_dir, f"resnet_k{k}_results.csv"), index=False)
            vlm_results.to_csv(os.path.join(level_dir, f"vlm_k{k}_results.csv"), index=False)
            
            # Print comparison
            print(f"  Level {level} (k={k}) Results:")
            print(f"    ResNet Accuracy: {resnet_metrics.get('accuracy', 0):.4f}")
            print(f"    VLM Accuracy: {vlm_metrics.get('accuracy', 0):.4f}")
            print(f"    Difference: {vlm_metrics.get('accuracy', 0) - resnet_metrics.get('accuracy', 0):+.4f}")
        
        comparison_results[f'level_{level}'] = level_results
        
        # Create level-specific comparison visualizations
        create_level_comparison_plots(level_results, k_values, level_dir, level_name)
    
    # Create overall comparison visualization
    create_overall_comparison_plots(comparison_results, k_values, comparison_dir)
    
    return comparison_results


def create_level_comparison_plots(level_results, k_values, output_dir, level_name):
    """Create comparison plots for a specific hierarchy level."""
    
    # Extract accuracy values
    resnet_accuracies = []
    vlm_accuracies = []
    
    for k in k_values:
        resnet_acc = level_results['resnet'][f'k{k}']['metrics'].get('accuracy', 0)
        vlm_acc = level_results['vlm'][f'k{k}']['metrics'].get('accuracy', 0)
        resnet_accuracies.append(resnet_acc)
        vlm_accuracies.append(vlm_acc)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Level Comparison: {level_name}', fontsize=16)
    
    # Accuracy comparison
    x = np.arange(len(k_values))
    width = 0.35
    
    axes[0].bar(x - width/2, resnet_accuracies, width, label='ResNet', alpha=0.7, color='blue')
    axes[0].bar(x + width/2, vlm_accuracies, width, label='VLM', alpha=0.7, color='green')
    
    axes[0].set_xlabel('k value')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{level_name} Classification Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(k_values)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Difference plot
    differences = [vlm - resnet for vlm, resnet in zip(vlm_accuracies, resnet_accuracies)]
    colors = ['green' if diff > 0 else 'red' for diff in differences]
    
    axes[1].bar(x, differences, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('k value')
    axes[1].set_ylabel('Accuracy Difference (VLM - ResNet)')
    axes[1].set_title(f'{level_name} Performance Difference')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(k_values)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{level_name.lower()}_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def create_overall_comparison_plots(comparison_results, k_values, output_dir):
    """Create overall comparison plots across all hierarchy levels."""
    
    # Extract data for all levels
    levels = [1, 2, 3]
    level_names = ["Category", "Subcategory", "Model"]
    
    # Best k performance for each level
    resnet_best = []
    vlm_best = []
    
    for level in levels:
        level_key = f'level_{level}'
        if level_key in comparison_results:
            level_data = comparison_results[level_key]
            
            # Find best k for each model
            resnet_accs = [level_data['resnet'][f'k{k}']['metrics'].get('accuracy', 0) for k in k_values]
            vlm_accs = [level_data['vlm'][f'k{k}']['metrics'].get('accuracy', 0) for k in k_values]
            
            resnet_best.append(max(resnet_accs))
            vlm_best.append(max(vlm_accs))
        else:
            resnet_best.append(0)
            vlm_best.append(0)
    
    # Create overall comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ResNet vs VLM: Hierarchical Classification Comparison', fontsize=16)
    
    # Overall accuracy comparison
    x = np.arange(len(levels))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, resnet_best, width, label='ResNet', alpha=0.7, color='blue')
    axes[0, 0].bar(x + width/2, vlm_best, width, label='VLM', alpha=0.7, color='green')
    axes[0, 0].set_xlabel('Hierarchy Level')
    axes[0, 0].set_ylabel('Best Accuracy')
    axes[0, 0].set_title('Best Performance Across Levels')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(level_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Performance degradation
    axes[0, 1].plot(levels, resnet_best, marker='o', linewidth=2, markersize=8, label='ResNet', color='blue')
    axes[0, 1].plot(levels, vlm_best, marker='s', linewidth=2, markersize=8, label='VLM', color='green')