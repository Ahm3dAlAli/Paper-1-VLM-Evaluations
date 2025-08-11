"""
ResNet vs VLM Comparison for Aircraft Classification - Updated Version

This script compares ResNet (ImageNet) and VLM (CLIP) embeddings
for Boeing vs. Airbus classification and OOD detection, focusing on the
differences in the image embeddings produced by each model.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.neighbors import NearestNeighbors
from umap import UMAP  # Correct import for UMAP
import seaborn as sns
from scipy.spatial.distance import cosine

# Import the existing protocol and enhanced OOD detection methods
from aircraft_protocol import AircraftProtocol
from enhanced_ood_detection import (
    knn_ood_detection_optimized,
    compare_text_to_image_similarity,
    visualize_ood_comparison
)


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
    
    # KNN options
    parser.add_argument("--k_values", type=str, default="1,3,5,10",
                        help="Comma-separated list of k values for KNN")
    
    # UMAP options
    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="resnet_vs_vlm_comparison",
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


def setup_output_dir(args):
    """Setup output directory for analysis results."""
    output_dir = args.output_dir
    
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def load_data(args):
    """
    Load and prepare data for analysis.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary with data.
    """
    print("\nLoading and preparing data...")
    
    try:
        # Define CSV paths
        train_csv = os.path.join(args.data_dir, args.train_csv)
        val_csv = os.path.join(args.data_dir, args.val_csv)
        test_csv = os.path.join(args.data_dir, args.test_csv)
        
        # Check if CSVs exist, if not, use the AircraftProtocol to create them
        from aircraft_protocol import AircraftProtocol
        temp_protocol = AircraftProtocol(data_dir=args.data_dir)
        
        # Load datasets
        train_df = temp_protocol.load_data(train_csv)
        val_df = temp_protocol.load_data(val_csv)
        test_df = temp_protocol.load_data(test_csv)
        
        # Prepare for Boeing vs. Airbus + Unknown classification
        known_manufacturers = ["Boeing", "Airbus"]
        
        # For training, use only Boeing and Airbus
        train_df_filtered = train_df[train_df['manufacturer'].isin(known_manufacturers)]
        val_df_filtered = val_df[val_df['manufacturer'].isin(known_manufacturers)]
        
        # For testing, include all manufacturers but label non-Boeing/Airbus as "Unknown"
        test_df_modified = test_df.copy()
        test_df_modified['original_manufacturer'] = test_df_modified['manufacturer']
        test_df_modified['manufacturer'] = test_df_modified['manufacturer'].apply(
            lambda x: x if x in known_manufacturers else "Unknown"
        )
        
        # Create a close-set test set (Boeing and Airbus only)
        test_df_close = test_df_modified[test_df_modified['manufacturer'].isin(known_manufacturers)]
        
        # Limit number of samples if specified
        if args.max_samples:
            train_df_filtered = train_df_filtered.sample(min(args.max_samples, len(train_df_filtered)), random_state=args.seed)
            val_df_filtered = val_df_filtered.sample(min(args.max_samples, len(val_df_filtered)), random_state=args.seed)
            test_df_modified = test_df_modified.sample(min(args.max_samples, len(test_df_modified)), random_state=args.seed)
            test_df_close = test_df_close.sample(min(args.max_samples, len(test_df_close)), random_state=args.seed)
        
        print(f"Using {len(train_df_filtered)} training samples (Boeing/Airbus only)")
        print(f"Using {len(val_df_filtered)} validation samples (Boeing/Airbus only)")
        print(f"Using {len(test_df_modified)} test samples (Boeing/Airbus + Unknown)")
        print(f"Using {len(test_df_close)} close-set test samples (Boeing/Airbus only)")
        
        # Combine all data for visualization and analysis
        all_df = pd.concat([train_df_filtered, val_df_filtered, test_df_modified])
        
        data = {
            'train_df': train_df_filtered,
            'val_df': val_df_filtered,
            'test_df': test_df_modified,
            'test_df_close': test_df_close,
            'all_df': all_df,
            'full_test_df': test_df  # Keep the original full test set
        }
        
        return data
    except Exception as e:
        print(f"Error in load_data: {e}")
        # Return minimal valid data to avoid crashes
        empty_df = pd.DataFrame(columns=['filename', 'manufacturer'])
        return {
            'train_df': empty_df,
            'val_df': empty_df,
            'test_df': empty_df,
            'test_df_close': empty_df,
            'all_df': empty_df,
            'full_test_df': empty_df
        }


def knn_binary_classification(embeddings, train_df, test_df, k=5):
    """
    Perform binary classification (Boeing vs. Airbus) using k-nearest neighbors.
    
    Args:
        embeddings: Dictionary mapping image filenames to embeddings.
        train_df: Training DataFrame (Boeing and Airbus only).
        test_df: Test DataFrame.
        k: Number of neighbors to use.
        
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
            train_embeddings.append(embeddings[filename].numpy() if isinstance(embeddings[filename], torch.Tensor) else embeddings[filename])
            train_labels.append(row['manufacturer'])
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
        img_embedding = embeddings[filename].numpy() if isinstance(embeddings[filename], torch.Tensor) else embeddings[filename]
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
        
        # Store result
        results.append({
            'filename': filename,
            'true_manufacturer': row['manufacturer'],
            'predicted': predicted,
            'confidence': confidence,
            'known_class': row['manufacturer'] in ["Boeing", "Airbus"],
            'is_correct': predicted == row['manufacturer'],
            'neighbor_distances': distances[0].tolist(),
            'neighbor_labels': neighbor_labels
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Only calculate metrics on known classes (Boeing and Airbus)
    known_results = results_df[results_df['known_class']]
    
    # Calculate metrics
    metrics = {}
    
    if len(known_results) > 0:
        y_true = known_results['true_manufacturer'].values
        y_pred = known_results['predicted'].values
        
        # Calculate metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
    else:
        metrics['accuracy'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
    
    return results_df, metrics


def compare_embeddings_umap(resnet_embeddings, vlm_embeddings, df, umap_params=None, output_dir=None):
    """
    Compare ResNet and VLM embeddings using UMAP visualization.
    """
    print("\nComparing embeddings with UMAP visualization...")
    
    if output_dir:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    else:
        vis_dir = "."
    
    # Get common filenames between both embedding sets
    common_filenames = set(resnet_embeddings.keys()) & set(vlm_embeddings.keys())
    common_df = df[df['filename'].isin(common_filenames)]
    
    # Extract embeddings and labels
    resnet_data = []
    vlm_data = []
    manufacturers = []
    
    for _, row in common_df.iterrows():
        filename = row['filename']
        if filename in resnet_embeddings and filename in vlm_embeddings:
            resnet_emb = resnet_embeddings[filename].numpy() if isinstance(resnet_embeddings[filename], torch.Tensor) else resnet_embeddings[filename]
            vlm_emb = vlm_embeddings[filename].numpy() if isinstance(vlm_embeddings[filename], torch.Tensor) else vlm_embeddings[filename]
            
            resnet_data.append(resnet_emb)
            vlm_data.append(vlm_emb)
            manufacturers.append(row['manufacturer'])
    
    resnet_data = np.array(resnet_data)
    vlm_data = np.array(vlm_data)
    manufacturers = np.array(manufacturers)
    
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
    
    # Create subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot ResNet embeddings
    for manufacturer in set(manufacturers):
        mask = manufacturers == manufacturer
        axes[0].scatter(
            resnet_2d[mask, 0],
            resnet_2d[mask, 1],
            label=manufacturer,
            alpha=0.7,
            edgecolors='none'
        )
    
    axes[0].set_title('ResNet Embeddings (UMAP)')
    axes[0].set_xlabel('UMAP Dimension 1')
    axes[0].set_ylabel('UMAP Dimension 2')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot VLM embeddings
    for manufacturer in set(manufacturers):
        mask = manufacturers == manufacturer
        axes[1].scatter(
            vlm_2d[mask, 0],
            vlm_2d[mask, 1],
            label=manufacturer,
            alpha=0.7,
            edgecolors='none'
        )
    
    axes[1].set_title('VLM Embeddings (UMAP)')
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "resnet_vs_vlm_umap.png"))
    plt.close()
    
    # Analyze embedding distances within each space (not across spaces)
    distances = {
        'resnet-internal': [],  # Distances between ResNet embeddings
        'vlm-internal': [],     # Distances between VLM embeddings
    }
    
    # Sample to avoid too many computations
    max_pairs = 1000
    random_indices = np.random.choice(len(resnet_data), min(max_pairs, len(resnet_data)), replace=False)
    
    for i in random_indices:
        # Sample a few other images to calculate internal distances
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
    
    plt.title('Distribution of Cosine Distances Within Embedding Spaces')
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
    
    return resnet_2d, vlm_2d, manufacturers


def analyze_resnet_vs_vlm(args, output_dir):
    """
    Analyze and compare ResNet and VLM embeddings for aircraft classification.
    
    Args:
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        None
    """
    # Load data
    data = load_data(args)
    
    # Find images directory
    images_dir = None
    for root, dirs, _ in os.walk(args.data_dir):
        if "images" in dirs:
            images_dir = os.path.join(root, "images")
            print(f"Found images directory at: {images_dir}")
            break
    
    if not images_dir:
        print("Could not find images directory.")
        return
    
    # Initialize models
    print("\nInitializing models...")
    
    # Initialize ResNet
    resnet_extractor = ResNetEmbeddingExtractor(model_name=args.resnet_model)
    
    # Initialize VLM (CLIP)
    vlm_protocol = AircraftProtocol(
        clip_model_name=args.clip_model,
        data_dir=args.data_dir
    )
    
    # Generate text embeddings (first step in VLM pipeline)
    print("\nGenerating text embeddings...")
    vlm_protocol.generate_text_embeddings()
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    
    # Extract ResNet embeddings
    resnet_embeddings = resnet_extractor.extract_embeddings(
        data['all_df'],
        images_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Extract VLM embeddings
    vlm_protocol.generate_image_embeddings(
        data['all_df'],
        batch_size=args.batch_size
    )
    vlm_embeddings = vlm_protocol.image_embeddings
    
    # Compare embeddings with UMAP
    umap_params = {
        'n_neighbors': args.umap_neighbors,
        'min_dist': args.umap_min_dist
    }
    
    resnet_2d, vlm_2d, manufacturers = compare_embeddings_umap(
        resnet_embeddings,
        vlm_embeddings,
        data['all_df'],
        umap_params=umap_params,
        output_dir=output_dir
    )
    
    # Run binary classification comparison if requested
    if args.analyze_all or args.analyze_knn:
        print("\nComparing binary classification performance...")
        
        # Create output directories
        classification_dir = os.path.join(output_dir, "classification")
        os.makedirs(classification_dir, exist_ok=True)
        
        # Parse k values
        k_values = [int(k) for k in args.k_values.split(",")]
        
        # Store results for comparison
        resnet_metrics = {}
        vlm_metrics = {}
        
        for k in k_values:
            print(f"\nRunning KNN classification with k={k}...")
            
            # ResNet classification
            print("ResNet classification...")
            resnet_results, resnet_k_metrics = knn_binary_classification(
                resnet_embeddings,
                data['train_df'],
                data['test_df_close'],  # Use close-set test data for binary classification
                k=k
            )
            
            # Save ResNet results
            resnet_results.to_csv(
                os.path.join(classification_dir, f"resnet_knn_k{k}_results.csv"),
                index=False
            )
            
            # VLM classification
            print("VLM classification...")
            vlm_results, vlm_k_metrics = knn_binary_classification(
                vlm_embeddings,
                data['train_df'],
                data['test_df_close'],  # Use close-set test data for binary classification
                k=k
            )
            
            # Save VLM results
            vlm_results.to_csv(
                os.path.join(classification_dir, f"vlm_knn_k{k}_results.csv"),
                index=False
            )
            
            # Store metrics
            resnet_metrics[f"k{k}"] = resnet_k_metrics
            vlm_metrics[f"k{k}"] = vlm_k_metrics
            
            # Print comparison
            print(f"\nBinary Classification Comparison (k={k}):")
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
        plt.title('ResNet vs VLM Binary Classification Accuracy')
        plt.xticks(x, k_values)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(classification_dir, "resnet_vs_vlm_accuracy.png"))
        plt.close()
        
        # Create F1 score comparison
        plt.figure(figsize=(12, 8))
        
        resnet_f1 = [resnet_metrics[f"k{k}"]["f1"] for k in k_values]
        vlm_f1 = [vlm_metrics[f"k{k}"]["f1"] for k in k_values]
        
        plt.bar(x - width/2, resnet_f1, width, label='ResNet')
        plt.bar(x + width/2, vlm_f1, width, label='VLM')
        
        plt.xlabel('k value')
        plt.ylabel('F1 Score')
        plt.title('ResNet vs VLM Binary Classification F1 Score')
        plt.xticks(x, k_values)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(classification_dir, "resnet_vs_vlm_f1.png"))
        plt.close()
    
    # Run open-set evaluation comparison if requested
    if args.analyze_all or args.analyze_open_set:
        print("\nComparing open-set evaluation performance...")
        
        # Create output directories
        openset_dir = os.path.join(output_dir, "open_set")
        os.makedirs(openset_dir, exist_ok=True)
        
        # Parse k values
        k_values = [int(k) for k in args.k_values.split(",")]
        
        # Store results for comparison
        resnet_ood_metrics = {}
        vlm_ood_metrics = {}
        
        for k in k_values:
            print(f"\nRunning OOD detection with k={k}...")
            
            # ResNet OOD detection
            print("ResNet OOD detection...")
            # Create ResNet protocol to use optimized OOD detection
            resnet_protocol = AircraftProtocol(data_dir=args.data_dir)
            resnet_protocol.image_embeddings = resnet_embeddings
            
            resnet_results, resnet_ood_k_metrics = knn_ood_detection_optimized(
                resnet_protocol,
                data['test_df'],  # Full test set including Unknown
                k=k,
                optimization_metric='balanced_accuracy'
            )
            
            # Save ResNet results
            resnet_results.to_csv(
                os.path.join(openset_dir, f"resnet_ood_k{k}_results.csv"),
                index=False
            )
            
            # VLM OOD detection
            print("VLM OOD detection...")
            vlm_results, vlm_ood_k_metrics = knn_ood_detection_optimized(
                vlm_protocol,
                data['test_df'],  # Full test set including Unknown
                k=k,
                optimization_metric='balanced_accuracy'
            )
            
            # Save VLM results
            vlm_results.to_csv(
                os.path.join(openset_dir, f"vlm_ood_k{k}_results.csv"),
                index=False
            )
            
            # Store metrics
            resnet_ood_metrics[f"ResNet k{k}"] = resnet_ood_k_metrics
            vlm_ood_metrics[f"VLM k{k}"] = vlm_ood_k_metrics
            
            # Print comparison
            print(f"\nOOD Detection Comparison (k={k}):")
            print(f"ResNet ROC AUC: {resnet_ood_k_metrics['roc_auc']:.4f}")
            print(f"VLM ROC AUC: {vlm_ood_k_metrics['roc_auc']:.4f}")
            print(f"ResNet Norm. Accuracy: {resnet_ood_k_metrics['normalized_accuracy']:.4f}")
            print(f"VLM Norm. Accuracy: {vlm_ood_k_metrics['normalized_accuracy']:.4f}")
        
        # Combine metrics for visualization
        combined_metrics = {**resnet_ood_metrics, **vlm_ood_metrics}
        visualize_ood_comparison(combined_metrics, openset_dir)
        
        # Compare text-image similarity for VLM (unique capability of VLM)
        print("\nEvaluating VLM text-image similarity for OOD detection...")
        text_image_results, text_image_metrics = compare_text_to_image_similarity(
            vlm_protocol,
            data['test_df'],
            output_dir=openset_dir
        )
        
        # Save text-image similarity results
        text_image_results.to_csv(
            os.path.join(openset_dir, "text_image_similarity_results.csv"),
            index=False
        )
        
        # Print text-image similarity results
        print("\nText-Image Similarity OOD Detection:")
        print(f"ROC AUC: {text_image_metrics.get('roc_auc', 0):.4f}")
        print(f"Normalized Accuracy: {text_image_metrics.get('normalized_accuracy', 0):.4f}")
    
    # Generate final comparison report
    print("\nGenerating comparison report...")
    
    report_path = os.path.join(output_dir, "resnet_vs_vlm_comparison.md")
    
    with open(report_path, 'w') as f:
        f.write("# ResNet vs VLM Comparison for Aircraft Classification\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model information
        f.write("## Model Information\n\n")
        f.write(f"- ResNet Model: {args.resnet_model} (ImageNet pretrained)\n")
        f.write(f"- VLM Model: {args.clip_model}\n\n")
        
        # Embedding space comparison
        f.write("## Embedding Space Comparison\n\n")
        
        f.write("### UMAP Visualization\n\n")
        f.write("![ResNet vs VLM UMAP](visualizations/resnet_vs_vlm_umap.png)\n\n")
        
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
        
        # Binary classification comparison
        if args.analyze_all or args.analyze_knn:
            f.write("## Binary Classification Comparison\n\n")
            f.write("### Accuracy Comparison\n\n")
            f.write("![Accuracy Comparison](classification/resnet_vs_vlm_accuracy.png)\n\n")
            
            f.write("### F1 Score Comparison\n\n")
            f.write("![F1 Score Comparison](classification/resnet_vs_vlm_f1.png)\n\n")
            
            # Add table with best results
            f.write("### Best Classification Results\n\n")
            f.write("| Model | Best k | Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|--------|----------|-----------|--------|----------|\n")
            
            # Find best ResNet results
            best_resnet_k = max(resnet_metrics.items(), key=lambda x: x[1]['accuracy'])
            best_resnet_metrics = best_resnet_k[1]
            
            # Find best VLM results
            best_vlm_k = max(vlm_metrics.items(), key=lambda x: x[1]['accuracy'])
            best_vlm_metrics = best_vlm_k[1]
            
            f.write(f"| ResNet | {best_resnet_k[0][1:]} | {best_resnet_metrics['accuracy']:.4f} | ")
            f.write(f"{best_resnet_metrics['precision']:.4f} | {best_resnet_metrics['recall']:.4f} | ")
            f.write(f"{best_resnet_metrics['f1']:.4f} |\n")
            
            f.write(f"| VLM | {best_vlm_k[0][1:]} | {best_vlm_metrics['accuracy']:.4f} | ")
            f.write(f"{best_vlm_metrics['precision']:.4f} | {best_vlm_metrics['recall']:.4f} | ")
            f.write(f"{best_vlm_metrics['f1']:.4f} |\n\n")
        
        # Open-set evaluation comparison
        if args.analyze_all or args.analyze_open_set:
            f.write("## Open-Set Evaluation Comparison\n\n")
            f.write("### ROC Curve Comparison\n\n")
            f.write("![ROC Curve Comparison](open_set/ood_roc_comparison.png)\n\n")
            
            f.write("### Performance Metrics Comparison\n\n")
            f.write("![Metrics Comparison](open_set/ood_metrics_comparison.png)\n\n")
            
            # Add table with best results
            f.write("### Best OOD Detection Results\n\n")
            f.write("| Model | Best k | ROC AUC | Normalized Accuracy | Precision | Recall | F1 Score |\n")
            f.write("|-------|--------|---------|---------------------|-----------|--------|----------|\n")
            
            # Find best ResNet results
            best_resnet_ood_k = max(resnet_ood_metrics.items(), key=lambda x: x[1]['normalized_accuracy'])
            best_resnet_ood_metrics = best_resnet_ood_k[1]
            
            # Find best VLM results
            best_vlm_ood_k = max(vlm_ood_metrics.items(), key=lambda x: x[1]['normalized_accuracy'])
            best_vlm_ood_metrics = best_vlm_ood_k[1]
            
            f.write(f"| ResNet | {best_resnet_ood_k[0].split('k')[1]} | {best_resnet_ood_metrics['roc_auc']:.4f} | ")
            f.write(f"{best_resnet_ood_metrics['normalized_accuracy']:.4f} | {best_resnet_ood_metrics['precision']:.4f} | ")
            f.write(f"{best_resnet_ood_metrics['recall']:.4f} | {best_resnet_ood_metrics['f1']:.4f} |\n")
            
            f.write(f"| VLM | {best_vlm_ood_k[0].split('k')[1]} | {best_vlm_ood_metrics['roc_auc']:.4f} | ")
            f.write(f"{best_vlm_ood_metrics['normalized_accuracy']:.4f} | {best_vlm_ood_metrics['precision']:.4f} | ")
            f.write(f"{best_vlm_ood_metrics['recall']:.4f} | {best_vlm_ood_metrics['f1']:.4f} |\n\n")
            
            # Add text-image similarity results (VLM only)
            if 'roc_auc' in text_image_metrics:
                f.write("### Text-Image Similarity for OOD Detection (VLM Only)\n\n")
                f.write("VLM models have the unique capability to compare image embeddings directly with text embeddings.\n\n")
                
                f.write("![Text-Image Similarity](open_set/boeing_airbus_similarity.png)\n\n")
                f.write("![Text-Image OOD ROC](open_set/text_image_ood_roc.png)\n\n")
                
                f.write("| Method | ROC AUC | Normalized Accuracy | Precision | Recall | F1 Score |\n")
                f.write("|--------|---------|---------------------|-----------|--------|----------|\n")
                f.write(f"| Text-Image | {text_image_metrics.get('roc_auc', 0):.4f} | ")
                f.write(f"{text_image_metrics.get('normalized_accuracy', 0):.4f} | ")
                f.write(f"{text_image_metrics.get('precision', 0):.4f} | ")
                f.write(f"{text_image_metrics.get('recall', 0):.4f} | ")
                f.write(f"{text_image_metrics.get('f1', 0):.4f} |\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Add model comparison conclusion
        if args.analyze_all or args.analyze_knn:
            acc_diff = best_vlm_metrics['accuracy'] - best_resnet_metrics['accuracy']
            
            if acc_diff > 0.05:
                f.write(f"- **Binary Classification**: VLM significantly outperforms ResNet by {acc_diff*100:.1f}% accuracy\n")
            elif acc_diff > 0:
                f.write(f"- **Binary Classification**: VLM slightly outperforms ResNet by {acc_diff*100:.1f}% accuracy\n")
            elif acc_diff < -0.05:
                f.write(f"- **Binary Classification**: ResNet significantly outperforms VLM by {-acc_diff*100:.1f}% accuracy\n")
            elif acc_diff < 0:
                f.write(f"- **Binary Classification**: ResNet slightly outperforms VLM by {-acc_diff*100:.1f}% accuracy\n")
            else:
                f.write("- **Binary Classification**: ResNet and VLM perform similarly\n")
        
        # Add OOD detection conclusion
        if args.analyze_all or args.analyze_open_set:
            ood_diff = best_vlm_ood_metrics['normalized_accuracy'] - best_resnet_ood_metrics['normalized_accuracy']
            
            if ood_diff > 0.05:
                f.write(f"- **Open-Set Evaluation**: VLM significantly outperforms ResNet by {ood_diff*100:.1f}% normalized accuracy\n")
            elif ood_diff > 0:
                f.write(f"- **Open-Set Evaluation**: VLM slightly outperforms ResNet by {ood_diff*100:.1f}% normalized accuracy\n")
            elif ood_diff < -0.05:
                f.write(f"- **Open-Set Evaluation**: ResNet significantly outperforms VLM by {-ood_diff*100:.1f}% normalized accuracy\n")
            elif ood_diff < 0:
                f.write(f"- **Open-Set Evaluation**: ResNet slightly outperforms VLM by {-ood_diff*100:.1f}% normalized accuracy\n")
            else:
                f.write("- **Open-Set Evaluation**: ResNet and VLM perform similarly\n")
        
        # Add embedding space conclusion
        if os.path.exists(os.path.join(output_dir, "visualizations", "embedding_distance_statistics.csv")):
            stats_df = pd.read_csv(os.path.join(output_dir, "visualizations", "embedding_distance_statistics.csv"), index_col=0)
            
            if 'resnet-internal' in stats_df.index and 'vlm-internal' in stats_df.index:
                internal_diff = stats_df.loc['resnet-internal', 'mean'] - stats_df.loc['vlm-internal', 'mean']
                
                if internal_diff > 0.05:
                    f.write("- **Embedding Space**: VLM creates more compact clusters than ResNet\n")
                elif internal_diff < -0.05:
                    f.write("- **Embedding Space**: ResNet creates more compact clusters than VLM\n")
                else:
                    f.write("- **Embedding Space**: Both models create similarly compact clusters\n")
            
            if 'cross-model' in stats_df.index:
                cross_model_mean = stats_df.loc['cross-model', 'mean']
                f.write(f"- **Cross-Model Comparison**: The mean cosine distance between ResNet and VLM embeddings is {cross_model_mean:.4f}, indicating ")
                
                if cross_model_mean > 0.7:
                    f.write("significantly different feature representations\n")
                elif cross_model_mean > 0.5:
                    f.write("moderately different feature representations\n")
                else:
                    f.write("similar feature representations\n")
        
        # Final comparison summary
        f.write("\n### Key Differences Between ResNet and VLM\n\n")
        f.write("1. **Feature Representation**: ")
        if 'cross-model' in stats_df.index and stats_df.loc['cross-model', 'mean'] > 0.5:
            f.write("The models learn substantially different feature representations, as evidenced by the high cross-model distances\n")
        else:
            f.write("Despite being trained differently, the models learn somewhat similar feature representations\n")
        
        f.write("2. **Zero-Shot Capabilities**: VLM can directly compare images to text embeddings, enabling zero-shot classification without labeled training data\n")
        
        f.write("3. **Open-Set Performance**: ")
        if args.analyze_all or args.analyze_open_set:
            if ood_diff > 0.05:
                f.write("VLM demonstrates superior performance in distinguishing between known and unknown manufacturers\n")
            elif ood_diff < -0.05:
                f.write("ResNet demonstrates superior performance in distinguishing between known and unknown manufacturers\n")
            else:
                f.write("Both models show comparable ability to distinguish between known and unknown manufacturers\n")
        else:
            f.write("Full comparison requires running open-set evaluation\n")
            
        f.write("4. **Semantic Understanding**: VLM embeddings are aligned with text, providing more interpretable features that bridge vision and language\n")
    
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
        
        # Run comparison analysis
        analyze_resnet_vs_vlm(args, output_dir)
        
        print("\nComparison analysis complete!")
        
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()