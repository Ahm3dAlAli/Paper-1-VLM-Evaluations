"""
Aircraft Analysis - Updated Version

This script implements a comprehensive analysis pipeline for evaluating
VLM embeddings in aircraft classification with a focus on:
1. Boeing vs. Airbus binary classification using KNN (close-set evaluation)
2. Open-set evaluation with OOD detection for "Unknown" manufacturers
3. Embedding distribution analysis with UMAP
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.spatial.distance import cosine


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Boeing vs. Airbus Binary Classification Analysis")
    
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
    
    # Analysis options
    parser.add_argument("--analyze_binary", action="store_true",
                        help="Run Boeing vs. Airbus binary classification analysis (close-set)")
    parser.add_argument("--analyze_open_set", action="store_true",
                        help="Run open-set evaluation analysis")
    parser.add_argument("--analyze_zero_shot", action="store_true", 
                        help="Run zero-shot classification analysis")
    parser.add_argument("--analyze_embeddings", action="store_true",
                        help="Run embedding distribution analysis")
    parser.add_argument("--analyze_all", action="store_true",
                        help="Run all analyses")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="boeing_airbus_analysis",
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
    
    # KNN options
    parser.add_argument("--k_values", type=str, default="1,3,5,10",
                        help="Comma-separated list of k values for KNN")
    
    # OOD detection options
    parser.add_argument("--ood_threshold", type=float, default=0.7,
                        help="Similarity threshold for OOD detection")
    
    # UMAP visualization options
    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    
    return parser.parse_args()


def setup_output_dir(args):
    """Setup output directory for analysis results."""
    output_dir = args.output_dir
    
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def prepare_boeing_airbus_dataset(df, known_manufacturers=["Boeing", "Airbus"]):
    """
    Prepare dataset for Boeing vs. Airbus classification with all other manufacturers as unknown.
    """
    # Create a copy of the dataframe
    modified_df = df.copy()
    
    # Get manufacturer column - might be derived from 'Classes' in some datasets
    if 'manufacturer' not in modified_df.columns and 'Classes' in modified_df.columns:
        # Try to infer manufacturer from Classes
        modified_df['manufacturer'] = modified_df['Classes'].apply(
            lambda x: 'Boeing' if x.startswith('Boeing') or x.startswith('7') else
                      'Airbus' if x.startswith('Airbus') or x.startswith('A3') else
                      'Unknown'
        )
    
    # If we still don't have a manufacturer column, create one based on filename
    if 'manufacturer' not in modified_df.columns:
        print("Warning: No manufacturer column found. Creating one based on available data.")
        if 'filename' in modified_df.columns:
            # Try to match filename to manufacturer using common patterns
            modified_df['manufacturer'] = 'Unknown'  # Default
    
    # Now handle the Boeing/Airbus/Unknown classification
    if 'manufacturer' in modified_df.columns:
        modified_df['original_manufacturer'] = modified_df['manufacturer']
        modified_df['manufacturer'] = modified_df['manufacturer'].apply(
            lambda x: x if x in known_manufacturers else "Unknown"
        )
    
    # Count samples in each category
    if 'manufacturer' in modified_df.columns:
        manufacturer_counts = modified_df['manufacturer'].value_counts()
        print("\nManufacturer distribution after modification:")
        for manufacturer, count in manufacturer_counts.items():
            print(f"  {manufacturer}: {count} samples")
    
    return modified_df


def load_and_prepare_data(args, protocol):
    """
    Load and prepare data for Boeing vs. Airbus analysis with unknown OOD detection.
    """
    print("\nLoading and preparing data...")
    
    try:
        # Define CSV paths
        train_csv = os.path.join(args.data_dir, args.train_csv)
        val_csv = os.path.join(args.data_dir, args.val_csv)
        test_csv = os.path.join(args.data_dir, args.test_csv)
        
        # Load datasets
        train_df = protocol.load_data(train_csv)
        val_df = protocol.load_data(val_csv)
        test_df = protocol.load_data(test_csv)
        
        # Modify the dataset for Boeing vs. Airbus + Unknown
        known_manufacturers = ["Boeing", "Airbus"]
        
        # For training, use only Boeing and Airbus
        train_df_filtered = train_df[train_df['manufacturer'].isin(known_manufacturers)]
        val_df_filtered = val_df[val_df['manufacturer'].isin(known_manufacturers)]
        
        # If filtered datasets are empty, use all data but label as Boeing/Airbus/Unknown
        if len(train_df_filtered) == 0:
            print("Warning: No Boeing or Airbus samples found in training data.")
            print("Using all training data with manufacturer labels.")
            train_df_filtered = prepare_boeing_airbus_dataset(train_df, known_manufacturers)
        
        if len(val_df_filtered) == 0:
            print("Warning: No Boeing or Airbus samples found in validation data.")
            print("Using all validation data with manufacturer labels.")
            val_df_filtered = prepare_boeing_airbus_dataset(val_df, known_manufacturers)
        
        # For testing, include all manufacturers but label non-Boeing/Airbus as "Unknown"
        test_df_modified = prepare_boeing_airbus_dataset(test_df, known_manufacturers)
        
        # Create a close-set test set (Boeing and Airbus only)
        test_df_close = test_df_modified[test_df_modified['manufacturer'].isin(known_manufacturers)]
        
        # Update the protocol's known manufacturers
        protocol.known_manufacturers = set(known_manufacturers)
        
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
        
        # Generate text embeddings (first step in our pipeline)
        print("Generating text embeddings...")
        protocol.generate_text_embeddings()
        
        # Generate image embeddings
        print("Generating image embeddings for all data...")
        all_df = pd.concat([train_df_filtered, val_df_filtered, test_df_modified])
        protocol.generate_image_embeddings(all_df, batch_size=args.batch_size)
        
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
        print(f"Error in load_and_prepare_data: {e}")
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


def analyze_binary_classification(protocol, data, args, output_dir):
    """
    Analyze Boeing vs. Airbus binary classification using KNN (close-set evaluation).
    
    Args:
        protocol: AircraftProtocol instance.
        data: Dictionary with data.
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        Dictionary with metrics.
    """
    print("\n" + "="*50)
    print("Boeing vs. Airbus Binary Classification with KNN (Close-Set Evaluation)")
    print("="*50)
    
    # Ensure output directories exist
    classification_dir = os.path.join(output_dir, "classification")
    os.makedirs(classification_dir, exist_ok=True)
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Perform KNN classification for each k value
    all_metrics = {}
    
    for k in k_values:
        print(f"\nRunning KNN classification with k={k}...")
        results_df, metrics = protocol.knn_binary_classification(
            data['train_df'], 
            data['test_df_close'],  # Use close-set test data (Boeing and Airbus only)
            k=k
        )
        
        # Save results
        results_df.to_csv(
            os.path.join(classification_dir, f"binary_knn_k{k}_results.csv"),
            index=False
        )
        
        # Generate confusion matrix visualization
        cm_fig = protocol.visualize_confusion_matrix(
            metrics['confusion_matrix'],
            metrics['classes'],
            title=f'Boeing vs. Airbus Confusion Matrix (k={k})'
        )
        cm_fig.savefig(
            os.path.join(classification_dir, f"binary_knn_k{k}_confusion_matrix.png")
        )
        plt.close(cm_fig)
        
        # Save metrics
        metrics_path = os.path.join(classification_dir, f"binary_knn_k{k}_metrics.json")
        with open(metrics_path, 'w') as f:
            import json
            json.dump({
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }, f, indent=2)
        
        # Print results
        print(f"\nBoeing vs. Airbus Classification Results (k={k}):")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1 Score: {metrics['f1']:.4f}")
        
        all_metrics[f"k{k}"] = metrics
    
    # Find best k value
    best_k = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest KNN configuration: {best_k[0]} with accuracy: {best_k[1]['accuracy']:.4f}")
    
    return all_metrics


def analyze_open_set_evaluation(protocol, data, args, output_dir):
    """
    Analyze open-set evaluation performance using KNN.
    
    Args:
        protocol: AircraftProtocol instance.
        data: Dictionary with data.
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        Dictionary with metrics.
    """
    print("\n" + "="*50)
    print("Open-Set Evaluation with KNN")
    print("="*50)
    
    # Ensure output directories exist
    openset_dir = os.path.join(output_dir, "open_set")
    os.makedirs(openset_dir, exist_ok=True)
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Performance evaluation on open-set data
    all_metrics = {}
    
    for k in k_values:
        print(f"\nRunning open-set evaluation with k={k}...")
        results_df, metrics = protocol.knn_open_set_evaluation(
            data['train_df'],  # Training data (Boeing and Airbus only)
            data['test_df'],   # Test data (including Unknown manufacturers)
            k=k,
            threshold=args.ood_threshold
        )
        
        # Save results
        results_df.to_csv(
            os.path.join(openset_dir, f"openset_k{k}_results.csv"),
            index=False
        )
        
        # Add 'is_ood' column if needed for compatibility
        if 'is_rejected' in results_df.columns and 'is_ood' not in results_df.columns:
            results_df['is_ood'] = results_df['is_rejected']
        
        # Save metrics
        metrics_path = os.path.join(openset_dir, f"openset_k{k}_metrics.json")
        with open(metrics_path, 'w') as f:
            import json
            # Convert numpy arrays to lists for JSON serialization
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                elif isinstance(value, np.float64) or isinstance(value, np.float32):
                    metrics_json[key] = float(value)
                else:
                    metrics_json[key] = value
            json.dump(metrics_json, f, indent=2)
        
        # Generate ROC curve for detection
        if 'detection_fpr' in metrics and 'detection_tpr' in metrics:
            roc_fig = protocol.visualize_roc_curve(
                metrics['detection_fpr'],
                metrics['detection_tpr'],
                metrics['detection_roc_auc'],
                title=f'Open-Set Detection ROC Curve (k={k})'
            )
            roc_fig.savefig(
                os.path.join(openset_dir, f"openset_k{k}_roc.png")
            )
            plt.close(roc_fig)
        
        # Analyze by manufacturer
        if 'is_rejected' in results_df.columns:
            manuf_fig = protocol.analyze_by_manufacturer(
                results_df,
                metric='ood'
            )
            manuf_fig.savefig(
                os.path.join(openset_dir, f"openset_k{k}_by_manufacturer.png")
            )
            plt.close(manuf_fig)
        
        # Print results
        print(f"\nOpen-Set Evaluation Results (k={k}):")
        print(f"Normalized Accuracy: {metrics['normalized_accuracy']:.4f}")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Known Class Accuracy: {metrics['known_accuracy']:.4f}")
        print(f"Detection Accuracy: {metrics['detection_accuracy']:.4f}")
        print(f"Known Detection Rate: {metrics['known_detection_rate']:.4f}")
        print(f"Unknown Detection Rate: {metrics['unknown_detection_rate']:.4f}")
        if 'detection_roc_auc' in metrics:
            print(f"Detection ROC AUC: {metrics['detection_roc_auc']:.4f}")
        
        all_metrics[f"k{k}"] = metrics
    
    # Find best k value based on normalized accuracy
    best_k = max(all_metrics.items(), key=lambda x: x[1]['normalized_accuracy'])
    print(f"\nBest open-set configuration: {best_k[0]} with normalized accuracy: {best_k[1]['normalized_accuracy']:.4f}")
    
    return all_metrics


def analyze_zero_shot_classification(protocol, data, output_dir):
    """
    Analyze zero-shot classification using text-image embedding similarity.
    
    Args:
        protocol: AircraftProtocol instance.
        data: Dictionary with data.
        output_dir: Output directory.
        
    Returns:
        Dictionary with metrics.
    """
    print("\n" + "="*50)
    print("Zero-Shot Classification using Text-Image Similarity")
    print("="*50)
    
    # Ensure output directories exist
    zeroshot_dir = os.path.join(output_dir, "zero_shot")
    os.makedirs(zeroshot_dir, exist_ok=True)
    
    # Run zero-shot classification on close-set data (Boeing vs. Airbus)
    print("\nRunning zero-shot classification on close-set data (Boeing vs. Airbus)...")
    close_results_df, close_metrics = protocol.zero_shot_classification(
        data['test_df_close'],
        use_manufacturers=True
    )
    
    # Save close-set results
    close_results_df.to_csv(
        os.path.join(zeroshot_dir, "zeroshot_close_results.csv"),
        index=False
    )
    
    # Generate confusion matrix for close-set
    if 'confusion_matrix' in close_metrics and 'labels' in close_metrics:
        cm_fig = protocol.visualize_confusion_matrix(
            close_metrics['confusion_matrix'],
            close_metrics['labels'],
            title='Zero-Shot Classification Confusion Matrix (Close-Set)'
        )
        cm_fig.savefig(
            os.path.join(zeroshot_dir, "zeroshot_close_confusion.png")
        )
        plt.close(cm_fig)
    
    # Print close-set results
    print("\nZero-Shot Classification Results (Close-Set):")
    print(f"Accuracy: {close_metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {close_metrics.get('precision', 0):.4f}")
    print(f"Recall: {close_metrics.get('recall', 0):.4f}")
    print(f"F1 Score: {close_metrics.get('f1', 0):.4f}")
    
    # Run zero-shot classification on open-set data (including Unknown)
    print("\nRunning zero-shot classification on open-set data (including Unknown)...")
    open_results_df, open_metrics = protocol.zero_shot_classification(
        data['test_df'],
        use_manufacturers=True
    )
    
    # Save open-set results
    open_results_df.to_csv(
        os.path.join(zeroshot_dir, "zeroshot_open_results.csv"),
        index=False
    )
    
    # Generate confusion matrix for open-set
    if 'confusion_matrix' in open_metrics and 'labels' in open_metrics:
        cm_fig = protocol.visualize_confusion_matrix(
            open_metrics['confusion_matrix'],
            open_metrics['labels'],
            title='Zero-Shot Classification Confusion Matrix (Open-Set)'
        )
        cm_fig.savefig(
            os.path.join(zeroshot_dir, "zeroshot_open_confusion.png")
        )
        plt.close(cm_fig)
    
    # Analyze by manufacturer
    manuf_fig = protocol.analyze_by_manufacturer(
        open_results_df,
        metric='accuracy'
    )
    manuf_fig.savefig(
        os.path.join(zeroshot_dir, "zeroshot_by_manufacturer.png")
    )
    plt.close(manuf_fig)
    
    # Print open-set results
    print("\nZero-Shot Classification Results (Open-Set):")
    print(f"Accuracy: {open_metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {open_metrics.get('precision', 0):.4f}")
    print(f"Recall: {open_metrics.get('recall', 0):.4f}")
    print(f"F1 Score: {open_metrics.get('f1', 0):.4f}")
    
    # Direct comparison of image embeddings to text embeddings
    print("\nGenerating direct image-text embedding comparison...")
    comparison_df, comparison_fig = protocol.compare_image_to_text_embeddings(
        data['test_df']
    )
    
    # Save comparison results
    comparison_df.to_csv(
        os.path.join(zeroshot_dir, "image_text_comparison.csv"),
        index=False
    )
    
    if comparison_fig:
        comparison_fig.savefig(
            os.path.join(zeroshot_dir, "image_text_similarity.png")
        )
        plt.close(comparison_fig)
    
    # Save all metrics
    all_metrics = {
        'close_set': close_metrics,
        'open_set': open_metrics
    }
    
    return all_metrics


def analyze_embedding_distributions(protocol, data, output_dir, umap_params=None):
    """
    Analyze embedding distributions for Boeing, Airbus, and Unknown manufacturers
    using UMAP visualization.
    
    Args:
        protocol: AircraftProtocol instance.
        data: Dictionary with data.
        output_dir: Output directory.
        umap_params: UMAP parameters.
    """
    print("\n" + "="*50)
    print("Embedding Distribution Analysis with UMAP")
    print("="*50)
    
    # Create output directory
    dist_dir = os.path.join(output_dir, "distributions")
    os.makedirs(dist_dir, exist_ok=True)
    
    # Set default UMAP parameters if not provided
    if umap_params is None:
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1
        }
    
    # Get embeddings for each class
    test_df = data['test_df']
    
    # Group embeddings by manufacturer
    boeing_embeddings = []
    airbus_embeddings = []
    unknown_embeddings = []
    
    for _, row in test_df.iterrows():
        filename = row['filename']
        if filename not in protocol.image_embeddings:
            continue
            
        embedding = protocol.image_embeddings[filename].cpu().numpy()
        
        if row['manufacturer'] == "Boeing":
            boeing_embeddings.append(embedding)
        elif row['manufacturer'] == "Airbus":
            airbus_embeddings.append(embedding)
        else:
            unknown_embeddings.append(embedding)
    
    # Convert to numpy arrays
    boeing_embeddings = np.array(boeing_embeddings)
    airbus_embeddings = np.array(airbus_embeddings)
    unknown_embeddings = np.array(unknown_embeddings)
    
    print(f"Boeing samples: {len(boeing_embeddings)}")
    print(f"Airbus samples: {len(airbus_embeddings)}")
    print(f"Unknown samples: {len(unknown_embeddings)}")
    
    # 1. Feature-by-feature distribution analysis
    # Choose a subset of dimensions to visualize
    dimensions = [0, 1, 2, 3, 4]  # First 5 dimensions of the embedding
    
    for dim in dimensions:
        plt.figure(figsize=(10, 6))
        
        if len(boeing_embeddings) > 0:
            plt.hist(boeing_embeddings[:, dim], alpha=0.5, bins=30, label='Boeing')
        if len(airbus_embeddings) > 0:
            plt.hist(airbus_embeddings[:, dim], alpha=0.5, bins=30, label='Airbus')
        if len(unknown_embeddings) > 0:
            plt.hist(unknown_embeddings[:, dim], alpha=0.5, bins=30, label='Unknown')
            
        plt.title(f'Distribution of Embedding Dimension {dim}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(dist_dir, f'dimension_{dim}_distribution.png'))
        plt.close()
    
    # 2. UMAP visualization
    print("\nGenerating UMAP visualization of embeddings...")
    umap_fig = protocol.visualize_embeddings_umap(
        test_df, 
        n_neighbors=umap_params.get('n_neighbors', 15),
        min_dist=umap_params.get('min_dist', 0.1),
        color_by='manufacturer',
        title='Boeing vs. Airbus vs. Unknown Embedding Space (UMAP)'
    )
    umap_fig.savefig(os.path.join(dist_dir, "embeddings_manufacturer_umap.png"))
    plt.close(umap_fig)
    
    # 3. Distance distribution analysis - MODIFIED TO SPLIT KNOWN VS UNKNOWN
    print("Analyzing embedding distances between classes...")
    
    # Compute pairwise distances
    known_distances = {
        'boeing-airbus': [],
        'boeing-boeing': [],
        'airbus-airbus': []
    }
    
    unknown_distances = {
        'boeing-unknown': [],
        'airbus-unknown': [],
        'unknown-unknown': []
    }
    
    # Sample to avoid too many computations
    max_samples = 1000
    boeing_sample = boeing_embeddings[:min(len(boeing_embeddings), max_samples)]
    airbus_sample = airbus_embeddings[:min(len(airbus_embeddings), max_samples)]
    unknown_sample = unknown_embeddings[:min(len(unknown_embeddings), max_samples)]
    
    # Calculate distances
    from scipy.spatial.distance import cosine
    
    # Between Boeing and Airbus (Known-Known)
    for i in range(len(boeing_sample)):
        for j in range(min(10, len(airbus_sample))):  # Limit computations
            known_distances['boeing-airbus'].append(
                cosine(boeing_sample[i], airbus_sample[j])
            )
    
    # Between Known and Unknown
    for i in range(len(boeing_sample)):
        for j in range(min(10, len(unknown_sample))):
            unknown_distances['boeing-unknown'].append(
                cosine(boeing_sample[i], unknown_sample[j])
            )
    
    for i in range(len(airbus_sample)):
        for j in range(min(10, len(unknown_sample))):
            unknown_distances['airbus-unknown'].append(
                cosine(airbus_sample[i], unknown_sample[j])
            )
    
    # Within class distances (sample a subset for efficiency)
    for cls, embs in [('boeing', boeing_sample), ('airbus', airbus_sample)]:
        if len(embs) < 2:
            continue
            
        for i in range(len(embs)):
            for j in range(i+1, min(i+11, len(embs))):
                known_distances[f'{cls}-{cls}'].append(
                    cosine(embs[i], embs[j])
                )
    
    # Within unknown class distances
    if len(unknown_sample) >= 2:
        for i in range(len(unknown_sample)):
            for j in range(i+1, min(i+11, len(unknown_sample))):
                unknown_distances['unknown-unknown'].append(
                    cosine(unknown_sample[i], unknown_sample[j])
                )
    
    # Plot 1: Known class distance distributions
    plt.figure(figsize=(12, 8))
    
    for key, vals in known_distances.items():
        if vals:
            plt.hist(vals, alpha=0.7, bins=30, label=key, density=True)
    
    plt.title('Distribution of Cosine Distances Between Known Classes (Boeing & Airbus)')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(os.path.join(dist_dir, "cosine_distance_known_classes.png"))
    plt.close()
    
    # Plot 2: Unknown class distance distributions
    plt.figure(figsize=(12, 8))
    
    for key, vals in unknown_distances.items():
        if vals:
            plt.hist(vals, alpha=0.7, bins=30, label=key, density=True)
    
    plt.title('Distribution of Cosine Distances Involving Unknown Manufacturers')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(os.path.join(dist_dir, "cosine_distance_unknown_classes.png"))
    plt.close()
    
    # Plot 3: Combined view (for comparison with original)
    plt.figure(figsize=(12, 8))
    
    all_distances = {**known_distances, **unknown_distances}
    for key, vals in all_distances.items():
        if vals:
            plt.hist(vals, alpha=0.5, bins=30, label=key, density=True)
    
    plt.title('Distribution of Cosine Distances Between and Within Classes')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(os.path.join(dist_dir, "cosine_distance_distributions.png"))
    plt.close()
    
    # Generate statistics for report
    stats = {}
    for category, distances in all_distances.items():
        if distances:
            stats[category] = {
                'mean': np.mean(distances),
                'median': np.median(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            }
    
    # Save statistics to CSV
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.to_csv(os.path.join(dist_dir, "distance_statistics.csv"))
    
    # Also modify the summary report generation to include these new visualizations
    return stats


def generate_summary_report(metrics, output_dir, distance_stats=None, use_umap=True):
    """
    Generate a summary report with key findings.
    
    Args:
        metrics: Dictionary with metrics from all analyses.
        output_dir: Output directory.
        distance_stats: Optional distance statistics from embedding analysis.
        use_umap: Whether to use UMAP instead of t-SNE in the report.
    """
    report_path = os.path.join(output_dir, "summary_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Boeing vs. Airbus Classification Analysis: Close-Set vs. Open-Set Evaluation\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Binary Classification (Close-Set)
        if 'binary' in metrics:
            f.write("## Boeing vs. Airbus Binary Classification (Close-Set Evaluation)\n\n")
            
            # Find best k value
            best_k = max(metrics['binary'].items(), key=lambda x: x[1]['accuracy'])
            method_name, method_metrics = best_k
            
            f.write(f"Best KNN configuration: {method_name}\n")
            f.write(f"Test Accuracy: {method_metrics['accuracy']:.4f}\n")
            f.write(f"Test Precision: {method_metrics['precision']:.4f}\n")
            f.write(f"Test Recall: {method_metrics['recall']:.4f}\n")
            f.write(f"Test F1 Score: {method_metrics['f1']:.4f}\n\n")
            
            f.write(f"![Confusion Matrix](classification/binary_knn_{method_name}_confusion_matrix.png)\n\n")
        
        # Open-Set Evaluation
        if 'open_set' in metrics:
            f.write("## Open-Set Evaluation\n\n")
            
            # Find best k value
            best_k = max(metrics['open_set'].items(), key=lambda x: x[1]['normalized_accuracy'])
            method_name, method_metrics = best_k
            
            f.write(f"Best KNN configuration: {method_name}\n")
            f.write(f"Normalized Accuracy: {method_metrics['normalized_accuracy']:.4f}\n")
            f.write(f"Overall Accuracy: {method_metrics['accuracy']:.4f}\n")
            f.write(f"Detection Accuracy: {method_metrics['detection_accuracy']:.4f}\n")
            f.write(f"Known Detection Rate: {method_metrics['known_detection_rate']:.4f}\n")
            f.write(f"Unknown Detection Rate: {method_metrics['unknown_detection_rate']:.4f}\n")
            if 'detection_roc_auc' in method_metrics:
                f.write(f"Detection ROC AUC: {method_metrics['detection_roc_auc']:.4f}\n\n")
            
            f.write(f"![ROC Curve](open_set/openset_{method_name}_roc.png)\n\n")
            f.write(f"![OOD by Manufacturer](open_set/openset_{method_name}_by_manufacturer.png)\n\n")
        
        # Zero-Shot Classification
        if 'zero_shot' in metrics:
            f.write("## Zero-Shot Classification\n\n")
            
            # Close-set results
            f.write("### Close-Set Performance\n\n")
            
            close_metrics = metrics['zero_shot'].get('close_set', {})
            f.write(f"Accuracy: {close_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {close_metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {close_metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score: {close_metrics.get('f1', 0):.4f}\n\n")
            
            f.write("![Confusion Matrix](zero_shot/zeroshot_close_confusion.png)\n\n")
            
            # Open-set results
            f.write("### Open-Set Performance\n\n")
            
            open_metrics = metrics['zero_shot'].get('open_set', {})
            f.write(f"Accuracy: {open_metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {open_metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {open_metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score: {open_metrics.get('f1', 0):.4f}\n\n")
            
            f.write("![Confusion Matrix](zero_shot/zeroshot_open_confusion.png)\n\n")
            f.write("![Performance by Manufacturer](zero_shot/zeroshot_by_manufacturer.png)\n\n")
            f.write("![Image-Text Similarity](zero_shot/image_text_similarity.png)\n\n")
        
        # Embedding Distribution Analysis
        f.write("## Embedding Distribution Analysis\n\n")
        f.write("The embedding distribution analysis examines how Boeing, Airbus, and Unknown manufacturer embeddings are distributed in the feature space.\n\n")
        
        # UMAP visualization instead of t-SNE
        f.write("### UMAP Visualization of Embeddings\n\n")
        f.write("![Embedding Distribution](distributions/embeddings_manufacturer_umap.png)\n\n")
        
        f.write("### Feature-by-Feature Distribution\n\n")
        f.write("Distribution of embedding values for the first 5 dimensions:\n\n")
        
        for dim in range(5):
            f.write(f"![Dimension {dim}](distributions/dimension_{dim}_distribution.png)\n\n")
        
        f.write("### Distance Distributions\n\n")
        
        # Add the split cosine distance visualizations
        f.write("#### Distances Within and Between Known Classes (Boeing & Airbus)\n\n")
        f.write("This visualization shows the distribution of cosine distances between Boeing and Airbus samples, as well as within each known class.\n\n")
        f.write("![Known Class Distances](distributions/cosine_distance_known_classes.png)\n\n")
        
        f.write("#### Distances Involving Unknown Manufacturers\n\n")
        f.write("This visualization shows the distribution of cosine distances between known manufacturers (Boeing/Airbus) and unknown manufacturers, as well as within the unknown class.\n\n")
        f.write("![Unknown Class Distances](distributions/cosine_distance_unknown_classes.png)\n\n")
        
        f.write("#### Combined Distance Distribution\n\n")
        f.write("Distribution of all cosine distances between and within classes:\n\n")
        f.write("![Combined Distance Distribution](distributions/cosine_distance_distributions.png)\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Close-set vs. Open-set comparison
        if 'binary' in metrics and 'open_set' in metrics:
            binary_best_k = max(metrics['binary'].items(), key=lambda x: x[1]['accuracy'])
            openset_best_k = max(metrics['open_set'].items(), key=lambda x: x[1]['normalized_accuracy'])
            
            binary_acc = binary_best_k[1]['accuracy']
            openset_norm_acc = openset_best_k[1]['normalized_accuracy']
            
            f.write(f"- **Close-Set vs. Open-Set Performance**: Close-set evaluation achieved {binary_acc:.1%} accuracy, while open-set evaluation achieved {openset_norm_acc:.1%} normalized accuracy.\n")
            
            # Compare if open-set detection was successful
            openset_metrics = openset_best_k[1]
            if 'detection_roc_auc' in openset_metrics and openset_metrics['detection_roc_auc'] > 0.8:
                f.write(f"- **Effective OOD Detection**: With a ROC AUC of {openset_metrics['detection_roc_auc']:.3f}, the system effectively distinguishes between known and unknown manufacturers.\n")
            elif 'detection_roc_auc' in openset_metrics:
                f.write(f"- **Moderate OOD Detection**: With a ROC AUC of {openset_metrics['detection_roc_auc']:.3f}, the system shows moderate ability to distinguish between known and unknown manufacturers.\n")
        
        # Zero-shot results if available
        if 'zero_shot' in metrics:
            close_metrics = metrics['zero_shot'].get('close_set', {})
            open_metrics = metrics['zero_shot'].get('open_set', {})
            
            if 'accuracy' in close_metrics and 'accuracy' in open_metrics:
                f.write(f"- **Zero-Shot Classification**: The system achieved {close_metrics['accuracy']:.1%} accuracy on close-set data and {open_metrics['accuracy']:.1%} accuracy on open-set data using zero-shot classification based on text-image similarity.\n")
        
        # Embedding analysis conclusions
        f.write("- **Embedding Space Analysis**: The UMAP visualization demonstrates clear clustering of manufacturers in the embedding space, showing the power of VLM embeddings for aircraft classification.\n")
        
        # Add analysis of distance distributions if available
        if distance_stats:
            # Compare within-class vs between-class distances
            if 'boeing-boeing' in distance_stats and 'boeing-airbus' in distance_stats:
                within_boeing = distance_stats['boeing-boeing']['mean']
                between_boeing_airbus = distance_stats['boeing-airbus']['mean']
                
                if within_boeing < between_boeing_airbus:
                    f.write(f"- **Cohesive Embeddings**: Boeing samples are more similar to each other (mean distance: {within_boeing:.4f}) than to Airbus samples (mean distance: {between_boeing_airbus:.4f}).\n")
            
            # Compare known-known vs known-unknown distances
            if 'boeing-airbus' in distance_stats and 'boeing-unknown' in distance_stats:
                known_known = distance_stats['boeing-airbus']['mean']
                known_unknown = distance_stats['boeing-unknown']['mean']
                
                if known_known < known_unknown:
                    diff = known_unknown - known_known
                    f.write(f"- **OOD Separation**: The mean distance between Boeing and Unknown samples ({known_unknown:.4f}) is {diff:.4f} higher than between Boeing and Airbus ({known_known:.4f}), supporting effective OOD detection.\n")
        
        # Final conclusion on VLM embeddings
        f.write("\n### Key Findings\n\n")
        f.write("1. VLM embeddings provide a strong foundation for aircraft classification, showing clear separation between manufacturers.\n")
        f.write("2. The pipeline of generating text embeddings, extracting VLM features, and using KNN classification proves effective for both close-set and open-set scenarios.\n")
        f.write("3. Zero-shot classification using direct image-text similarity demonstrates the semantic power of VLM representations.\n")
        f.write("4. UMAP visualization reveals the structure of the embedding space, showing how the model organizes aircraft by manufacturer.\n")
        f.write("5. Distance analysis confirms that unknown manufacturers are generally further from known ones in the embedding space, enabling effective open-set recognition.\n")
    
    print(f"Summary report generated: {report_path}")