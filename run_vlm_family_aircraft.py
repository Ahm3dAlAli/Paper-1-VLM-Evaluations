#!/usr/bin/env python
"""
Run Aircraft Classification Analysis

This script runs the full pipeline for evaluating vision-language model embeddings:
1. Generate text embeddings for aircraft models and manufacturers
2. Extract image features through VLM
3. Run KNN classifier for aircraft model classification
4. Run KNN classifier for Boeing vs. Airbus binary classification
5. Perform open-set evaluation for unknown manufacturers
6. Compare image embeddings with text embeddings
7. Visualize embedding space using UMAP

Usage:
    python run_vlm_aircraft.py --analyze_all
    python run_vlm_aircraft.py --analyze_models --analyze_binary --k_values 1,3,5,10
    python run_vlm_aircraft.py --analyze_open_set --analyze_embeddings
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from aircraft_family_protocol import AircraftProtocol
from aircraft_family_analysis import (
    setup_output_dir,
    load_and_prepare_data,
    analyze_model_classification,
    analyze_binary_classification,
    analyze_open_set_evaluation,
    analyze_zero_shot_classification,
    analyze_embedding_distributions,
    generate_summary_report
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Aircraft Classification Analysis")
    
    # Data options
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
                        help="CLIP model to use (default: openai/clip-vit-base-patch32)")
    
    # Analysis options
    parser.add_argument("--analyze_models", action="store_true",
                        help="Run aircraft model classification analysis")
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
    
    # KNN options
    parser.add_argument("--k_values", type=str, default="1,3,5,10",
                        help="K values for KNN (comma-separated)")
    
    # OOD options
    parser.add_argument("--ood_threshold", type=float, default=0.7,
                        help="Similarity threshold for OOD detection")
    
    # UMAP options
    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="aircraft_results",
                        help="Directory to save results")
    parser.add_argument("--timestamp", action="store_true",
                        help="Add timestamp to output directory")
    
    # Execution options
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # If no analysis is specified, run all
    if not any([args.analyze_models, args.analyze_binary, args.analyze_open_set, 
                args.analyze_zero_shot, args.analyze_embeddings, args.analyze_all]):
        args.analyze_all = True
        print("No specific analysis requested. Running all analyses.")
    
    return args


def create_dataset_csvs(data_dir):
    """
    Create train/val/test CSV files from the annotation files.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        Tuple of paths to the created CSV files (train_csv, val_csv, test_csv)
    """
    # Find the annotation directory
    annotation_dir = None
    for root, dirs, files in os.walk(data_dir):
        if any(f.endswith('.txt') and ('manufacturer' in f or 'families' in f or 'mailies' in f) for f in files):
            annotation_dir = root
            print(f"Found annotation directory at: {annotation_dir}")
            break
    
    if not annotation_dir:
        print("Could not find annotation directory with manufacturer/families files")
        return None, None, None
    
    # Try to find the annotation files
    splits = ['train', 'val', 'test']
    csv_paths = []
    
    for split in splits:
        # Look for standard annotation files
        annotation_files = []
        for prefix in ['images_manufacturer_', 'images_family_', 'images_class_']:
            file_path = os.path.join(annotation_dir, f"{prefix}{split}.txt")
            if os.path.exists(file_path):
                annotation_files.append((file_path, prefix))
        
        if not annotation_files:
            print(f"Warning: No annotation file found for {split}")
            csv_paths.append(None)
            continue
        
        # Use the first available annotation file
        annotation_file, prefix = annotation_files[0]
        print(f"Using {annotation_file} for {split}")
        
        # Read the annotation file and create a DataFrame
        rows = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    class_value = parts[1]
                    
                    rows.append({
                        'filename': f"{image_id}.jpg",
                        'Classes': class_value,
                        'Labels': 0  # Default label
                    })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = os.path.join(data_dir, f"{split}.csv")
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
        
        # Print statistics
        print(f"Created {split} CSV with {len(df)} samples")
        class_counts = df['Classes'].value_counts()
        print(f"  Number of unique classes: {len(class_counts)}")
        print(f"  Top 5 classes: {class_counts.head(5).to_dict()}")
    
    return tuple(csv_paths)


def main():
    """Main function to run the aircraft classification analysis."""
    args = parse_args()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Verify data directory structure and create CSV files if needed
        train_csv = os.path.join(args.data_dir, args.train_csv)
        val_csv = os.path.join(args.data_dir, args.val_csv)
        test_csv = os.path.join(args.data_dir, args.test_csv)
        
        # Check if CSVs exist, if not create them
        if not all(os.path.exists(csv) for csv in [train_csv, val_csv, test_csv]):
            print("Creating CSV files from annotation files...")
            train_csv, val_csv, test_csv = create_dataset_csvs(args.data_dir)
            
            if not all([train_csv, val_csv, test_csv]):
                print("Failed to create all necessary CSV files.")
                sys.exit(1)
        
        # Setup output directory
        output_dir = setup_output_dir(args)
        print(f"Results will be saved to: {output_dir}")
        
        # Initialize protocol
        print(f"Initializing AircraftProtocol with {args.clip_model}...")
        protocol = AircraftProtocol(
            clip_model_name=args.clip_model,
            data_dir=args.data_dir
        )
        
        # Load and prepare data - This also generates text embeddings first
        data = load_and_prepare_data(args, protocol)
        
        # Check if we have any embeddings before proceeding
        if not protocol.image_embeddings:
            print("Warning: No image embeddings were generated. Cannot proceed with analysis.")
            print("This might be due to incorrect image paths or missing images.")
            sys.exit(1)
        
        # Run requested analyses
        metrics = {}
        
        if args.analyze_all or args.analyze_models:
            print("\nRunning aircraft model classification analysis...")
            metrics['model'] = analyze_model_classification(protocol, data, args, output_dir)
        
        if args.analyze_all or args.analyze_binary:
            print("\nRunning Boeing vs. Airbus binary classification analysis (close-set)...")
            metrics['binary'] = analyze_binary_classification(protocol, data, args, output_dir)
        
        if args.analyze_all or args.analyze_open_set:
            print("\nRunning open-set evaluation analysis...")
            metrics['open_set'] = analyze_open_set_evaluation(protocol, data, args, output_dir)
        
        if args.analyze_all or args.analyze_zero_shot:
            print("\nRunning zero-shot classification analysis...")
            metrics['zero_shot'] = analyze_zero_shot_classification(protocol, data, output_dir)
        
        if args.analyze_all or args.analyze_embeddings:
            print("\nRunning embedding distribution analysis with UMAP...")
            umap_params = {
                'n_neighbors': args.umap_neighbors,
                'min_dist': args.umap_min_dist
            }
            distance_stats = analyze_embedding_distributions(protocol, data, output_dir, umap_params)
        else:
            distance_stats = None
        
        # Generate summary report
        if metrics:
            print("\nGenerating summary report...")
            generate_summary_report(metrics, output_dir, distance_stats)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print("\nNext steps:")
        print("- Review the summary_report.md file for key findings")
        print("- Explore the visualizations to understand the embedding space")
        print("- Compare aircraft model classification with manufacturer classification")
        print("- Analyze how text embeddings relate to image embeddings in zero-shot classification")
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure to have all required modules installed:")
        print("pip install torch torchvision transformers pillow matplotlib pandas seaborn scikit-learn tqdm umap-learn")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()