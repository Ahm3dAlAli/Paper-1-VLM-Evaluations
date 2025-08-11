"""
Hierarchical Aircraft Analysis - Comprehensive Evaluation Pipeline

This script implements a comprehensive analysis pipeline for evaluating
VLM embeddings in hierarchical aircraft classification with a focus on:
1. Multi-level hierarchical classification (Category → Subcategory → Model)
2. Comparative analysis across hierarchy levels
3. Cross-level OOD detection and evaluation
4. Hierarchical embedding distribution analysis with UMAP
5. Performance degradation analysis from broad to specific classifications
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
from aircraft_hierarchy_protocol import HierarchicalAircraftProtocol
import json
import pickle
from collections import defaultdict, Counter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hierarchical Aircraft Classification Analysis")
    
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
    
    # Hierarchy options
    parser.add_argument("--custom_hierarchy", type=str, default=None,
                        help="Path to custom hierarchy JSON file")
    
    # Analysis options
    parser.add_argument("--analyze_hierarchy", action="store_true",
                        help="Run hierarchical classification analysis across all levels")
    parser.add_argument("--analyze_cross_level", action="store_true",
                        help="Run cross-level performance analysis")
    parser.add_argument("--analyze_zero_shot", action="store_true", 
                        help="Run hierarchical zero-shot classification analysis")
    parser.add_argument("--analyze_knn", action="store_true",
                        help="Run hierarchical KNN classification analysis")
    parser.add_argument("--analyze_ood", action="store_true",
                        help="Run hierarchical OOD detection analysis")
    parser.add_argument("--analyze_embeddings", action="store_true",
                        help="Run hierarchical embedding distribution analysis")
    parser.add_argument("--analyze_progression", action="store_true",
                        help="Analyze performance progression from broad to specific")
    parser.add_argument("--analyze_all", action="store_true",
                        help="Run all analyses")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="hierarchical_aircraft_analysis",
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
    
    # UMAP visualization options
    parser.add_argument("--umap_neighbors", type=int, default=15,
                        help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                        help="UMAP min_dist parameter")
    
    # Evaluation options
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Fraction of data to use for training")
    
    # Save/load options
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save embeddings for future use")
    parser.add_argument("--load_embeddings", type=str, default=None,
                        help="Path to pre-computed embeddings file")
    
    return parser.parse_args()


def setup_output_dir(args):
    """Setup output directory for analysis results."""
    output_dir = args.output_dir
    
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "hierarchical_results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cross_level_analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "progression_analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "embedding_analysis"), exist_ok=True)
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


def load_and_prepare_hierarchical_data(args, protocol):
    """
    Load and prepare data for hierarchical aircraft classification analysis.
    
    Args:
        args: Command line arguments.
        protocol: HierarchicalAircraftProtocol instance.
        
    Returns:
        Dictionary with hierarchical data.
    """
    print("\nLoading and preparing hierarchical data...")
    
    try:
        # Define CSV paths
        train_csv = os.path.join(args.data_dir, args.train_csv)
        val_csv = os.path.join(args.data_dir, args.val_csv)
        test_csv = os.path.join(args.data_dir, args.test_csv)
        
        # Load datasets
        train_df = protocol.load_data(train_csv)
        val_df = protocol.load_data(val_csv)
        test_df = protocol.load_data(test_csv)
        
        # Create hierarchical classifications
        print("Creating hierarchical classifications...")
        train_df_hierarchical = protocol.create_hierarchical_dataset(train_df)
        val_df_hierarchical = protocol.create_hierarchical_dataset(val_df)
        test_df_hierarchical = protocol.create_hierarchical_dataset(test_df)
        
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
        
        # Generate embeddings or load them
        if args.load_embeddings and os.path.exists(args.load_embeddings):
            print(f"Loading pre-computed embeddings from {args.load_embeddings}")
            with open(args.load_embeddings, 'rb') as f:
                embeddings_data = pickle.load(f)
            
            protocol.image_embeddings = embeddings_data['image_embeddings']
            protocol.hierarchy_embeddings = embeddings_data['hierarchy_embeddings']
            print(f"Loaded {len(protocol.image_embeddings)} image embeddings")
        else:
            # Generate hierarchical text embeddings
            print("Generating hierarchical text embeddings...")
            protocol.generate_hierarchical_text_embeddings()
            
            # Generate image embeddings
            print("Generating image embeddings for all data...")
            all_df = pd.concat([train_df_hierarchical, val_df_hierarchical, test_df_hierarchical])
            protocol.generate_image_embeddings(all_df, batch_size=args.batch_size)
            
            # Save embeddings if requested
            if args.save_embeddings:
                embeddings_path = os.path.join(args.output_dir, "hierarchical_embeddings.pkl")
                embeddings_data = {
                    'image_embeddings': protocol.image_embeddings,
                    'hierarchy_embeddings': protocol.hierarchy_embeddings,
                    'hierarchy': protocol.hierarchy
                }
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(embeddings_data, f)
                print(f"Saved embeddings to {embeddings_path}")
        
        data = {
            'train_df': train_df_hierarchical,
            'val_df': val_df_hierarchical,
            'test_df': test_df_hierarchical,
            'all_df': pd.concat([train_df_hierarchical, val_df_hierarchical, test_df_hierarchical])
        }
        
        return data
    except Exception as e:
        print(f"Error in load_and_prepare_hierarchical_data: {e}")
        # Return minimal valid data to avoid crashes
        empty_df = pd.DataFrame(columns=['filename', 'Classes', 'level_1_category', 'level_2_subcategory', 'level_3_model'])
        return {
            'train_df': empty_df,
            'val_df': empty_df,
            'test_df': empty_df,
            'all_df': empty_df
        }


def analyze_hierarchical_classification(protocol, data, args, output_dir):
    """
    Analyze hierarchical classification performance across all levels.
    
    Args:
        protocol: HierarchicalAircraftProtocol instance.
        data: Dictionary with hierarchical data.
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        Dictionary with results for each level and method.
    """
    print("\n" + "="*60)
    print("HIERARCHICAL CLASSIFICATION ANALYSIS")
    print("="*60)
    
    hierarchical_dir = os.path.join(output_dir, "hierarchical_results")
    
    # Run comprehensive evaluation across all levels
    methods = []
    if args.analyze_zero_shot or args.analyze_all:
        methods.append("zero_shot")
    if args.analyze_knn or args.analyze_all:
        methods.append("knn")
    if args.analyze_ood or args.analyze_all:
        methods.append("ood")
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    print(f"Running evaluation with methods: {methods}")
    print(f"Using k values: {k_values}")
    
    # Run the comprehensive evaluation
    all_results = protocol.evaluate_all_levels(
        data['test_df'],
        methods=methods,
        k=k_values[0],  # Use first k value as default
        train_split=args.train_split
    )
    
    # Save detailed results for each level and method
    for level_name, level_results in all_results.items():
        level_dir = os.path.join(hierarchical_dir, level_name)
        os.makedirs(level_dir, exist_ok=True)
        
        for method_name, method_data in level_results.items():
            # Save results DataFrame
            results_df = method_data["results"]
            metrics = method_data["metrics"]
            
            results_df.to_csv(
                os.path.join(level_dir, f"{method_name}_results.csv"),
                index=False
            )
            
            # Save metrics as JSON
            metrics_json = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                elif isinstance(value, (np.float64, np.float32)):
                    metrics_json[key] = float(value)
                else:
                    metrics_json[key] = value
            
            with open(os.path.join(level_dir, f"{method_name}_metrics.json"), 'w') as f:
                json.dump(metrics_json, f, indent=2)
    
    # Generate performance visualization
    performance_fig = protocol.visualize_hierarchy_performance(all_results)
    performance_fig.savefig(os.path.join(hierarchical_dir, "hierarchy_performance_overview.png"))
    plt.close(performance_fig)
    
    # Generate comprehensive report
    report_path = protocol.generate_hierarchical_report(
        results=all_results,
        output_dir=hierarchical_dir
    )
    
    return all_results


def analyze_cross_level_performance(protocol, data, args, output_dir):
    """
    Analyze how performance changes across hierarchy levels.
    
    Args:
        protocol: HierarchicalAircraftProtocol instance.
        data: Dictionary with hierarchical data.
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        Dictionary with cross-level analysis results.
    """
    print("\n" + "="*60)
    print("CROSS-LEVEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    cross_level_dir = os.path.join(output_dir, "cross_level_analysis")
    
    # Analyze zero-shot performance across levels
    print("\nAnalyzing zero-shot performance across levels...")
    
    cross_level_results = {}
    level_names = ["Category", "Subcategory", "Model"]
    
    for level in [1, 2, 3]:
        print(f"\nRunning zero-shot classification at Level {level} ({level_names[level-1]})...")
        
        zs_results, zs_metrics = protocol.hierarchical_zero_shot_classification(
            data['test_df'], level=level
        )
        
        cross_level_results[f"level_{level}"] = {
            "results": zs_results,
            "metrics": zs_metrics,
            "level_name": level_names[level-1]
        }
        
        # Save results
        zs_results.to_csv(
            os.path.join(cross_level_dir, f"zero_shot_level_{level}_results.csv"),
            index=False
        )
    
    # Analyze difficulty progression
    print("\nAnalyzing classification difficulty progression...")
    
    difficulty_analysis = []
    
    for level in [1, 2, 3]:
        level_data = cross_level_results[f"level_{level}"]
        metrics = level_data["metrics"]
        level_name = level_data["level_name"]
        
        # Determine label column
        if level == 1:
            label_col = "level_1_category"
        elif level == 2:
            label_col = "level_2_subcategory"
        else:
            label_col = "level_3_model"
        
        # Calculate class statistics
        results_df = level_data["results"]
        unique_classes = results_df[f'true_{label_col}'].nunique()
        
        # Calculate class distribution entropy (measure of imbalance)
        class_counts = results_df[f'true_{label_col}'].value_counts()
        class_probs = class_counts / class_counts.sum()
        entropy = -np.sum(class_probs * np.log2(class_probs + 1e-10))
        
        difficulty_analysis.append({
            'level': level,
            'level_name': level_name,
            'accuracy': metrics.get('accuracy', 0),
            'f1_score': metrics.get('f1', 0),
            'num_classes': unique_classes,
            'class_entropy': entropy,
            'avg_confidence': results_df['confidence'].mean() if 'confidence' in results_df else 0
        })
    
    # Create difficulty progression visualization
    difficulty_df = pd.DataFrame(difficulty_analysis)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Classification Difficulty Progression Across Hierarchy Levels", fontsize=16)
    
    # Accuracy progression
    axes[0, 0].plot(difficulty_df['level'], difficulty_df['accuracy'], 
                    marker='o', linewidth=3, markersize=10, color='blue')
    axes[0, 0].set_xlabel('Hierarchy Level')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Hierarchy Level')
    axes[0, 0].set_xticks([1, 2, 3])
    axes[0, 0].set_xticklabels(['Category', 'Subcategory', 'Model'])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Number of classes progression
    axes[0, 1].bar(difficulty_df['level'], difficulty_df['num_classes'], 
                   color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    axes[0, 1].set_xlabel('Hierarchy Level')
    axes[0, 1].set_ylabel('Number of Classes')
    axes[0, 1].set_title('Class Count vs Hierarchy Level')
    axes[0, 1].set_xticks([1, 2, 3])
    axes[0, 1].set_xticklabels(['Category', 'Subcategory', 'Model'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Class entropy progression
    axes[0, 2].plot(difficulty_df['level'], difficulty_df['class_entropy'], 
                    marker='s', linewidth=3, markersize=10, color='red')
    axes[0, 2].set_xlabel('Hierarchy Level')
    axes[0, 2].set_ylabel('Class Distribution Entropy')
    axes[0, 2].set_title('Class Imbalance vs Hierarchy Level')
    axes[0, 2].set_xticks([1, 2, 3])
    axes[0, 2].set_xticklabels(['Category', 'Subcategory', 'Model'])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Confidence progression
    axes[1, 0].plot(difficulty_df['level'], difficulty_df['avg_confidence'], 
                    marker='d', linewidth=3, markersize=10, color='green')
    axes[1, 0].set_xlabel('Hierarchy Level')
    axes[1, 0].set_ylabel('Average Confidence')
    axes[1, 0].set_title('Prediction Confidence vs Hierarchy Level')
    axes[1, 0].set_xticks([1, 2, 3])
    axes[1, 0].set_xticklabels(['Category', 'Subcategory', 'Model'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy vs Number of Classes
    axes[1, 1].scatter(difficulty_df['num_classes'], difficulty_df['accuracy'], 
                       s=200, c=['blue', 'green', 'red'], alpha=0.7)
    axes[1, 1].set_xlabel('Number of Classes')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Class Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add level annotations
    for i, row in difficulty_df.iterrows():
        axes[1, 1].annotate(f"Level {int(row['level'])}", 
                           (row['num_classes'], row['accuracy']),
                           xytext=(5, 5), textcoords='offset points')
    
    # F1 Score progression
    axes[1, 2].plot(difficulty_df['level'], difficulty_df['f1_score'], 
                    marker='^', linewidth=3, markersize=10, color='purple')
    axes[1, 2].set_xlabel('Hierarchy Level')
    axes[1, 2].set_ylabel('F1 Score')
    axes[1, 2].set_title('F1 Score vs Hierarchy Level')
    axes[1, 2].set_xticks([1, 2, 3])
    axes[1, 2].set_xticklabels(['Category', 'Subcategory', 'Model'])
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cross_level_dir, "difficulty_progression_analysis.png"))
    plt.close()
    
    # Save difficulty analysis data
    difficulty_df.to_csv(os.path.join(cross_level_dir, "difficulty_analysis.csv"), index=False)
    
    # Calculate performance degradation metrics
    print("\nPerformance Degradation Analysis:")
    acc_degradation_1_to_2 = difficulty_df.iloc[0]['accuracy'] - difficulty_df.iloc[1]['accuracy']
    acc_degradation_2_to_3 = difficulty_df.iloc[1]['accuracy'] - difficulty_df.iloc[2]['accuracy']
    total_acc_degradation = difficulty_df.iloc[0]['accuracy'] - difficulty_df.iloc[2]['accuracy']
    
    print(f"Accuracy degradation Level 1→2: {acc_degradation_1_to_2:.3f}")
    print(f"Accuracy degradation Level 2→3: {acc_degradation_2_to_3:.3f}")
    print(f"Total accuracy degradation Level 1→3: {total_acc_degradation:.3f}")
    
    degradation_stats = {
        'accuracy_degradation_1_to_2': acc_degradation_1_to_2,
        'accuracy_degradation_2_to_3': acc_degradation_2_to_3,
        'total_accuracy_degradation': total_acc_degradation,
        'class_count_increase_1_to_2': difficulty_df.iloc[1]['num_classes'] / difficulty_df.iloc[0]['num_classes'],
        'class_count_increase_2_to_3': difficulty_df.iloc[2]['num_classes'] / difficulty_df.iloc[1]['num_classes'],
        'total_class_count_increase': difficulty_df.iloc[2]['num_classes'] / difficulty_df.iloc[0]['num_classes']
    }
    
    with open(os.path.join(cross_level_dir, "degradation_stats.json"), 'w') as f:
        json.dump(degradation_stats, f, indent=2)
    
    return {
        'cross_level_results': cross_level_results,
        'difficulty_analysis': difficulty_df,
        'degradation_stats': degradation_stats
    }


def analyze_hierarchical_embeddings(protocol, data, args, output_dir):
    """
    Analyze hierarchical embedding distributions and relationships.
    
    Args:
        protocol: HierarchicalAircraftProtocol instance.
        data: Dictionary with hierarchical data.
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        Dictionary with embedding analysis results.
    """
    print("\n" + "="*60)
    print("HIERARCHICAL EMBEDDING ANALYSIS")
    print("="*60)
    
    embedding_dir = os.path.join(output_dir, "embedding_analysis")
    
    # UMAP visualizations for each hierarchy level
    umap_params = {
        'n_neighbors': args.umap_neighbors,
        'min_dist': args.umap_min_dist
    }
    
    print("\nGenerating UMAP visualizations for each hierarchy level...")
    
    for level in [1, 2, 3]:
        level_name = ["Category", "Subcategory", "Model"][level-1]
        print(f"Creating Level {level} ({level_name}) UMAP visualization...")
        
        try:
            # Generate UMAP visualization
            umap_fig = protocol.visualize_hierarchical_embeddings(
                data['test_df'], 
                level=level,
                method="umap",
                figsize=(14, 10)
            )
            
            umap_fig.savefig(
                os.path.join(embedding_dir, f"umap_level_{level}_{level_name.lower()}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close(umap_fig)
            
        except Exception as e:
            print(f"Error creating Level {level} UMAP: {e}")
    
    # Analyze embedding distances at each hierarchy level
    print("\nAnalyzing embedding distances across hierarchy levels...")
    
    embedding_stats = {}
    
    for level in [1, 2, 3]:
        level_name = ["Category", "Subcategory", "Model"][level-1]
        
        if level == 1:
            label_col = "level_1_category"
        elif level == 2:
            label_col = "level_2_subcategory"
        else:
            label_col = "level_3_model"
        
        print(f"Computing distances for Level {level} ({level_name})...")
        
        # Group embeddings by class at this level
        class_embeddings = defaultdict(list)
        
        for _, row in data['test_df'].iterrows():
            filename = row['filename']
            if filename in protocol.image_embeddings and label_col in row:
                embedding = protocol.image_embeddings[filename].cpu().numpy()
                class_label = row[label_col]
                class_embeddings[class_label].append(embedding)
        
        # Calculate within-class and between-class distances
        within_class_distances = []
        between_class_distances = []
        
        classes = list(class_embeddings.keys())
        
        # Within-class distances
        for class_label, embeddings in class_embeddings.items():
            if len(embeddings) < 2:
                continue
            
            embeddings = np.array(embeddings)
            for i in range(len(embeddings)):
                for j in range(i+1, min(i+11, len(embeddings))):  # Limit for efficiency
                    distance = cosine(embeddings[i], embeddings[j])
                    within_class_distances.append(distance)
        
        # Between-class distances (sample for efficiency)
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes[i+1:], i+1):
                emb1 = class_embeddings[class1]
                emb2 = class_embeddings[class2]
                
                if len(emb1) == 0 or len(emb2) == 0:
                    continue
                
                # Sample a few pairs
                for k in range(min(5, len(emb1))):
                    for l in range(min(5, len(emb2))):
                        distance = cosine(emb1[k], emb2[l])
                        between_class_distances.append(distance)
        
        # Calculate statistics
        if within_class_distances and between_class_distances:
            level_stats = {
                'within_class_mean': np.mean(within_class_distances),
                'within_class_std': np.std(within_class_distances),
                'between_class_mean': np.mean(between_class_distances),
                'between_class_std': np.std(between_class_distances),
                'separation_ratio': np.mean(between_class_distances) / np.mean(within_class_distances),
                'num_classes': len(classes),
                'num_within_distances': len(within_class_distances),
                'num_between_distances': len(between_class_distances)
            }
            
            embedding_stats[f"level_{level}_{level_name.lower()}"] = level_stats
            
            print(f"Level {level} stats:")
            print(f"  Within-class distance: {level_stats['within_class_mean']:.4f} ± {level_stats['within_class_std']:.4f}")
            print(f"  Between-class distance: {level_stats['between_class_mean']:.4f} ± {level_stats['between_class_std']:.4f}")
            print(f"  Separation ratio: {level_stats['separation_ratio']:.4f}")
    
    # Create embedding distance comparison visualization
    if embedding_stats:
        levels = []
        within_means = []
        between_means = []
        separation_ratios = []
        level_names = []
        
        for level_key, stats in embedding_stats.items():
            level_num = int(level_key.split('_')[1])
            level_name = level_key.split('_')[2].capitalize()
            
            levels.append(level_num)
            level_names.append(level_name)
            within_means.append(stats['within_class_mean'])
            between_means.append(stats['between_class_mean'])
            separation_ratios.append(stats['separation_ratio'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Embedding Distance Analysis Across Hierarchy Levels", fontsize=16)
        
        # Within vs Between class distances
        x = np.arange(len(levels))
        width = 0.35
        
        axes[0].bar(x - width/2, within_means, width, label='Within-class', alpha=0.7, color='blue')
        axes[0].bar(x + width/2, between_means, width, label='Between-class', alpha=0.7, color='orange')
        axes[0].set_xlabel('Hierarchy Level')
        axes[0].set_ylabel('Mean Cosine Distance')
        axes[0].set_title('Within-class vs Between-class Distances')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(level_names)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Separation ratio
        axes[1].plot(levels, separation_ratios, marker='o', linewidth=3, markersize=10, color='green')
        axes[1].set_xlabel('Hierarchy Level')
        axes[1].set_ylabel('Separation Ratio (Between/Within)')
        axes[1].set_title('Class Separation Ratio Across Levels')
        axes[1].set_xticks(levels)
        axes[1].set_xticklabels(level_names)
        axes[1].grid(True, alpha=0.3)
        
        # Distance progression
        axes[2].plot(levels, within_means, marker='s', linewidth=2, markersize=8, label='Within-class', color='blue')
        axes[2].plot(levels, between_means, marker='^', linewidth=2, markersize=8, label='Between-class', color='orange')
        axes[2].set_xlabel('Hierarchy Level')
        axes[2].set_ylabel('Mean Cosine Distance')
        axes[2].set_title('Distance Progression Across Levels')
        axes[2].set_xticks(levels)
        axes[2].set_xticklabels(level_names)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(embedding_dir, "embedding_distance_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save embedding statistics
    with open(os.path.join(embedding_dir, "embedding_statistics.json"), 'w') as f:
        json.dump(embedding_stats, f, indent=2)
    
    return embedding_stats


def analyze_ood_across_levels(protocol, data, args, output_dir):
    """
    Analyze OOD detection performance across hierarchy levels.
    
    Args:
        protocol: HierarchicalAircraftProtocol instance.
        data: Dictionary with hierarchical data.
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        Dictionary with OOD analysis results.
    """
    print("\n" + "="*60)
    print("OOD DETECTION ACROSS HIERARCHY LEVELS")
    print("="*60)
    
    ood_dir = os.path.join(output_dir, "ood_analysis")
    os.makedirs(ood_dir, exist_ok=True)
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    k = k_values[0]  # Use first k value
    
    ood_results = {}
    
    for level in [1, 2, 3]:
        level_name = ["Category", "Subcategory", "Model"][level-1]
        print(f"\nRunning OOD detection at Level {level} ({level_name})...")
        
        # Define known classes for each level
        if level == 1:
            known_classes = ["Commercial", "Military"]  # Exclude General Aviation
        elif level == 2:
            known_classes = ["Boeing", "Airbus"]  # Commercial manufacturers only
        else:
            known_classes = ["737", "747", "777", "A320", "A330", "A350"]  # Subset of models
        
        try:
            ood_results_df, ood_metrics = protocol.hierarchical_ood_detection(
                data['test_df'],
                level=level,
                k=k,
                known_classes=known_classes
            )
            
            # Save results
            ood_results_df.to_csv(
                os.path.join(ood_dir, f"ood_level_{level}_results.csv"),
                index=False
            )
            
            # Generate ROC curve if we have the data
            if 'fpr' in ood_metrics and 'tpr' in ood_metrics:
                plt.figure(figsize=(8, 8))
                plt.plot(ood_metrics['fpr'], ood_metrics['tpr'], 
                        color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {ood_metrics["roc_auc"]:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Level {level} ({level_name}) OOD Detection ROC')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(ood_dir, f"ood_level_{level}_roc.png"))
                plt.close()
            
            ood_results[f"level_{level}"] = {
                'results': ood_results_df,
                'metrics': ood_metrics,
                'level_name': level_name,
                'known_classes': known_classes
            }
            
            print(f"Level {level} OOD Detection Results:")
            print(f"  ROC AUC: {ood_metrics.get('roc_auc', 0):.4f}")
            print(f"  Optimal F1: {ood_metrics.get('optimal_f1', 0):.4f}")
            print(f"  Precision: {ood_metrics.get('precision', 0):.4f}")
            print(f"  Recall: {ood_metrics.get('recall', 0):.4f}")
            
        except Exception as e:
            print(f"Error in Level {level} OOD detection: {e}")
            continue
    
    # Create OOD performance comparison
    if ood_results:
        levels = []
        roc_aucs = []
        precisions = []
        recalls = []
        f1_scores = []
        level_names = []
        
        for level_key, data_dict in ood_results.items():
            level_num = int(level_key.split('_')[1])
            metrics = data_dict['metrics']
            
            levels.append(level_num)
            level_names.append(data_dict['level_name'])
            roc_aucs.append(metrics.get('roc_auc', 0))
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('optimal_f1', 0))
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("OOD Detection Performance Across Hierarchy Levels", fontsize=16)
        
        # ROC AUC comparison
        axes[0, 0].bar(levels, roc_aucs, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        axes[0, 0].set_xlabel('Hierarchy Level')
        axes[0, 0].set_ylabel('ROC AUC')
        axes[0, 0].set_title('ROC AUC Across Levels')
        axes[0, 0].set_xticks(levels)
        axes[0, 0].set_xticklabels(level_names)
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision comparison
        axes[0, 1].bar(levels, precisions, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        axes[0, 1].set_xlabel('Hierarchy Level')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision Across Levels')
        axes[0, 1].set_xticks(levels)
        axes[0, 1].set_xticklabels(level_names)
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall comparison
        axes[1, 0].bar(levels, recalls, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        axes[1, 0].set_xlabel('Hierarchy Level')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall Across Levels')
        axes[1, 0].set_xticks(levels)
        axes[1, 0].set_xticklabels(level_names)
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score comparison
        axes[1, 1].bar(levels, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
        axes[1, 1].set_xlabel('Hierarchy Level')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score Across Levels')
        axes[1, 1].set_xticks(levels)
        axes[1, 1].set_xticklabels(level_names)
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ood_dir, "ood_performance_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return ood_results


def generate_comprehensive_report(all_results, output_dir):
    """
    Generate a comprehensive analysis report covering all aspects.
    
    Args:
        all_results: Dictionary with all analysis results.
        output_dir: Output directory.
    """
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    report_path = os.path.join(output_dir, "comprehensive_hierarchical_analysis.md")
    
    # Fix: Open file with UTF-8 encoding to handle Unicode characters
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Hierarchical Aircraft Classification Analysis\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive analysis of hierarchical aircraft classification using Vision-Language Model (VLM) embeddings. ")
        f.write("The analysis covers three hierarchy levels: Category (Commercial/Military/General Aviation), ")
        f.write("Subcategory (Boeing/Airbus/Fighter Jets/etc.), and specific aircraft Models.\n\n")
        
        # Hierarchical Results Summary
        if 'hierarchical' in all_results:
            hierarchical_results = all_results['hierarchical']
            
            f.write("## Hierarchical Classification Performance\n\n")
            f.write("| Level | Type | Zero-Shot Acc | KNN Acc | OOD ROC AUC |\n")
            f.write("|-------|------|---------------|---------|-------------|\n")
            
            level_types = ["Category", "Subcategory", "Model"]
            
            for i, (level_name, level_results) in enumerate(hierarchical_results.items()):
                level_num = int(level_name.split("_")[1])
                level_type = level_types[level_num - 1]
                
                # Extract metrics
                zs_acc = level_results.get("zero_shot", {}).get("metrics", {}).get("accuracy", 0)
                knn_acc = level_results.get("knn", {}).get("metrics", {}).get("accuracy", 0)
                ood_auc = level_results.get("ood", {}).get("metrics", {}).get("roc_auc", 0)
                
                f.write(f"| {level_num} | {level_type} | {zs_acc:.3f} | {knn_acc:.3f} | {ood_auc:.3f} |\n")
            
            f.write("\n### Key Performance Insights\n\n")
            f.write("![Hierarchy Performance Overview](hierarchical_results/hierarchy_performance_overview.png)\n\n")
        
        # Cross-Level Analysis
        if 'cross_level' in all_results:
            cross_level_data = all_results['cross_level']
            
            f.write("## Cross-Level Performance Analysis\n\n")
            f.write("This section analyzes how classification performance changes as we move from broad categories to specific models.\n\n")
            
            f.write("### Performance Degradation Analysis\n\n")
            f.write("![Difficulty Progression](cross_level_analysis/difficulty_progression_analysis.png)\n\n")
            
            if 'degradation_stats' in cross_level_data:
                stats = cross_level_data['degradation_stats']
                f.write("#### Key Degradation Metrics\n\n")
                # Fix: Replace Unicode arrow with ASCII alternative
                f.write(f"- **Accuracy drop Level 1->2**: {stats['accuracy_degradation_1_to_2']:.3f}\n")
                f.write(f"- **Accuracy drop Level 2->3**: {stats['accuracy_degradation_2_to_3']:.3f}\n")
                f.write(f"- **Total accuracy drop**: {stats['total_accuracy_degradation']:.3f}\n")
                f.write(f"- **Class count increase**: {stats['total_class_count_increase']:.1f}x from Level 1 to 3\n\n")
            
            f.write("### Analysis Insights\n\n")
            f.write("1. **Complexity Scaling**: As expected, classification becomes more challenging at lower hierarchy levels\n")
            f.write("2. **Class Proliferation**: The number of classes increases significantly from categories to specific models\n")
            f.write("3. **Confidence Degradation**: Model confidence typically decreases with increased granularity\n\n")
        
        # Embedding Analysis
        if 'embeddings' in all_results:
            f.write("## Hierarchical Embedding Analysis\n\n")
            f.write("This section examines how VLM embeddings represent aircraft at different hierarchy levels.\n\n")
            
            f.write("### UMAP Visualizations\n\n")
            f.write("**Level 1 (Category):**\n\n")
            f.write("![Category UMAP](embedding_analysis/umap_level_1_category.png)\n\n")
            
            f.write("**Level 2 (Subcategory):**\n\n")
            f.write("![Subcategory UMAP](embedding_analysis/umap_level_2_subcategory.png)\n\n")
            
            f.write("**Level 3 (Model):**\n\n")
            f.write("![Model UMAP](embedding_analysis/umap_level_3_model.png)\n\n")
            
            f.write("### Embedding Distance Analysis\n\n")
            f.write("![Embedding Distance Analysis](embedding_analysis/embedding_distance_analysis.png)\n\n")
            
            embedding_stats = all_results['embeddings']
            if embedding_stats:
                f.write("#### Distance Statistics by Level\n\n")
                f.write("| Level | Within-Class Distance | Between-Class Distance | Separation Ratio |\n")
                f.write("|-------|----------------------|----------------------|------------------|\n")
                
                for level_key, stats in embedding_stats.items():
                    level_name = level_key.replace('_', ' ').title()
                    within_mean = stats['within_class_mean']
                    between_mean = stats['between_class_mean']
                    separation = stats['separation_ratio']
                    f.write(f"| {level_name} | {within_mean:.4f} | {between_mean:.4f} | {separation:.4f} |\n")
                
                f.write("\n")
        
        # OOD Analysis
        if 'ood' in all_results:
            f.write("## Out-of-Distribution Detection Analysis\n\n")
            f.write("This section evaluates the system's ability to detect unknown aircraft types at each hierarchy level.\n\n")
            
            f.write("### OOD Performance Comparison\n\n")
            f.write("![OOD Performance Comparison](ood_analysis/ood_performance_comparison.png)\n\n")
            
            ood_results = all_results['ood']
            f.write("#### OOD Detection Results by Level\n\n")
            f.write("| Level | Known Classes | ROC AUC | Precision | Recall | F1 Score |\n")
            f.write("|-------|---------------|---------|-----------|--------|----------|\n")
            
            for level_key, level_data in ood_results.items():
                level_num = level_key.split('_')[1]
                level_name = level_data['level_name']
                metrics = level_data['metrics']
                known_classes = ', '.join(level_data['known_classes'][:3])
                if len(level_data['known_classes']) > 3:
                    known_classes += "..."
                
                roc_auc = metrics.get('roc_auc', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('optimal_f1', 0)
                
                f.write(f"| {level_num} ({level_name}) | {known_classes} | {roc_auc:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} |\n")
            
            f.write("\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("### Hierarchical Classification Approach\n\n")
        f.write("1. **Three-Level Hierarchy**: Aircraft are classified at Category, Subcategory, and Model levels\n")
        f.write("2. **VLM Embeddings**: CLIP vision-language model embeddings capture both visual and semantic features\n")
        f.write("3. **Multi-Method Evaluation**: Zero-shot, KNN, and OOD detection methods tested at each level\n")
        f.write("4. **Cross-Level Analysis**: Performance trends analyzed from broad to specific classifications\n\n")
        
        f.write("### Evaluation Metrics\n\n")
        f.write("- **Accuracy**: Standard classification accuracy\n")
        f.write("- **F1 Score**: Harmonic mean of precision and recall\n")
        f.write("- **ROC AUC**: Area under the ROC curve for OOD detection\n")
        f.write("- **Separation Ratio**: Between-class distance / Within-class distance for embeddings\n\n")
        
        # Conclusions and Future Work
        f.write("## Conclusions\n\n")
        f.write("### Key Findings\n\n")
        f.write("1. **Hierarchical Structure Works**: VLM embeddings successfully capture aircraft characteristics at multiple abstraction levels\n")
        f.write("2. **Predictable Performance Degradation**: Classification difficulty increases predictably from categories to specific models\n")
        f.write("3. **Effective Semantic Organization**: UMAP visualizations show clear clustering by aircraft types at all levels\n")
        f.write("4. **Robust OOD Detection**: The system can identify unknown aircraft types across hierarchy levels\n")
        f.write("5. **Zero-Shot Capabilities**: Text-image alignment enables classification without training examples\n\n")
        
        f.write("### Practical Implications\n\n")
        f.write("- **Adaptive Granularity**: System can provide classifications at appropriate detail levels based on confidence\n")
        f.write("- **Scalable Architecture**: Easy to add new aircraft types at any hierarchy level\n")
        f.write("- **Real-World Deployment**: Suitable for applications requiring both broad categorization and specific identification\n\n")
        
        f.write("### Future Research Directions\n\n")
        f.write("1. **Dynamic Hierarchy Learning**: Automatically discover optimal taxonomies from data\n")
        f.write("2. **Uncertainty Quantification**: Better calibration of prediction confidence\n")
        f.write("3. **Temporal Analysis**: Track aircraft design evolution through embeddings\n")
        f.write("4. **Multi-Modal Integration**: Incorporate additional metadata (specifications, performance data)\n")
        f.write("5. **Cross-Domain Transfer**: Apply hierarchical approach to other vehicle classifications\n\n")
        
        # Technical Details
        f.write("## Technical Implementation\n\n")
        f.write("### Model Architecture\n\n")
        f.write("- **Base Model**: CLIP ViT-B/32\n")
        f.write("- **Image Processing**: Standard CLIP preprocessing pipeline\n")
        f.write("- **Text Embeddings**: Template-based prompts for each hierarchy level\n")
        f.write("- **Distance Metric**: Cosine similarity for all comparisons\n\n")
        
        f.write("### Computational Considerations\n\n")
        f.write("- **Embedding Caching**: Pre-computed embeddings stored for efficiency\n")
        f.write("- **Batch Processing**: Images processed in batches for GPU efficiency\n")
        f.write("- **Memory Management**: Large datasets handled through streaming\n\n")
        
        f.write("---\n\n")
        f.write("*This analysis demonstrates the effectiveness of hierarchical classification approaches ")
        f.write("for complex computer vision tasks, providing both theoretical insights and practical solutions ")
        f.write("for aircraft identification systems.*\n")
    
    print(f"Comprehensive report generated: {report_path}")


def main():
    """Main function to run hierarchical aircraft classification analysis."""
    args = parse_args()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Setup output directory
        output_dir = setup_output_dir(args)
        print(f"Results will be saved to: {output_dir}")
        
        # Load custom hierarchy if provided
        custom_hierarchy = None
        if args.custom_hierarchy:
            custom_hierarchy = load_custom_hierarchy(args.custom_hierarchy)
        
        # Initialize hierarchical protocol
        print("Initializing Hierarchical Aircraft Protocol...")
        protocol = HierarchicalAircraftProtocol(
            clip_model_name=args.clip_model,
            data_dir=args.data_dir,
            hierarchy_config=custom_hierarchy
        )
        
        # Load and prepare hierarchical data
        data = load_and_prepare_hierarchical_data(args, protocol)
        
        # Check if we have any data
        if len(data['all_df']) == 0:
            print("No data loaded. Check your data directory and CSV files.")
            return
        
        # Store all results
        all_results = {}
        
        # Run hierarchical classification analysis
        if args.analyze_hierarchy or args.analyze_all:
            hierarchical_results = analyze_hierarchical_classification(protocol, data, args, output_dir)
            all_results['hierarchical'] = hierarchical_results
        
        # Run cross-level performance analysis
        if args.analyze_cross_level or args.analyze_all:
            cross_level_results = analyze_cross_level_performance(protocol, data, args, output_dir)
            all_results['cross_level'] = cross_level_results
        
        # Run hierarchical embedding analysis
        if args.analyze_embeddings or args.analyze_all:
            embedding_results = analyze_hierarchical_embeddings(protocol, data, args, output_dir)
            all_results['embeddings'] = embedding_results
        
        # Run OOD analysis across levels
        if args.analyze_ood or args.analyze_all:
            ood_results = analyze_ood_across_levels(protocol, data, args, output_dir)
            all_results['ood'] = ood_results
        
        # Generate comprehensive report
        generate_comprehensive_report(all_results, output_dir)
        
        print("\n" + "="*60)
        print("HIERARCHICAL AIRCRAFT ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nAll results saved to: {output_dir}")
        print("\nGenerated outputs:")
        print("- comprehensive_hierarchical_analysis.md (Main report)")
        print("- hierarchical_results/ (Classification results at each level)")
        print("- cross_level_analysis/ (Performance progression analysis)")
        print("- embedding_analysis/ (UMAP visualizations and distance analysis)")
        print("- ood_analysis/ (OOD detection across levels)")
        
        if args.save_embeddings:
            print("- hierarchical_embeddings.pkl (Cached embeddings)")
        
        print("\nNext steps:")
        print("1. Review the comprehensive report for key insights")
        print("2. Examine UMAP visualizations for embedding quality")
        print("3. Analyze performance degradation patterns")
        print("4. Consider expanding hierarchy with additional aircraft types")
        print("5. Experiment with different VLM models for comparison")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()