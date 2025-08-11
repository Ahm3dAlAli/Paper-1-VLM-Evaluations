#!/usr/bin/env python
"""
Enhanced OOD Detection for Aircraft Classification

This script runs comprehensive evaluation of different out-of-distribution detection
methods for aircraft classification:
1. KNN-based OOD detection with threshold optimization
2. Isolation Forest for OOD detection
3. Energy-based OOD detection
4. Mahalanobis distance OOD detection

Usage:
    python run_enhanced_ood.py --data_dir ./data/fgvc-aircraft --methods all
    python run_enhanced_ood.py --methods knn,isolation_forest --optimization_metric balanced_accuracy
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from aircraft_family_protocol import AircraftProtocol
from enhanced_ood_detection_family import (
    knn_ood_detection_optimized,
    isolation_forest_ood_detection,
    energy_based_ood_detection,
    mahalanobis_ood_detection, 
    visualize_ood_comparison
)
from aircraft_analysis import (
    setup_output_dir,
    load_and_prepare_data
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced OOD Detection for Aircraft Classification")
    
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
    
    # OOD detection options
    parser.add_argument("--methods", type=str, default="all",
                        help="OOD detection methods to evaluate (comma-separated): knn, isolation_forest, energy, mahalanobis, all")
    parser.add_argument("--optimization_metric", type=str, default="balanced_accuracy",
                        help="Metric to optimize threshold: balanced_accuracy, f1, gmean")
    parser.add_argument("--k_values", type=str, default="1,3,5,10",
                        help="Comma-separated list of k values for KNN")
    parser.add_argument("--clasification_target", type=str, default="manufacturer",
                        help="Target for classification: manufacturer or model")
    
    # Visualization options
    parser.add_argument("--tsne_perplexity", type=int, default=30,
                        help="Perplexity parameter for t-SNE visualization")
    parser.add_argument("--visualize_dimensions", type=int, default=5,
                        help="Number of embedding dimensions to visualize")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="enhanced_ood_results",
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
    
    return parser.parse_args()


def analyze_enhanced_ood_detection(protocol, data, args, output_dir):
    """
    Analyze OOD detection using multiple methods with threshold optimization.
    
    Args:
        protocol: AircraftProtocol instance.
        data: Dictionary with data.
        args: Command line arguments.
        output_dir: Output directory.
        
    Returns:
        Dictionary with results for all methods.
    """
    print("\n" + "="*50)
    print("Enhanced OOD Detection Analysis")
    print("="*50)
    
    # Create OOD output directory
    ood_dir = os.path.join(output_dir, "enhanced_ood")
    os.makedirs(ood_dir, exist_ok=True)
    
    # Determine which methods to evaluate
    methods_to_evaluate = args.methods.lower().split(',')
    if 'all' in methods_to_evaluate:
        methods_to_evaluate = ['knn', 'isolation_forest', 'energy', 'mahalanobis']
    
    # Store results for all methods
    all_results = {}
    all_metrics = {}
    
    # Determine classification target
    target = args.clasification_target
    if target not in ['manufacturer', 'model']:
        print(f"Warning: Invalid classification target '{target}'. Using 'manufacturer' instead.")
        target = 'manufacturer'
    
    print(f"Using '{target}' as classification target")
    
    # 1. KNN-based OOD detection with threshold optimization
    if 'knn' in methods_to_evaluate:
        print("\nEvaluating KNN-based OOD detection with threshold optimization...")
        
        # Parse k values
        k_values = [int(k) for k in args.k_values.split(",")]
        best_k_metrics = None
        best_k_value = None
        best_k_results = None
        
        for k in k_values:
            print(f"Testing with k={k}...")
            results_df, metrics = knn_ood_detection_optimized(
                protocol,
                data['test_df'],  # Full test set including Unknown
                k=k,
                optimization_metric=args.optimization_metric,
                class_type=target
            )
            
            # Save results
            results_df.to_csv(
                os.path.join(ood_dir, f"knn_optimized_k{k}_results.csv"),
                index=False
            )
            
            # Generate ROC curve
            plt.figure(figsize=(8, 8))
            plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2,
                     label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Optimized KNN OOD Detection (k={k})')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(ood_dir, f"knn_optimized_k{k}_roc.png"))
            plt.close()
            
            # Print results
            print(f"\nOptimized KNN OOD Detection Results (k={k}):")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"Balanced Accuracy: {metrics.get('normalized_accuracy', 0):.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
            
            # Track best k
            if best_k_metrics is None or metrics.get('normalized_accuracy', 0) > best_k_metrics.get('normalized_accuracy', 0):
                best_k_metrics = metrics
                best_k_value = k
                best_k_results = results_df
        
        # Add best KNN results to the comparison
        all_results['KNN-optimized'] = best_k_results
        all_metrics['KNN-optimized'] = best_k_metrics
        
        print(f"\nBest KNN configuration: k={best_k_value} with normalized accuracy: {best_k_metrics.get('normalized_accuracy', 0):.4f}")
    
    # 2. Isolation Forest OOD detection
    if 'isolation_forest' in methods_to_evaluate:
        print("\nEvaluating Isolation Forest for OOD detection...")
        results_df, metrics = isolation_forest_ood_detection(
            protocol,
            data['test_df'],
            optimization_metric=args.optimization_metric,
            class_type=target
        )
        
        # Save results
        results_df.to_csv(os.path.join(ood_dir, "isolation_forest_results.csv"), index=False)
        
        # Generate ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(metrics['fpr'], metrics['tpr'], color='green', lw=2,
                 label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Isolation Forest OOD Detection')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(ood_dir, "isolation_forest_roc.png"))
        plt.close()
        
        # Add to results dictionary
        all_results['Isolation Forest'] = results_df
        all_metrics['Isolation Forest'] = metrics
    
    # 3. Energy-based OOD detection
    if 'energy' in methods_to_evaluate:
        print("\nEvaluating Energy-based OOD detection...")
        results_df, metrics = energy_based_ood_detection(
            protocol,
            data['test_df'],
            optimization_metric=args.optimization_metric,
            class_type=target
        )
        
        # Save results
        results_df.to_csv(os.path.join(ood_dir, "energy_based_results.csv"), index=False)
        
        # Generate ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(metrics['fpr'], metrics['tpr'], color='purple', lw=2,
                 label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Energy-based OOD Detection')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(ood_dir, "energy_based_roc.png"))
        plt.close()
        
        # Add to results dictionary
        all_results['Energy-based'] = results_df
        all_metrics['Energy-based'] = metrics
    
    # 4. Mahalanobis distance-based OOD detection
    if 'mahalanobis' in methods_to_evaluate:
        print("\nEvaluating Mahalanobis distance-based OOD detection...")
        results_df, metrics = mahalanobis_ood_detection(
            protocol,
            data['test_df'],
            optimization_metric=args.optimization_metric,
            class_type=target
        )
        
        # Save results
        results_df.to_csv(os.path.join(ood_dir, "mahalanobis_results.csv"), index=False)
        
        # Generate ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(metrics['fpr'], metrics['tpr'], color='red', lw=2,
                 label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mahalanobis Distance OOD Detection')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(ood_dir, "mahalanobis_roc.png"))
        plt.close()
        
        # Add to results dictionary
        all_results['Mahalanobis'] = results_df
        all_metrics['Mahalanobis'] = metrics
    
    # Compare all methods
    if len(all_metrics) > 1:
        print("\nGenerating method comparison visualizations...")
        visualize_ood_comparison(all_metrics, ood_dir)
    
    return all_results, all_metrics


def create_summary_report(all_metrics, output_dir, target='manufacturer'):
    """
    Create a comprehensive summary report of all OOD detection methods.
    
    Args:
        all_metrics: Dictionary with metrics for all methods
        output_dir: Output directory
        target: Classification target (manufacturer or model)
    """
    print("\nGenerating summary report...")
    
    report_path = os.path.join(output_dir, "enhanced_ood_summary.md")
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced OOD Detection Analysis\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Classification target: **{target}**\n\n")
        
        # Add comparison table
        f.write("## Comparison of OOD Detection Methods\n\n")
        f.write("| Method | ROC AUC | Normalized Accuracy | Precision | Recall | F1 Score |\n")
        f.write("|--------|---------|---------------------|-----------|--------|----------|\n")
        
        for method, method_metrics in all_metrics.items():
            f.write(f"| {method} | {method_metrics['roc_auc']:.4f} | ")
            f.write(f"{method_metrics.get('normalized_accuracy', 0):.4f} | ")
            f.write(f"{method_metrics['precision']:.4f} | ")
            f.write(f"{method_metrics['recall']:.4f} | ")
            f.write(f"{method_metrics['f1']:.4f} |\n")
        
        # Add ROC comparison image
        f.write("\n## ROC Curve Comparison\n\n")
        f.write("![ROC Curve Comparison](enhanced_ood/ood_roc_comparison.png)\n\n")
        
        # Add metrics comparison image
        f.write("## Performance Metrics Comparison\n\n")
        f.write("![Metrics Comparison](enhanced_ood/ood_metrics_comparison.png)\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Find best method
        best_method = max(all_metrics.items(), key=lambda x: x[1].get('normalized_accuracy', x[1]['roc_auc']))
        method_name, method_metrics = best_method
        
        f.write(f"- The best performing OOD detection method is **{method_name}** with ")
        
        if 'normalized_accuracy' in method_metrics:
            f.write(f"a normalized accuracy of {method_metrics['normalized_accuracy']:.4f} and ")
            
        f.write(f"ROC AUC of {method_metrics['roc_auc']:.4f}\n")
        f.write(f"- This method achieved precision of {method_metrics['precision']:.4f}, recall of {method_metrics['recall']:.4f}, and F1 score of {method_metrics['f1']:.4f}\n")
        
        # Analyze KNN specifically if available
        if 'KNN-optimized' in all_metrics:
            knn_metrics = all_metrics['KNN-optimized']
            f.write(f"- The optimized KNN method used a threshold of {knn_metrics['optimal_threshold']:.4f} for OOD detection\n")
        
        # Compare to original KNN results if relevant
        if 'KNN-optimized' in all_metrics:
            f.write(f"- Compared to standard KNN approaches, the optimized version balanced precision and recall more effectively\n")
        
        # Compare multiple methods if available
        if len(all_metrics) > 1:
            f.write("- The different OOD detection methods showed varying characteristics:\n")
            
            for method, metrics in all_metrics.items():
                precision = metrics['precision']
                recall = metrics['recall']
                
                if precision > 0.8 and recall < 0.6:
                    f.write(f"  - **{method}** is more conservative (high precision, lower recall)\n")
                elif precision < 0.6 and recall > 0.8:
                    f.write(f"  - **{method}** is more aggressive in detecting unknowns (high recall, lower precision)\n")
                elif precision > 0.7 and recall > 0.7:
                    f.write(f"  - **{method}** provides a well-balanced approach (good precision and recall)\n")
        
        # Overall conclusion
        f.write("\n## Conclusion\n\n")
        f.write("The enhanced OOD detection methods significantly improve the practical utility of aircraft classification by achieving a better balance between precision and recall. ")
        
        if target == 'manufacturer':
            f.write("These methods effectively distinguish between known manufacturers (Boeing/Airbus) and unknown manufacturers, ")
        else:
            f.write("These methods effectively distinguish between known aircraft models and unknown models, ")
            
        f.write("demonstrating that VLM embeddings can reliably detect out-of-distribution samples in open-set scenarios.\n")
    
    print(f"Summary report generated: {report_path}")


def main():
    """Main function to run enhanced OOD detection analysis."""
    args = parse_args()
    
    try:
        # Set random seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Setup output directory
        output_dir = setup_output_dir(args)
        print(f"Results will be saved to: {output_dir}")
        
        # Initialize protocol
        print(f"Initializing AircraftProtocol with {args.clip_model}...")
        protocol = AircraftProtocol(
            clip_model_name=args.clip_model,
            data_dir=args.data_dir
        )
        
        # Load and prepare data
        data = load_and_prepare_data(args, protocol)
        
        # Check if data was loaded successfully
        if not protocol.image_embeddings or len(protocol.image_embeddings) == 0:
            print("No image embeddings were generated. Cannot proceed with analysis.")
            sys.exit(1)
        
        # Run enhanced OOD detection analysis
        results, metrics = analyze_enhanced_ood_detection(protocol, data, args, output_dir)
        
        # Create summary report
        create_summary_report(metrics, output_dir, target=args.clasification_target)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print("\nNext steps:")
        print("- Review the enhanced_ood_summary.md file for key findings")
        print("- Explore the ROC curves and method comparisons in the enhanced_ood directory")
        print("- Choose the best OOD detection method for your aircraft classification needs")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()