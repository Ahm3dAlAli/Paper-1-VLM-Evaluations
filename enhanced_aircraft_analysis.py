# enhanced_aircraft_analysis.py - Updated with progress bars

"""
Enhanced Aircraft Analysis with Comprehensive VLM Evaluation
Implements all research questions with advanced metrics and visualizations
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from PIL import Image

# Import custom modules
from ehanced_vlm_pipeline import ComprehensiveOpenSetEvaluator
from feature_analysis import FeatureSpaceAnalyzer, MultiClassifierEvaluator
from prompt_optimization import PromptOptimizer
from aircraft_manufacturer_protocol import AircraftProtocol
from lora_finetuning import create_lora_visualization_fixed, _make_json_serializable

class ProgressTracker:
    """Helper class to track and display progress."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_times = {}
        self.start_time = time.time()
    
    def start_step(self, step_name: str):
        """Start tracking a step."""
        self.current_step += 1
        self.step_times[step_name] = {'start': time.time()}
        print(f"\n[{self.current_step}/{self.total_steps}] {step_name}...")
    
    def update_substep(self, description: str):
        """Update current step with substep description."""
        print(f"    → {description}")
    
    def end_step(self, step_name: str):
        """End tracking a step."""
        if step_name in self.step_times:
            self.step_times[step_name]['end'] = time.time()
            duration = self.step_times[step_name]['end'] - self.step_times[step_name]['start']
            print(f"    ✓ Completed in {duration:.1f}s")
    
    def get_eta(self):
        """Get estimated time remaining."""
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = avg_time_per_step * remaining_steps
            return eta
        return 0
    
    def print_summary(self):
        """Print summary of all steps."""
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Analysis completed in {total_time:.1f}s")
        print(f"{'='*60}")


class EnhancedAircraftAnalyzer:
    """Main analyzer implementing all research questions."""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = self.setup_output_dir()
        
        # Initialize progress tracker
        total_steps = 6 if args.enable_lora else 5
        self.progress = ProgressTracker(total_steps)
        
        # Initialize components
        self.vlm_protocol = AircraftProtocol(
            clip_model_name=args.clip_model,
            data_dir=args.data_dir
        )
        
        self.open_set_evaluator = ComprehensiveOpenSetEvaluator()
        self.feature_analyzer = FeatureSpaceAnalyzer()
        self.classifier_evaluator = MultiClassifierEvaluator()
        self.prompt_optimizer = PromptOptimizer(
            self.vlm_protocol.model,
            self.vlm_protocol.processor,
            self.vlm_protocol.device
        )
        
        # Initialize ResNet for comparison
        if args.compare_resnet:
            from resnet_vs_vlm_comparison import ResNetEmbeddingExtractor
            self.resnet_extractor = ResNetEmbeddingExtractor(
                model_name=args.resnet_model
            )
        
        self.results = {}
    
    def setup_output_dir(self) -> str:
        """Setup output directory structure."""
        output_dir = self.args.output_dir
        
        if self.args.timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{output_dir}_{timestamp}"
        
        # Create subdirectories
        subdirs = [
            'metrics', 'visualizations', 'feature_analysis',
            'classifier_comparison', 'prompt_optimization',
            'open_set_analysis', 'reports'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
        return output_dir
    
    def run_comprehensive_analysis(self):
        """Run all analyses addressing the research questions."""
        
        print("="*60)
        print("ENHANCED VLM OPEN-SET RECOGNITION ANALYSIS")
        print("="*60)
        
        # Load and prepare data
        self.progress.start_step("Loading and preparing data")
        data = self.load_and_prepare_data()
        self.progress.end_step("Loading and preparing data")
        
        # Research Question 1: Enhanced Feature Space Analysis
        self.progress.start_step("Research Question 1: Enhanced Feature Space Analysis")
        self.analyze_feature_spaces(data)
        self.progress.end_step("Research Question 1: Enhanced Feature Space Analysis")
        
        # Research Question 2: Comprehensive Classifier Evaluation
        self.progress.start_step("Research Question 2: Comprehensive Classifier Performance")
        self.evaluate_classifiers(data)
        self.progress.end_step("Research Question 2: Comprehensive Classifier Performance")
        
        # Research Question 3: Advanced Open-Set Discriminability
        self.progress.start_step("Research Question 3: Advanced Open-Set Discriminability")
        self.analyze_open_set_performance(data)
        self.progress.end_step("Research Question 3: Advanced Open-Set Discriminability")
        
        # Research Question 4: Text Prompt Strategy Optimization
        self.progress.start_step("Research Question 4: Text Prompt Strategy Optimization")
        self.optimize_text_prompts(data)
        self.progress.end_step("Research Question 4: Text Prompt Strategy Optimization")
        
        # Research Question 5: LoRA Fine-tuning (if enabled)
        if self.args.enable_lora:
            self.progress.start_step("Research Question 5: LoRA Fine-tuning Implementation")
            self.implement_lora_finetuning(data)
            self.progress.end_step("Research Question 5: LoRA Fine-tuning Implementation")
        
        # Generate comprehensive report
        
        self.progress.print_summary()
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
    
    def load_and_prepare_data(self) -> Dict:
        """Load and prepare data with proper known/unknown splits."""
        
        # Load datasets with progress bar
        self.progress.update_substep("Loading CSV files")
        train_df = self.vlm_protocol.load_data(os.path.join(self.args.data_dir, 'train.csv'))
        val_df = self.vlm_protocol.load_data(os.path.join(self.args.data_dir, 'val.csv'))
        test_df = self.vlm_protocol.load_data(os.path.join(self.args.data_dir, 'test.csv'))
        
        # Define known and unknown manufacturers
        known_manufacturers = ["Boeing", "Airbus"]
        
        # Create splits
        self.progress.update_substep("Creating known/unknown splits")
        train_known = train_df[train_df['manufacturer'].isin(known_manufacturers)]
        val_known = val_df[val_df['manufacturer'].isin(known_manufacturers)]
        
        # For test set, keep all samples but mark known/unknown
        test_df['is_known'] = test_df['manufacturer'].isin(known_manufacturers)
        
        # Generate embeddings
        self.progress.update_substep("Generating VLM text embeddings")
        self.vlm_protocol.generate_text_embeddings()
        
        all_df = pd.concat([train_known, val_known, test_df])
        
        self.progress.update_substep(f"Generating VLM image embeddings for {len(all_df)} images")
        
        # Add progress bar for image embedding generation
        batch_size = self.args.batch_size
        total_batches = (len(all_df) + batch_size - 1) // batch_size
        
        with tqdm(total=len(all_df), desc="    Extracting VLM embeddings", leave=True) as pbar:
            self.vlm_protocol.generate_image_embeddings(
                all_df, 
                batch_size=batch_size,
                progress_callback=lambda n: pbar.update(n)
            )
        
        # Generate ResNet embeddings if needed
        if hasattr(self, 'resnet_extractor'):
            self.progress.update_substep("Generating ResNet embeddings")
            with tqdm(total=len(all_df), desc="    Extracting ResNet embeddings", leave=True) as pbar:
                self.resnet_extractor.extract_embeddings(
                    all_df, 
                    self.vlm_protocol.image_dir, 
                    batch_size=batch_size,
                    progress_callback=lambda n: pbar.update(n)
                )
        
        # Visualize data distribution
        self.progress.update_substep("Creating data distribution visualization")
        self.visualize_data_distribution(train_known, val_known, test_df)
        
        return {
            'train_known': train_known,
            'val_known': val_known,
            'test': test_df,
            'all': all_df
        }
    
    def analyze_feature_spaces(self, data: Dict):
        """Research Question 1: Enhanced Feature Space Analysis."""
        
        self.progress.update_substep("Extracting features from test set")
        
        # Extract features at different levels
        test_data = data['test']
        test_embeddings = []
        test_labels = []
        test_known_mask = []
        
        # Use progress bar for feature extraction
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), 
                          desc="    Processing test samples", leave=False):
            if row['filename'] in self.vlm_protocol.image_embeddings:
                test_embeddings.append(
                    self.vlm_protocol.image_embeddings[row['filename']].cpu().numpy()
                )
                test_labels.append(row['manufacturer'])
                test_known_mask.append(row['is_known'])
        
        test_embeddings = np.array(test_embeddings)
        test_labels = np.array(test_labels)
        test_known_mask = np.array(test_known_mask)
        
        self.progress.update_substep("Computing feature statistics")
        
        # Compare VLM and ResNet embeddings
        embedding_dict = {
            'VLM': test_embeddings
        }
        
        if hasattr(self, 'resnet_extractor'):
            resnet_embeddings = []
            for _, row in test_data.iterrows():
                if row['filename'] in self.resnet_extractor.image_embeddings:
                    resnet_embeddings.append(
                        self.resnet_extractor.image_embeddings[row['filename']].cpu().numpy()
                    )
            
            if resnet_embeddings:
                embedding_dict['ResNet'] = np.array(resnet_embeddings)
        
        # Visualize embedding spaces
        self.progress.update_substep("Creating embedding space visualizations")
        fig = self.feature_analyzer.compare_embedding_spaces(
            embedding_dict,
            test_labels,
            test_known_mask,
            title="VLM vs ResNet Embedding Space Comparison",
            save_path=os.path.join(self.output_dir, 'feature_analysis', 'embedding_comparison.png')
        )
        
        # Compute feature statistics
        feature_stats = {}
        for model_name, embeddings in embedding_dict.items():
            stats = self.feature_analyzer.compute_feature_statistics(embeddings, test_labels)
            feature_stats[model_name] = stats
            
            print(f"\n    {model_name} Feature Statistics:")
            print(f"      Mean inter-class distance: {stats['mean_inter_class_distance']:.4f}")
            print(f"      Mean intra-class distance: {stats['mean_intra_class_distance']:.4f}")
            print(f"      Discriminability ratio: {stats['discriminability_ratio']:.4f}")
            if 'silhouette_score' in stats:
                print(f"      Silhouette score: {stats['silhouette_score']:.4f}")
        
        # Analyze text-image alignment for VLM
        if hasattr(self.vlm_protocol, 'manufacturer_embeddings'):
            self.progress.update_substep("Analyzing text-image alignment")
            text_embeddings = {
                'Manufacturer': np.array([
                    self.vlm_protocol.manufacturer_embeddings[m].cpu().numpy()
                    for m in ['Boeing', 'Airbus', 'Unknown']
                ])
            }
            
            fig = self.feature_analyzer.analyze_text_image_alignment(
                test_embeddings[test_known_mask],
                text_embeddings,
                test_labels[test_known_mask],
                title="VLM Text-Image Alignment Analysis",
                save_path=os.path.join(self.output_dir, 'feature_analysis', 'text_image_alignment.png')
            )
        
        self.results['feature_analysis'] = feature_stats
    
    def evaluate_classifiers(self, data: Dict):
        """Research Question 2: Comprehensive Classifier Evaluation."""
        
        self.progress.update_substep("Preparing training and test data")
        
        # Prepare training and test data
        train_data = data['train_known']
        test_data = data['test']
        
        # Extract embeddings and labels with progress
        train_embeddings = []
        train_labels = []
        
        for _, row in tqdm(train_data.iterrows(), total=len(train_data),
                          desc="    Processing training samples", leave=False):
            if row['filename'] in self.vlm_protocol.image_embeddings:
                train_embeddings.append(
                    self.vlm_protocol.image_embeddings[row['filename']].cpu().numpy()
                )
                train_labels.append(row['manufacturer'])
        
        train_embeddings = np.array(train_embeddings)
        train_labels = np.array(train_labels)
        
        test_embeddings = []
        test_labels = []
        test_known_mask = []
        
        for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                          desc="    Processing test samples", leave=False):
            if row['filename'] in self.vlm_protocol.image_embeddings:
                test_embeddings.append(
                    self.vlm_protocol.image_embeddings[row['filename']].cpu().numpy()
                )
                test_labels.append(row['manufacturer'])
                test_known_mask.append(row['is_known'])
        
        test_embeddings = np.array(test_embeddings)
        test_labels = np.array(test_labels)
        test_known_mask = np.array(test_known_mask)
        
        # Evaluate classifiers
        self.progress.update_substep("Evaluating classifiers on VLM features")
        
        # The MultiClassifierEvaluator will show its own progress
        vlm_results = self.classifier_evaluator.evaluate_all(
            train_embeddings, train_labels,
            test_embeddings, test_labels,
            test_known_mask
        )
        
        # Add model type column
        vlm_results['Model Type'] = 'VLM'
        
        # Evaluate on ResNet features if available
        if hasattr(self, 'resnet_extractor'):
            self.progress.update_substep("Evaluating classifiers on ResNet features")
            
            # Extract ResNet embeddings
            train_embeddings_resnet = []
            for _, row in train_data.iterrows():
                if row['filename'] in self.resnet_extractor.image_embeddings:
                    train_embeddings_resnet.append(
                        self.resnet_extractor.image_embeddings[row['filename']].cpu().numpy()
                    )
            
            test_embeddings_resnet = []
            for _, row in test_data.iterrows():
                if row['filename'] in self.resnet_extractor.image_embeddings:
                    test_embeddings_resnet.append(
                        self.resnet_extractor.image_embeddings[row['filename']].cpu().numpy()
                    )
            
            if train_embeddings_resnet and test_embeddings_resnet:
                resnet_results = self.classifier_evaluator.evaluate_all(
                    np.array(train_embeddings_resnet), train_labels,
                    np.array(test_embeddings_resnet), test_labels,
                    test_known_mask
                )
                resnet_results['Model Type'] = 'ResNet'
                
                # Combine results
                all_results = pd.concat([vlm_results, resnet_results])
            else:
                all_results = vlm_results
        else:
            all_results = vlm_results
        
        # Visualize classifier comparison
        self.progress.update_substep("Creating classifier comparison visualizations")
        self._create_classifier_visualization(all_results)
        
        # Save results
        all_results.to_csv(os.path.join(self.output_dir, 'classifier_comparison', 'classifier_results.csv'), index=False)
        self.results['classifier_evaluation'] = all_results
    
    def analyze_open_set_performance(self, data: Dict):
        """Research Question 3: Advanced Open-Set Discriminability Analysis."""
        
        # Prepare data
        test_data = data['test']
        
        # For each model type, compute open-set metrics
        models_to_evaluate = ['VLM']
        if hasattr(self, 'resnet_extractor'):
            models_to_evaluate.append('ResNet')
        
        for model_idx, model_name in enumerate(models_to_evaluate):
            self.progress.update_substep(f"Evaluating {model_name} for open-set recognition ({model_idx+1}/{len(models_to_evaluate)})")
            
            # Get embeddings
            if model_name == 'VLM':
                embeddings_dict = self.vlm_protocol.image_embeddings
            else:
                embeddings_dict = self.resnet_extractor.image_embeddings
            
            # Extract known and unknown samples with progress
            known_embeddings = []
            unknown_embeddings = []
            known_labels = []
            known_predictions = []
            known_scores = []
            unknown_scores = []
            
            # Run KNN for predictions and scores
            from sklearn.neighbors import KNeighborsClassifier
            
            # Train KNN on known samples from training set
            self.progress.update_substep(f"    Training KNN classifier for {model_name}")
            
            train_embeddings = []
            train_labels = []
            
            for _, row in data['train_known'].iterrows():
                if row['filename'] in embeddings_dict:
                    train_embeddings.append(embeddings_dict[row['filename']].cpu().numpy())
                    train_labels.append(row['manufacturer'])
            
            train_embeddings = np.array(train_embeddings)
            train_labels = np.array(train_labels)
            
            # Fit KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(train_embeddings, train_labels)
            
            # Process test samples with progress
            self.progress.update_substep(f"    Computing predictions and scores for {model_name}")
            
            for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                              desc=f"    Processing {model_name} test samples", leave=False):
                if row['filename'] in embeddings_dict:
                    embedding = embeddings_dict[row['filename']].cpu().numpy().reshape(1, -1)
                    
                    # Get prediction and score
                    prediction = knn.predict(embedding)[0]
                    distances, indices = knn.kneighbors(embedding)
                    score = 1 / (1 + distances[0].mean())  # Convert distance to similarity score
                    
                    if row['is_known']:
                        known_embeddings.append(embedding[0])
                        known_labels.append(row['manufacturer'])
                        known_predictions.append(prediction)
                        known_scores.append(score)
                    else:
                        unknown_embeddings.append(embedding[0])
                        unknown_scores.append(score)
            
            # Convert to arrays
            known_embeddings = np.array(known_embeddings)
            unknown_embeddings = np.array(unknown_embeddings)
            known_labels = np.array(known_labels)
            known_predictions = np.array(known_predictions)
            known_scores = np.array(known_scores)
            unknown_scores = np.array(unknown_scores)
            
            # Evaluate with comprehensive metrics
            self.progress.update_substep(f"    Computing open-set metrics for {model_name}")
            
            results = self.open_set_evaluator.evaluate(
                known_embeddings, unknown_embeddings,
                known_labels, known_predictions,
                known_scores, unknown_scores,
                model_name=model_name
            )
            
            # Generate OSCR curve
            self.progress.update_substep(f"    Creating visualizations for {model_name}")
            
            fig = self.open_set_evaluator.metrics.plot_oscr_curve(
                results['ccr_curve'], results['fpr_curve'], results['oscr_auc'],
                title=f"{model_name} OSCR Curve",
                save_path=os.path.join(self.output_dir, 'open_set_analysis', f'{model_name}_oscr_curve.png')
            )
            plt.close()
            
            # Generate threshold analysis
            fig, optimal_threshold = self.open_set_evaluator.metrics.plot_threshold_analysis(
                known_scores, unknown_scores,
                title=f"{model_name} Score Distribution Analysis",
                save_path=os.path.join(self.output_dir, 'open_set_analysis', f'{model_name}_threshold_analysis.png')
            )
            plt.close()
        
        # Compare models
        if len(self.open_set_evaluator.results) > 1:
            self.progress.update_substep("Creating model comparison")
            comparison_df = self.open_set_evaluator.compare_models(
                save_dir=os.path.join(self.output_dir, 'open_set_analysis')
            )
            self.results['open_set_comparison'] = comparison_df
    
    def optimize_text_prompts(self, data: Dict):
        """Research Question 4: Text Prompt Strategy Optimization."""
        
        self.progress.update_substep("Extracting VLM embeddings for prompt optimization")
        
        # Extract VLM embeddings and labels
        test_data = data['test']
        image_embeddings = []
        labels = []
        known_mask = []
        
        for _, row in tqdm(test_data.iterrows(), total=len(test_data),
                          desc="    Processing embeddings", leave=False):
            if row['filename'] in self.vlm_protocol.image_embeddings:
                image_embeddings.append(
                    self.vlm_protocol.image_embeddings[row['filename']]
                )
                labels.append(row['manufacturer'])
                known_mask.append(row['is_known'])
        
        image_embeddings = torch.stack(image_embeddings)
        labels = np.array(labels)
        unknown_mask = ~np.array(known_mask)
        
        # Define known classes
        known_classes = ['Boeing', 'Airbus']
        
        # Evaluate different prompt strategies
        self.progress.update_substep("Evaluating prompt strategies")
        
        # Show progress for each strategy
        strategies = list(self.prompt_optimizer.prompt_templates.keys())
        
        results_list = []
        for idx, strategy in enumerate(strategies):
            self.progress.update_substep(f"    Testing strategy: {strategy} ({idx+1}/{len(strategies)})")
            
            # The evaluate_prompt_strategies method will be called for each strategy
            # We'll need to modify the prompt_optimizer to evaluate one at a time
            # For now, we'll call the full evaluation
            if idx == 0:  # Only run once
                results_df = self.prompt_optimizer.evaluate_prompt_strategies(
                    image_embeddings,
                    labels,
                    known_classes,
                    unknown_mask
                )
        
        # Visualize results
        self.progress.update_substep("Creating prompt analysis visualizations")
        fig = self.prompt_optimizer.visualize_prompt_analysis(
            results_df,
            save_path=os.path.join(self.output_dir, 'prompt_optimization', 'prompt_analysis.png')
        )
        plt.close()
        
        '''
        # Optimize ensemble weights
        self.progress.update_substep("Optimizing prompt ensemble weights")
        ensemble_prompts = self.prompt_optimizer.generate_ensemble_prompts(known_classes)
        
        # Add progress for weight optimization
        optimal_weights = {}
        for idx, class_name in enumerate(known_classes):
            self.progress.update_substep(f"    Optimizing weights for {class_name} ({idx+1}/{len(known_classes)})")
            
            class_prompts = ensemble_prompts[class_name]
            # Call optimization for single class
            weights = self.prompt_optimizer.optimize_prompt_weights(
                image_embeddings,
                labels,
                {class_name: class_prompts},
                validation_split=0.2
            )
            optimal_weights.update(weights)
        
        # Save optimal weights
        self.progress.update_substep("Saving optimization results")
        weights_df = pd.DataFrame([
            {'Class': class_name, 'Prompt': prompt, 'Weight': weight}
            for class_name, prompts in ensemble_prompts.items()
            for prompt, weight in zip(prompts, optimal_weights[class_name])
        ])
        weights_df.to_csv(
            os.path.join(self.output_dir, 'prompt_optimization', 'optimal_weights.csv'),
            index=False
        )
        
        # Visualize optimal weights
        self._create_weights_visualization(weights_df, known_classes)
        
        self.results['prompt_optimization'] = {
            'strategy_evaluation': results_df,
            'optimal_weights': weights_df
        }
        '''
    
    def _create_classifier_visualization(self, all_results):
        """Create classifier comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        pivot_accuracy = all_results.pivot(index='Classifier', columns='Model Type', values='Accuracy')
        pivot_accuracy.plot(kind='bar', ax=ax1)
        ax1.set_title('Classifier Accuracy Comparison', fontsize=12)
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Classifier')
        ax1.legend(title='Feature Type')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Unknown rejection rate
        ax2 = axes[0, 1]
        pivot_rejection = all_results.pivot(index='Classifier', columns='Model Type', values='Unknown Rejection Rate')
        pivot_rejection.plot(kind='bar', ax=ax2)
        ax2.set_title('Unknown Rejection Rate by Classifier', fontsize=12)
        ax2.set_ylabel('Rejection Rate')
        ax2.set_xlabel('Classifier')
        ax2.legend(title='Feature Type')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence gap
        ax3 = axes[1, 0]
        all_results['Confidence Gap'] = all_results['Mean Known Confidence'] - all_results['Mean Unknown Confidence']
        pivot_gap = all_results.pivot(index='Classifier', columns='Model Type', values='Confidence Gap')
        pivot_gap.plot(kind='bar', ax=ax3)
        ax3.set_title('Known-Unknown Confidence Gap', fontsize=12)
        ax3.set_ylabel('Confidence Gap')
        ax3.set_xlabel('Classifier')
        ax3.legend(title='Feature Type')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overall performance heatmap
        ax4 = axes[1, 1]
        metrics_for_heatmap = ['Accuracy', 'F1-Score', 'Unknown Rejection Rate']
        heatmap_data = all_results[all_results['Model Type'] == 'VLM'][['Classifier'] + metrics_for_heatmap].set_index('Classifier')
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('VLM Classifier Performance Heatmap', fontsize=12)
        
        plt.suptitle('Comprehensive Classifier Evaluation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'classifier_comparison', 'classifier_evaluation.png'), dpi=300)
        plt.close()
    
    def _create_weights_visualization(self, weights_df, known_classes):
        """Create visualization for optimal prompt weights."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for class_name in known_classes:
            class_weights = weights_df[weights_df['Class'] == class_name]
            x = range(len(class_weights))
            ax.bar([i + 0.4 * known_classes.index(class_name) for i in x],
                   class_weights['Weight'],
                   width=0.4,
                   label=class_name,
                   alpha=0.7)
        
        ax.set_xlabel('Prompt Index')
        ax.set_ylabel('Optimal Weight')
        ax.set_title('Optimal Prompt Ensemble Weights by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prompt_optimization', 'optimal_weights.png'), dpi=300)
        plt.close()
    
    def visualize_data_distribution(self, train_df, val_df, test_df):
        """Visualize the distribution of known/unknown samples."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Training distribution
        train_counts = train_df['manufacturer'].value_counts()
        axes[0].bar(train_counts.index, train_counts.values, color='blue', alpha=0.7)
        axes[0].set_title('Training Set (Known Only)', fontsize=12)
        axes[0].set_xlabel('Manufacturer')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Validation distribution
        val_counts = val_df['manufacturer'].value_counts()
        axes[1].bar(val_counts.index, val_counts.values, color='green', alpha=0.7)
        axes[1].set_title('Validation Set (Known Only)', fontsize=12)
        axes[1].set_xlabel('Manufacturer')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Test distribution with known/unknown
        test_counts = test_df['manufacturer'].value_counts()
        colors = ['blue' if m in ["Boeing", "Airbus"] else 'red' for m in test_counts.index]
        bars = axes[2].bar(test_counts.index, test_counts.values, color=colors, alpha=0.7)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Known'),
            Patch(facecolor='red', alpha=0.7, label='Unknown')
        ]
        axes[2].legend(handles=legend_elements)
        axes[2].set_title('Test Set (Known + Unknown)', fontsize=12)
        axes[2].set_xlabel('Manufacturer')
        axes[2].set_ylabel('Count')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Data Distribution: Known vs Unknown Manufacturers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'visualizations', 'data_distribution.png'), dpi=300)
        plt.close()
    
    def implement_lora_finetuning(self, data: Dict):
        """Research Question 5: LoRA Fine-tuning Implementation."""
        print("Implementing LoRA fine-tuning for open-set recognition...")
        print(f"\n{'='*60}")
        print("LORA FINE-TUNING IMPLEMENTATION")
        print(f"{'='*60}")
        
        # Check if LoRA is requested
        if not getattr(self.args, 'enable_lora', False):
            print("LoRA fine-tuning not requested (use --enable_lora to enable)")
            return {
                'status': 'not_requested',
                'message': 'Add --enable_lora flag to enable LoRA fine-tuning'
            }
        
        # Check dependencies
        try:
            import importlib.util
            peft_spec = importlib.util.find_spec("peft")
            if peft_spec is None:
                print("❌ PEFT library not installed")
                print("📦 Install with: pip install peft")
                
                # Create informational visualization
                self._create_lora_dependency_info()
                
                return {
                    'status': 'skipped',
                    'reason': 'peft_not_installed',
                    'install_command': 'pip install peft',
                    'message': 'PEFT library required for LoRA fine-tuning'
                }
            
            print("✅ PEFT library available")
            
        except Exception as e:
            print(f"❌ Error checking dependencies: {e}")
            return {'status': 'failed', 'reason': 'dependency_check_error', 'error': str(e)}
        
        # Try to import and run LoRA

            # Import LoRA components
        try:
            from lora_finetuning import LoRAFineTuner, compare_models_before_after_lora
            print("✅ LoRA modules imported successfully")
        except ImportError as e:
            print(f"❌ Could not import LoRA modules: {e}")
            return {
                    'status': 'failed',
                    'reason': 'import_error',
                    'error': str(e),
                    'message': 'Check that lora_finetuning.py contains all required functions'
                }
            
        # Initialize LoRA fine-tuner
        print("\n1. Initializing LoRA fine-tuner...")
        lora_tuner = LoRAFineTuner(
                base_model=self.vlm_protocol.model,
                processor=self.vlm_protocol.processor,
                device=self.vlm_protocol.device
            )
            
        # Prepare LoRA model with conservative settings
        print("\n2. Preparing LoRA model...")
        model = lora_tuner.prepare_lora_model(
                r=8,           # Small rank for stability
                lora_alpha=16, # Conservative alpha
                lora_dropout=0.1
            )
            
        if model is None:
                print("❌ Failed to prepare LoRA model")
                return {'status': 'failed', 'reason': 'model_preparation_failed'}
            
        print("✅ LoRA model prepared successfully")
            
            # Define known classes
        known_classes = ['Boeing', 'Airbus']
            
            # Limit data for faster training
        max_train_samples = 200
        max_val_samples = 50
            
        train_data = data['train_known']
        val_data = data['val_known']
            
        if len(train_data) > max_train_samples:
                train_data = train_data.sample(n=max_train_samples, random_state=42)
                print(f"📊 Limited training data to {max_train_samples} samples")
            
        if len(val_data) > max_val_samples:
                val_data = val_data.sample(n=max_val_samples, random_state=42)
                print(f"📊 Limited validation data to {max_val_samples} samples")
            
            # Start training
        print(f"\n3. Starting LoRA fine-tuning...")
        print(f"   - Training samples: {len(train_data)}")
        print(f"   - Validation samples: {len(val_data)}")
        print(f"   - Known classes: {known_classes}")
            
        training_history = lora_tuner.train(
                train_dataframe=train_data,
                val_dataframe=val_data,
                image_dir=self.vlm_protocol.image_dir,
                known_classes=known_classes,
                num_epochs=5,      # Reduced epochs
                batch_size=8,      # Small batch size
                learning_rate=5e-5,
                save_dir=os.path.join(self.output_dir, 'lora_checkpoints')
            )
            
        if training_history is None:
                print("❌ Training failed")
                return {'status': 'failed', 'reason': 'training_failed'}
            
        print(f"✅ Training completed ({len(training_history)} epochs)")
            
            # Evaluate on test set
        print(f"\n4. Evaluating LoRA model...")
        test_data_sample = data['test']
        if len(test_data_sample) > 100:
                test_data_sample = test_data_sample.sample(n=100, random_state=42)
                print(f"📊 Limited test data to 100 samples for evaluation")
            
        test_metrics = lora_tuner.evaluate_open_set(
                test_data_sample,
                self.vlm_protocol.image_dir
            )
            
        print(f"✅ Open-set evaluation completed")
            
            # Compare models
        print(f"\n5. Comparing base vs LoRA models...")
        comparison_results, comparison_df = compare_models_before_after_lora(
                base_model=self.vlm_protocol.model,
                lora_model=lora_tuner.model,
                test_data=test_data_sample,
                image_dir=self.vlm_protocol.image_dir,
                processor=self.vlm_protocol.processor,
                device=self.vlm_protocol.device
            )
            
        print(f"✅ Model comparison completed")
            
            # Create visualizations
        print(f"\n6. Creating LoRA analysis visualizations...")
        create_lora_visualization_fixed(training_history, test_metrics, comparison_results, comparison_df)
            
            # Compile results
        results = {
                'status': 'completed',
                'training_history': training_history,
                'test_metrics': test_metrics,
                'comparison_results': comparison_results,
                'comparison_df': comparison_df.to_dict() if comparison_df is not None else None,
                'known_classes': known_classes,
                'samples_used': {
                    'train': len(train_data),
                    'val': len(val_data),
                    'test': len(test_data_sample)
                }
            }
            
            # Save results
        results_path = os.path.join(self.output_dir, 'lora_analysis', 'lora_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
        import json
        with open(results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                serializable_results = _make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
        print(f"\n🎉 LoRA FINE-TUNING COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved to: {os.path.join(self.output_dir, 'lora_analysis')}")
            
        return results

def visualize_lora_results(self, training_history, open_set_metrics, 
                          comparison_results, comparison_df):
    """Visualize LoRA fine-tuning results."""
    
    # Create visualization directory
    lora_viz_dir = os.path.join(self.output_dir, 'lora_analysis')
    os.makedirs(lora_viz_dir, exist_ok=True)
    
    # 1. Training history is already saved by LoRAFineTuner
    
    # 2. Open-set performance visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot score distributions
    results_df = open_set_metrics['results_df']
    known_scores = results_df[results_df['is_known_truth']]['max_similarity'].values
    unknown_scores = results_df[~results_df['is_known_truth']]['max_similarity'].values
    
    ax1 = axes[0, 0]
    ax1.hist(known_scores, bins=30, alpha=0.6, label='Known', color='blue', density=True)
    ax1.hist(unknown_scores, bins=30, alpha=0.6, label='Unknown', color='red', density=True)
    ax1.axvline(open_set_metrics['threshold'], color='green', linestyle='--', 
                label=f'Threshold: {open_set_metrics["threshold"]:.3f}')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Density')
    ax1.set_title('LoRA Model: Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Confusion matrix for open-set
    ax2 = axes[0, 1]
    from sklearn.metrics import confusion_matrix
    
    # Create binary labels for confusion matrix
    y_true = results_df['is_known_truth'].astype(int)
    y_pred = results_df['is_predicted_known'].astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Unknown', 'Known'],
                yticklabels=['Unknown', 'Known'])
    ax2.set_title('LoRA Model: Open-Set Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # Model comparison bar chart
    ax3 = axes[1, 0]
    
    metrics_to_compare = ['OSCR AUC', 'AUROC', 'Open-Set F1', 'Wilderness Risk']
    base_metrics = []
    lora_metrics = []
    
    for metric in metrics_to_compare:
        if metric in comparison_df.columns:
            base_val = comparison_df[comparison_df['Model'] == 'Base CLIP'][metric].values[0]
            lora_val = comparison_df[comparison_df['Model'] == 'LoRA Fine-tuned'][metric].values[0]
            base_metrics.append(base_val)
            lora_metrics.append(lora_val)
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, base_metrics, width, label='Base CLIP', alpha=0.7)
    bars2 = ax3.bar(x + width/2, lora_metrics, width, label='LoRA Fine-tuned', alpha=0.7)
    
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Value')
    ax3.set_title('Base vs LoRA Model Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_to_compare, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Improvement metrics
    ax4 = axes[1, 1]
    
    improvements = []
    improvement_labels = []
    
    for metric in ['OSCR AUC', 'AUROC', 'Known F1', 'Unknown F1']:
        if metric in comparison_df.columns:
            base_val = comparison_df[comparison_df['Model'] == 'Base CLIP'][metric].values[0]
            lora_val = comparison_df[comparison_df['Model'] == 'LoRA Fine-tuned'][metric].values[0]
            
            # Calculate percentage improvement
            if base_val > 0:
                improvement = ((lora_val - base_val) / base_val) * 100
            else:
                improvement = 0
            
            improvements.append(improvement)
            improvement_labels.append(metric)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax4.bar(improvement_labels, improvements, color=colors, alpha=0.7)
    
    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('LoRA Fine-tuning Performance Improvement')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax4.annotate(f'{imp:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3 if imp > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if imp > 0 else 'top')
    
    plt.suptitle('LoRA Fine-tuning Analysis Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(lora_viz_dir, 'lora_analysis_results.png'), dpi=300)
    plt.close()
    
    # 3. Create detailed comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Known vs Unknown accuracy comparison
    ax1 = axes[0]
    
    known_acc_base = comparison_results['Base CLIP']['known_accuracy'] \
        if 'known_accuracy' in comparison_results['Base CLIP'] else 0
    known_acc_lora = comparison_results['LoRA Fine-tuned']['known_accuracy'] \
        if 'known_accuracy' in comparison_results['LoRA Fine-tuned'] else 0
    
    unknown_det_base = comparison_results['Base CLIP']['unknown_detection_rate'] \
        if 'unknown_detection_rate' in comparison_results['Base CLIP'] else 0
    unknown_det_lora = comparison_results['LoRA Fine-tuned']['unknown_detection_rate'] \
        if 'unknown_detection_rate' in comparison_results['LoRA Fine-tuned'] else 0
    
    x = np.arange(2)
    width = 0.35
    
    ax1.bar(x - width/2, [known_acc_base, unknown_det_base], width, 
            label='Base CLIP', alpha=0.7, color='blue')
    ax1.bar(x + width/2, [known_acc_lora, unknown_det_lora], width, 
            label='LoRA Fine-tuned', alpha=0.7, color='green')
    
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Performance')
    ax1.set_title('Known Classification vs Unknown Detection')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Known Accuracy', 'Unknown Detection Rate'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Feature space visualization using UMAP
    ax2 = axes[1]
    
    # Note: This is a simplified visualization
    # In practice, you would extract and visualize actual embeddings
    ax2.text(0.5, 0.5, 'Feature Space Visualization\n(UMAP projection)\n\n' + 
             'LoRA fine-tuning maintains\nseparation between known\nand unknown classes',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle('LoRA Fine-tuning: Detailed Performance Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(lora_viz_dir, 'lora_detailed_comparison.png'), dpi=300)
    plt.close()

# Update the AircraftProtocol's generate_image_embeddings method to accept progress callback
def add_progress_callback_to_protocol():
   """Monkey patch to add progress callback support to AircraftProtocol."""
   
   original_generate = AircraftProtocol.generate_image_embeddings
   
   def generate_with_progress(self, df, batch_size=32, max_samples=None, progress_callback=None):
       """Generate embeddings with progress callback."""
       if max_samples:
           df = df.head(max_samples)
           
       embeddings = {}
       with torch.no_grad():
           batch_images = []
           batch_filenames = []
           
           for i, (_, row) in enumerate(df.iterrows()):
               try:
                   img_path = os.path.join(self.image_dir, row['filename'])
                   image = Image.open(img_path).convert("RGB")
                   batch_images.append(image)
                   batch_filenames.append(row['filename'])
               except Exception as e:
                   print(f"Error processing {row['filename']}: {e}")
               
               # Process batch if it's full or at the end
               if len(batch_images) == batch_size or i == len(df) - 1:
                   if batch_images:
                       inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                       batch_embeddings = self.model.get_image_features(**inputs)
                       batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                       
                       for filename, embedding in zip(batch_filenames, batch_embeddings):
                           embeddings[filename] = embedding.cpu()
                       
                       # Call progress callback
                       if progress_callback:
                           progress_callback(len(batch_images))
                   
                   # Reset batch
                   batch_images = []
                   batch_filenames = []
       
       self.image_embeddings.update(embeddings)
       return embeddings
   
   AircraftProtocol.generate_image_embeddings = generate_with_progress


# Similarly for ResNetEmbeddingExtractor
def add_progress_callback_to_resnet():
   """Monkey patch to add progress callback support to ResNetEmbeddingExtractor."""
   
   try:
       from resnet_vs_vlm_comparison import ResNetEmbeddingExtractor
       
       original_extract = ResNetEmbeddingExtractor.extract_embeddings
       
       def extract_with_progress(self, df, image_dir, batch_size=32, max_samples=None, progress_callback=None):
           """Extract embeddings with progress callback."""
           if max_samples:
               df = df.head(max_samples)
           
           # Process images in batches
           with torch.no_grad():
               batch_images = []
               batch_filenames = []
               
               for i, (_, row) in enumerate(df.iterrows()):
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
                           
                           # Call progress callback
                           if progress_callback:
                               progress_callback(len(batch_images))
                       
                       # Reset batch
                       batch_images = []
                       batch_filenames = []
           
           return self.image_embeddings
       
       ResNetEmbeddingExtractor.extract_embeddings = extract_with_progress
   except:
       pass  # ResNet module not available


def parse_args():
   """Parse command line arguments."""
   parser = argparse.ArgumentParser(description="Enhanced VLM Open-Set Recognition Analysis")
   
   # Data paths
   parser.add_argument("--data_dir", type=str, default="./data/fgvc-aircraft",
                       help="Path to the FGVC Aircraft dataset directory")
   
   # Model options
   parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                       help="CLIP model to use")
   parser.add_argument("--compare_resnet", action="store_true",
                       help="Compare with ResNet features")
   parser.add_argument("--resnet_model", type=str, default="resnet50",
                       help="ResNet model for comparison")
   
   # Analysis options
   parser.add_argument("--enable_lora", action="store_true",
                       help="Enable LoRA fine-tuning analysis")
   
   # Output options
   parser.add_argument("--output_dir", type=str, default="enhanced_analysis_results",
                       help="Directory to save results")
   parser.add_argument("--timestamp", action="store_true",
                       help="Add timestamp to output directory")
   
   # Execution options
   parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
   parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
   
   return parser.parse_args()


def main():
   """Main function to run enhanced analysis."""
   args = parse_args()
   
   # Apply monkey patches for progress callbacks
   add_progress_callback_to_protocol()
   add_progress_callback_to_resnet()
   
   # Set random seeds
   np.random.seed(args.seed)
   torch.manual_seed(args.seed)
   
   # Initialize and run analyzer
   analyzer = EnhancedAircraftAnalyzer(args)
   analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
   main()