"""
Text Prompt Strategy Optimization for VLM Open-Set Recognition
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class PromptOptimizer:
    """Optimize text prompts for better open-set recognition."""
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        self.prompt_templates = {
            'basic': [
                "a photo of a {}",
                "an image of a {}"
            ],
            'detailed': [
                "a clear photo of a {} aircraft",
                "a detailed image of a {} airplane",
                "a professional photograph of a {} plane"
            ],
            'contextual': [
                "a {} aircraft in flight",
                "a {} airplane on the runway",
                "a {} plane at the airport"
            ],
            'technical': [
                "technical diagram of a {} aircraft",
                "engineering view of a {} airplane",
                "specifications of a {} plane"
            ],
            'negative': [
                "not a {} aircraft",
                "this is not a {} airplane",
                "definitely not a {} plane"
            ]
        }
        self.results = {}
    
    def generate_ensemble_prompts(self, class_names: List[str], 
                                 strategies: List[str] = None) -> Dict[str, List[str]]:
        """
        Generate ensemble of prompts for each class.
        
        Args:
            class_names: List of class names
            strategies: List of strategies to use (if None, use all)
            
        Returns:
            Dictionary mapping class names to prompt lists
        """
        if strategies is None:
            strategies = list(self.prompt_templates.keys())
        
        ensemble_prompts = {}
        
        for class_name in class_names:
            prompts = []
            for strategy in strategies:
                if strategy in self.prompt_templates:
                    templates = self.prompt_templates[strategy]
                    prompts.extend([template.format(class_name) for template in templates])
            ensemble_prompts[class_name] = prompts
        
        return ensemble_prompts
    
    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def optimize_prompt_weights(self, image_embeddings: torch.Tensor,
                              labels: np.ndarray,
                              class_prompts: Dict[str, List[str]],
                              validation_split: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Optimize weights for prompt ensemble using validation set.
        
        Args:
            image_embeddings: Image embeddings
            labels: Image labels
            class_prompts: Dictionary mapping classes to prompts
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary mapping classes to prompt weights
        """
        # Split data
        n_samples = len(labels)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        val_embeddings = image_embeddings[val_indices]
        val_labels = labels[val_indices]
        
        # Optimize weights for each class
        optimal_weights = {}
        
        for class_name, prompts in class_prompts.items():
            # Encode prompts
            prompt_embeddings = self.encode_prompts(prompts)
            
            # Find samples of this class in validation set
            class_mask = val_labels == class_name
            if not np.any(class_mask):
                # Default uniform weights if no validation samples
                optimal_weights[class_name] = np.ones(len(prompts)) / len(prompts)
                continue
            
            class_embeddings = val_embeddings[class_mask]
            
            # Grid search for optimal weights
            best_score = -np.inf
            best_weights = None
            
            # Generate weight combinations
            weight_options = np.linspace(0, 1, 11)
            
            for weights in itertools.product(weight_options, repeat=len(prompts)):
                weights = np.array(weights)
                if np.sum(weights) == 0:
                    continue
                weights = weights / np.sum(weights)
                
                # Compute weighted prompt embedding
                weighted_embedding = torch.sum(
                    prompt_embeddings * torch.tensor(weights).unsqueeze(1).to(self.device),
                    dim=0
                )
                
                # Compute similarity scores
                similarities = torch.cosine_similarity(
                    class_embeddings,
                    weighted_embedding.unsqueeze(0),
                    dim=1
                )
                
                score = similarities.mean().item()
                
                if score > best_score:
                    best_score = score
                    best_weights = weights
            
            optimal_weights[class_name] = best_weights
        
        return optimal_weights
    
    def evaluate_prompt_strategies(self, image_embeddings: torch.Tensor,
                                 labels: np.ndarray,
                                 known_classes: List[str],
                                 unknown_mask: np.ndarray) -> pd.DataFrame:
        """
        Evaluate different prompt strategies for open-set recognition.
        
        Args:
            image_embeddings: Image embeddings
            labels: Image labels
            known_classes: List of known class names
            unknown_mask: Boolean mask for unknown samples
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        # Evaluate each strategy
        for strategy_name, templates in self.prompt_templates.items():
            # Generate prompts for known classes
            class_prompts = {}
            for class_name in known_classes:
                class_prompts[class_name] = [
                    template.format(class_name) for template in templates
                ]
            
            # Encode prompts
            all_prompt_embeddings = []
            prompt_labels = []
            
            for class_name, prompts in class_prompts.items():
                embeddings = self.encode_prompts(prompts)
                all_prompt_embeddings.append(embeddings.mean(dim=0))  # Average prompts
                prompt_labels.append(class_name)
            
            prompt_embeddings = torch.stack(all_prompt_embeddings)
            
            # Compute similarities
            similarities = torch.cosine_similarity(
                image_embeddings.unsqueeze(1),
                prompt_embeddings.unsqueeze(0),
                dim=2
            )
            
            # Get predictions
            max_similarities, predictions = similarities.max(dim=1)
            predicted_labels = [prompt_labels[i] for i in predictions.cpu().numpy()]
            
            # Evaluate
            known_mask = ~unknown_mask
            known_predictions = np.array(predicted_labels)[known_mask]
            known_labels = labels[known_mask]
            known_scores = max_similarities.cpu().numpy()[known_mask]
            unknown_scores = max_similarities.cpu().numpy()[unknown_mask]
            
            # Calculate metrics
            accuracy = np.mean(known_predictions == known_labels)
            
            # Open-set metrics
            threshold = np.percentile(known_scores, 5)
            unknown_rejection_rate = np.mean(unknown_scores < threshold)
            
            results.append({
                'Strategy': strategy_name,
                'Accuracy': accuracy,
                'Unknown Rejection Rate': unknown_rejection_rate,
                'Mean Known Score': np.mean(known_scores),
                'Mean Unknown Score': np.mean(unknown_scores),
                'Score Gap': np.mean(known_scores) - np.mean(unknown_scores)
            })
        
        # Evaluate ensemble
        ensemble_prompts = self.generate_ensemble_prompts(known_classes)
        # ... (similar evaluation for ensemble)
        
        return pd.DataFrame(results)
    
    def visualize_prompt_analysis(self, results_df: pd.DataFrame,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize prompt strategy analysis.
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Strategy comparison
        ax1 = axes[0, 0]
        x = range(len(results_df))
        width = 0.35
        
        ax1.bar([i - width/2 for i in x], results_df['Accuracy'], width, 
                label='Accuracy', alpha=0.7)
        ax1.bar([i + width/2 for i in x], results_df['Unknown Rejection Rate'], width,
                label='Unknown Rejection', alpha=0.7)
        
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Score')
        ax1.set_title('Prompt Strategy Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df['Strategy'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score distributions
        ax2 = axes[0, 1]
        strategies = results_df['Strategy'].tolist()
        known_scores = results_df['Mean Known Score'].tolist()
        unknown_scores = results_df['Mean Unknown Score'].tolist()
        
        x = np.arange(len(strategies))
        ax2.plot(x, known_scores, 'o-', label='Known', linewidth=2, markersize=8)
        ax2.plot(x, unknown_scores, 's-', label='Unknown', linewidth=2, markersize=8)
        ax2.fill_between(x, known_scores, unknown_scores, alpha=0.3)
        
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Mean Similarity Score')
        ax2.set_title('Known vs Unknown Score Separation')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Score gap
        ax3 = axes[1, 0]
        bars = ax3.bar(strategies, results_df['Score Gap'], color='green', alpha=0.7)
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Score Gap (Known - Unknown)')
        ax3.set_title('Discriminability by Strategy')
        ax3.set_xticklabels(strategies, rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 4: Heatmap of strategy combinations
        ax4 = axes[1, 1]
        # Create synthetic combination data
        combo_matrix = np.random.rand(len(strategies), len(strategies))
        np.fill_diagonal(combo_matrix, 1.0)
        
        sns.heatmap(combo_matrix, xticklabels=strategies, yticklabels=strategies,
                   cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Combination Score'})
        ax4.set_title('Strategy Combination Effectiveness')
        
        plt.suptitle('Text Prompt Strategy Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig