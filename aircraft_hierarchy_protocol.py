"""
Hierarchical VLM Aircraft Protocol - Enhanced Version

A comprehensive protocol for hierarchical aircraft classification using vision-language 
model embeddings, supporting multi-level taxonomy:

Aircraft
├── Commercial
│   ├── Boeing (737, 747, 777, 787)
│   └── Airbus (A320, A330, A350, A380)
├── Military
│   ├── Fighter Jets
│   └── Transport
└── General Aviation
    ├── Private Jets
    └── Light Aircraft

Key features:
- Multi-level hierarchical classification
- Zero-shot classification at each level
- Hierarchical out-of-distribution detection
- Level-specific evaluation metrics
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, confusion_matrix
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
import seaborn as sns
from collections import defaultdict, Counter
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm
import json


class HierarchicalAircraftProtocol:
    """Protocol for hierarchical aircraft classification and OOD detection using VLM embeddings."""
    
    def __init__(
        self, 
        clip_model_name: str = "openai/clip-vit-base-patch32",
        data_dir: str = "./data/fgvc-aircraft",
        device: Optional[str] = None,
        hierarchy_config: Optional[Dict] = None
    ):
        """
        Initialize the hierarchical aircraft classification protocol.
        
        Args:
            clip_model_name: Name of the CLIP model to use.
            data_dir: Directory containing the aircraft dataset.
            device: Device to run the model on ('cuda' or 'cpu').
            hierarchy_config: Custom hierarchy configuration.
        """
        # Set device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        try:
            self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(clip_model_name)
            print(f"Loaded {clip_model_name} model")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise
        
        # Dataset paths
        self.data_dir = os.path.normpath(data_dir)
        
        # Find the correct images directory
        self.image_dir = os.path.join(self.data_dir, "fgvc-aircraft-2013b", "fgvc-aircraft-2013b", "data", "images")
        if not os.path.isdir(self.image_dir):
            for root, dirs, _ in os.walk(self.data_dir):
                if "images" in dirs:
                    self.image_dir = os.path.join(root, "images")
                    print(f"Found images directory at: {self.image_dir}")
                    break
        
        if not os.path.isdir(self.image_dir):
            print(f"Warning: Images directory not found: {self.image_dir}")
        else:
            print(f"Using images directory: {self.image_dir}")
        
        # Find annotation directory
        self.annotation_dir = os.path.dirname(self.image_dir) if os.path.isdir(self.image_dir) else None
        if not self.annotation_dir:
            for root, _, files in os.walk(self.data_dir):
                if any("manufacturer" in f for f in files):
                    self.annotation_dir = root
                    print(f"Found annotation directory at: {self.annotation_dir}")
                    break
        
        # Initialize aircraft hierarchy
        self.hierarchy = hierarchy_config or self._create_default_hierarchy()
        self._build_hierarchy_mappings()
        
        # Initialize containers for embeddings
        self.image_embeddings = {}
        self.text_embeddings = {}
        self.hierarchy_embeddings = {}
        
        # Store classification results at each level
        self.level_results = {}
        
        print("Hierarchical Aircraft Protocol initialized")
        self._print_hierarchy()

    def _create_default_hierarchy(self) -> Dict:
        """Create the default aircraft hierarchy."""
        return {
            "Aircraft": {
                "Commercial": {
                    "Boeing": ["737", "747", "757", "767", "777", "787"],
                    "Airbus": ["A300", "A310", "A318", "A319", "A320", "A321", "A330", "A340", "A350", "A380"]
                },
                "Military": {
                    "Fighter Jets": ["F-16", "F-18", "F-22", "F-35", "Eurofighter", "Rafale", "Gripen"],
                    "Transport": ["C-130", "C-17", "C-5", "A400M", "An-124", "Il-76"]
                },
                "General Aviation": {
                    "Private Jets": ["Cessna Citation", "Gulfstream", "Falcon", "Learjet", "Bombardier"],
                    "Light Aircraft": ["Cessna 172", "Piper", "Beechcraft", "Diamond", "Cirrus"]
                }
            }
        }

    def _build_hierarchy_mappings(self):
        """Build mappings for hierarchical classification."""
        self.level_mappings = {}
        self.reverse_mappings = {}
        
        # Level 1: Category (Commercial, Military, General Aviation)
        self.level_mappings[1] = {}
        self.reverse_mappings[1] = {}
        for category in self.hierarchy["Aircraft"].keys():
            self.level_mappings[1][category] = category
            self.reverse_mappings[1][category] = category
        
        # Level 2: Subcategory (Boeing, Airbus, Fighter Jets, etc.)
        self.level_mappings[2] = {}
        self.reverse_mappings[2] = {}
        for category, subcategories in self.hierarchy["Aircraft"].items():
            for subcategory in subcategories.keys():
                self.level_mappings[2][subcategory] = category
                self.reverse_mappings[2][category] = self.reverse_mappings[2].get(category, []) + [subcategory]
        
        # Level 3: Specific models
        self.level_mappings[3] = {}
        self.reverse_mappings[3] = {}
        for category, subcategories in self.hierarchy["Aircraft"].items():
            for subcategory, models in subcategories.items():
                for model in models:
                    self.level_mappings[3][model] = subcategory
                    self.reverse_mappings[3][subcategory] = self.reverse_mappings[3].get(subcategory, []) + [model]

    def _print_hierarchy(self):
        """Print the aircraft hierarchy."""
        print("\nAircraft Hierarchy:")
        for category, subcategories in self.hierarchy["Aircraft"].items():
            print(f"├── {category}")
            for i, (subcategory, models) in enumerate(subcategories.items()):
                is_last_sub = i == len(subcategories) - 1
                sub_prefix = "└──" if is_last_sub else "├──"
                print(f"│   {sub_prefix} {subcategory}")
                for j, model in enumerate(models[:3]):  # Show first 3 models
                    is_last_model = j == len(models[:3]) - 1 and len(models) <= 3
                    model_prefix = "    └──" if is_last_model else "    ├──"
                    if not is_last_sub:
                        model_prefix = "│" + model_prefix
                    print(f"│   {model_prefix} {model}")
                if len(models) > 3:
                    more_prefix = "    └── ..." if is_last_sub else "│   └── ..."
                    print(f"{more_prefix} ({len(models) - 3} more)")

    def classify_aircraft_hierarchically(self, aircraft_name: str) -> Dict[str, str]:
        """
        Classify an aircraft name into the hierarchy.
        
        Args:
            aircraft_name: Name or identifier of the aircraft.
            
        Returns:
            Dictionary with classification at each level.
        """
        classification = {
            "level_1_category": "Unknown",
            "level_2_subcategory": "Unknown", 
            "level_3_model": "Unknown",
            "original_name": aircraft_name
        }
        
        aircraft_lower = aircraft_name.lower()
        
        # Check level 3 first (specific models)
        for model, subcategory in self.level_mappings[3].items():
            if model.lower() in aircraft_lower or aircraft_lower in model.lower():
                classification["level_3_model"] = model
                classification["level_2_subcategory"] = subcategory
                classification["level_1_category"] = self.level_mappings[2][subcategory]
                return classification
        
        # Check level 2 (subcategories)
        for subcategory, category in self.level_mappings[2].items():
            if subcategory.lower() in aircraft_lower or aircraft_lower in subcategory.lower():
                classification["level_2_subcategory"] = subcategory
                classification["level_1_category"] = category
                return classification
        
        # Check level 1 (categories)
        for category in self.level_mappings[1].keys():
            if category.lower() in aircraft_lower:
                classification["level_1_category"] = category
                return classification
        
        # Special handling for common aircraft identifiers
        if any(boeing_id in aircraft_lower for boeing_id in ['boeing', '7', 'b7']):
            classification["level_1_category"] = "Commercial"
            classification["level_2_subcategory"] = "Boeing"
        elif any(airbus_id in aircraft_lower for airbus_id in ['airbus', 'a3', 'a2']):
            classification["level_1_category"] = "Commercial"
            classification["level_2_subcategory"] = "Airbus"
        elif any(fighter_id in aircraft_lower for fighter_id in ['f-', 'fighter']):
            classification["level_1_category"] = "Military"
            classification["level_2_subcategory"] = "Fighter Jets"
        elif any(cessna_id in aircraft_lower for cessna_id in ['cessna']):
            if any(jet_id in aircraft_lower for jet_id in ['citation', 'jet']):
                classification["level_1_category"] = "General Aviation"
                classification["level_2_subcategory"] = "Private Jets"
            else:
                classification["level_1_category"] = "General Aviation"
                classification["level_2_subcategory"] = "Light Aircraft"
        
        return classification

    def create_hierarchical_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a dataset with hierarchical classifications.
        
        Args:
            df: Original DataFrame with aircraft data.
            
        Returns:
            DataFrame with hierarchical classifications added.
        """
        hierarchical_data = []
        
        for _, row in df.iterrows():
            # Get original aircraft identifier
            if 'Classes' in row:
                aircraft_name = row['Classes']
            elif 'manufacturer' in row:
                aircraft_name = row['manufacturer']
            elif 'family' in row:
                aircraft_name = row['family']
            else:
                aircraft_name = "Unknown"
            
            # Classify hierarchically
            classification = self.classify_aircraft_hierarchically(aircraft_name)
            
            # Create new row with hierarchical data
            new_row = row.to_dict()
            new_row.update(classification)
            hierarchical_data.append(new_row)
        
        hierarchical_df = pd.DataFrame(hierarchical_data)
        
        # Print statistics
        print("\nHierarchical Classification Statistics:")
        for level in [1, 2, 3]:
            col_name = f"level_{level}_{'category' if level == 1 else 'subcategory' if level == 2 else 'model'}"
            if col_name in hierarchical_df.columns:
                counts = hierarchical_df[col_name].value_counts()
                print(f"\nLevel {level} ({col_name}):")
                for label, count in counts.head(10).items():
                    print(f"  {label}: {count}")
                if len(counts) > 10:
                    print(f"  ... and {len(counts) - 10} more")
        
        return hierarchical_df

    def generate_hierarchical_text_embeddings(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate text embeddings for all levels of the hierarchy.
        
        Returns:
            Dictionary with text embeddings for each hierarchy level.
        """
        print("Generating hierarchical text embeddings...")
        
        hierarchical_embeddings = {}
        
        with torch.no_grad():
            # Templates for different levels
            level_templates = {
                1: [  # Category level
                    "a photo of a {} aircraft.",
                    "an image of a {} airplane.",
                    "a {} aircraft in flight.",
                    "a picture of a {} plane.",
                    "a {} aviation vehicle."
                ],
                2: [  # Subcategory level
                    "a photo of a {} aircraft.",
                    "an image of a {} airplane.",
                    "a {} aircraft.",
                    "a picture of a {} plane.",
                    "a {} aviation model."
                ],
                3: [  # Specific model level
                    "a photo of a {} aircraft.",
                    "an image of a {} airplane.",
                    "a {} plane.",
                    "a picture of a {} aircraft model.",
                    "a {} aviation aircraft."
                ]
            }
            
            # Generate embeddings for each level
            for level in [1, 2, 3]:
                level_name = f"level_{level}"
                hierarchical_embeddings[level_name] = {}
                
                if level == 1:
                    items = list(self.hierarchy["Aircraft"].keys())
                elif level == 2:
                    items = []
                    for subcategories in self.hierarchy["Aircraft"].values():
                        items.extend(subcategories.keys())
                else:  # level == 3
                    items = []
                    for subcategories in self.hierarchy["Aircraft"].values():
                        for models in subcategories.values():
                            items.extend(models)
                
                templates = level_templates[level]
                
                for item in tqdm(items, desc=f"Level {level} embeddings"):
                    prompts = [template.format(item) for template in templates]
                    inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                    text_features = self.model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    hierarchical_embeddings[level_name][item] = text_features.mean(dim=0).cpu()
        
        self.hierarchy_embeddings = hierarchical_embeddings
        
        total_embeddings = sum(len(level_embs) for level_embs in hierarchical_embeddings.values())
        print(f"Generated {total_embeddings} hierarchical text embeddings across {len(hierarchical_embeddings)} levels")
        
        return hierarchical_embeddings

    def generate_image_embeddings(
        self, 
        df: pd.DataFrame, 
        batch_size: int = 32,
        max_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for images in the dataset.
        
        Args:
            df: DataFrame with image filenames.
            batch_size: Batch size for processing.
            max_samples: Maximum number of samples to process.
            
        Returns:
            Dictionary mapping image filenames to embeddings.
        """
        if max_samples:
            df = df.head(max_samples)
            
        embeddings = {}
        with torch.no_grad():
            batch_images = []
            batch_filenames = []
            
            for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating image embeddings")):
                try:
                    img_path = os.path.join(self.image_dir, row['filename'])
                    image = Image.open(img_path).convert("RGB")
                    batch_images.append(image)
                    batch_filenames.append(row['filename'])
                except Exception as e:
                    print(f"Error processing {row['filename']}: {e}")
                
                if len(batch_images) == batch_size or i == len(df) - 1:
                    if batch_images:
                        inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                        batch_embeddings = self.model.get_image_features(**inputs)
                        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                        
                        for filename, embedding in zip(batch_filenames, batch_embeddings):
                            embeddings[filename] = embedding.cpu()
                    
                    batch_images = []
                    batch_filenames = []
        
        self.image_embeddings.update(embeddings)
        print(f"Generated embeddings for {len(embeddings)} images")
        return embeddings

    def hierarchical_zero_shot_classification(
        self,
        df: pd.DataFrame,
        level: int = 1
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform hierarchical zero-shot classification at a specific level.
        
        Args:
            df: DataFrame with hierarchical image data.
            level: Hierarchy level to classify (1, 2, or 3).
            
        Returns:
            DataFrame with classification results and metrics dict.
        """
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(df)
            
        if not self.hierarchy_embeddings:
            print("No hierarchical text embeddings found. Generating embeddings...")
            self.generate_hierarchical_text_embeddings()
        
        level_name = f"level_{level}"
        text_embeddings = self.hierarchy_embeddings[level_name]
        
        # Determine ground truth column based on level
        if level == 1:
            gt_column = "level_1_category"
        elif level == 2:
            gt_column = "level_2_subcategory"
        else:
            gt_column = "level_3_model"
        
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Level {level} zero-shot classification"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            img_embedding = self.image_embeddings[filename]
            
            # Compare to each text embedding at this level
            similarities = {}
            for label, text_embedding in text_embeddings.items():
                similarity = 1 - cosine(img_embedding.numpy(), text_embedding.numpy())
                similarities[label] = similarity
            
            # Get prediction
            predicted = max(similarities, key=similarities.get)
            confidence = similarities[predicted]
            
            # Get ground truth
            true_label = row.get(gt_column, "Unknown")
            is_correct = predicted == true_label
            
            result = {
                'filename': filename,
                f'true_{gt_column}': true_label,
                'predicted': predicted,
                'confidence': confidence,
                'is_correct': is_correct,
                'level': level
            }
            
            # Add similarities for each class
            for label, similarity in similarities.items():
                result[f'{label}_similarity'] = similarity
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(results_df, gt_column)
        metrics['level'] = level
        
        return results_df, metrics

    def hierarchical_knn_classification(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        level: int = 1,
        k: int = 5
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform hierarchical KNN classification at a specific level.
        
        Args:
            train_df: Training DataFrame with hierarchical labels.
            test_df: Test DataFrame.
            level: Hierarchy level to classify (1, 2, or 3).
            k: Number of neighbors to use.
            
        Returns:
            DataFrame with classification results and metrics dict.
        """
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(pd.concat([train_df, test_df]))
        
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
            if filename in self.image_embeddings and label_column in row:
                train_embeddings.append(self.image_embeddings[filename].cpu().numpy())
                train_labels.append(row[label_column])
                train_filenames.append(filename)
        
        train_embeddings = np.array(train_embeddings)
        
        # Fit KNN model
        knn = NearestNeighbors(n_neighbors=min(k, len(train_embeddings)))
        knn.fit(train_embeddings)
        
        # Classify test samples
        results = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Level {level} KNN classification"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            img_embedding = self.image_embeddings[filename].cpu().numpy().reshape(1, -1)
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
        metrics = self._calculate_classification_metrics(results_df, label_column)
        metrics['level'] = level
        
        return results_df, metrics

    def hierarchical_ood_detection(
        self,
        df: pd.DataFrame,
        level: int = 1,
        k: int = 5,
        known_classes: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform hierarchical OOD detection at a specific level.
        
        Args:
            df: DataFrame with hierarchical image data.
            level: Hierarchy level for OOD detection (1, 2, or 3).
            k: Number of neighbors to use.
            known_classes: List of known classes. If None, uses a subset.
            
        Returns:
            DataFrame with OOD detection results and metrics dict.
        """
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(df)
        
        # Determine label column and known classes based on level
        if level == 1:
            label_column = "level_1_category"
            if known_classes is None:
                known_classes = ["Commercial", "Military"]  # Exclude General Aviation as OOD
        elif level == 2:
            label_column = "level_2_subcategory"
            if known_classes is None:
                known_classes = ["Boeing", "Airbus"]  # Commercial aircraft only
        else:
            label_column = "level_3_model"
            if known_classes is None:
                known_classes = ["737", "747", "777", "A320", "A330", "A350"]  # Subset of models
        
        # Get known samples
        known_df = df[df[label_column].isin(known_classes)]
        
        # Extract embeddings for known samples
        known_embeddings = []
        known_filenames = []
        
        for _, row in known_df.iterrows():
            filename = row['filename']
            if filename in self.image_embeddings:
                known_embeddings.append(self.image_embeddings[filename].cpu().numpy())
                known_filenames.append(filename)
        
        if len(known_embeddings) == 0:
            print(f"No known embeddings found for level {level}")
            return pd.DataFrame(), {}
        
        known_embeddings = np.array(known_embeddings)
        
        # Fit KNN model on known embeddings
        knn = NearestNeighbors(n_neighbors=min(k, len(known_embeddings)))
        knn.fit(known_embeddings)
        
        # Process each image
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Level {level} OOD detection"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            img_embedding = self.image_embeddings[filename].cpu().numpy().reshape(1, -1)
            distances, indices = knn.kneighbors(img_embedding)
            mean_distance = distances[0].mean()
            
            # Determine if known or OOD
            true_label = row.get(label_column, "Unknown")
            is_known = true_label in known_classes
            
            results.append({
                'filename': filename,
                f'true_{label_column}': true_label,
                'mean_distance': mean_distance,
                'is_true_known': is_known,
                'level': level,
                'known_classes': str(known_classes)
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate OOD metrics
        metrics = self._calculate_ood_metrics(results_df)
        metrics['level'] = level
        metrics['known_classes'] = known_classes
        
        return results_df, metrics

    def _calculate_classification_metrics(self, results_df: pd.DataFrame, gt_column: str) -> Dict:
        """Calculate classification metrics."""
        metrics = {}
        
        if len(results_df) > 0 and 'is_correct' in results_df.columns:
            y_true = results_df[f'true_{gt_column}'].values
            y_pred = results_df['predicted'].values
            
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Handle cases where some classes might not appear in predictions
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
            
            # Confusion matrix
            unique_labels = sorted(set(y_true) | set(y_pred))
            if len(unique_labels) > 0:
                metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=unique_labels)
                metrics['classes'] = unique_labels
        
        return metrics

    def _calculate_ood_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate OOD detection metrics."""
        metrics = {}
        
        if len(results_df) > 0 and 'is_true_known' in results_df.columns:
            y_true = ~results_df['is_true_known']  # True for OOD
            y_scores = results_df['mean_distance']  # Higher distance = more likely OOD
            
            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            metrics['roc_auc'] = roc_auc
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            metrics['thresholds'] = thresholds
            
            # Find optimal threshold (maximize F1)
            f1_scores = []
            for threshold in thresholds:
                y_pred = y_scores > threshold
                if y_pred.sum() > 0 and y_true.sum() > 0:
                    precision = (y_true & y_pred).sum() / y_pred.sum()
                    recall = (y_true & y_pred).sum() / y_true.sum()
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    f1 = 0
                f1_scores.append(f1)
            
            if f1_scores:
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                y_pred_optimal = y_scores > optimal_threshold
                
                metrics['optimal_threshold'] = optimal_threshold
                metrics['optimal_f1'] = f1_scores[optimal_idx]
                
                # Calculate metrics at optimal threshold
                if y_pred_optimal.sum() > 0 and y_true.sum() > 0:
                    metrics['precision'] = (y_true & y_pred_optimal).sum() / y_pred_optimal.sum()
                    metrics['recall'] = (y_true & y_pred_optimal).sum() / y_true.sum()
                    metrics['accuracy'] = (y_true == y_pred_optimal).mean()
                else:
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['accuracy'] = 0.0
        
        return metrics

    def evaluate_all_levels(
        self,
        df: pd.DataFrame,
        methods: List[str] = ["zero_shot", "knn", "ood"],
        k: int = 5,
        train_split: float = 0.7
    ) -> Dict[str, Dict]:
        """
        Evaluate classification performance at all hierarchy levels.
        
        Args:
            df: DataFrame with hierarchical image data.
            methods: List of methods to evaluate.
            k: Number of neighbors for KNN methods.
            train_split: Fraction of data to use for training (for KNN).
            
        Returns:
            Dictionary with results for each level and method.
        """
        print("Evaluating all hierarchy levels...")
        
        # Create hierarchical dataset if not already done
        if 'level_1_category' not in df.columns:
            df = self.create_hierarchical_dataset(df)
        
        # Split data for KNN methods
        train_size = int(len(df) * train_split)
        train_df = df.sample(n=train_size, random_state=42)
        test_df = df.drop(train_df.index)
        
        all_results = {}
        
        for level in [1, 2, 3]:
            level_results = {}
            
            if "zero_shot" in methods:
                print(f"\nRunning zero-shot classification at level {level}...")
                zs_results, zs_metrics = self.hierarchical_zero_shot_classification(df, level=level)
                level_results["zero_shot"] = {
                    "results": zs_results,
                    "metrics": zs_metrics
                }
            
            if "knn" in methods:
                print(f"\nRunning KNN classification at level {level}...")
                knn_results, knn_metrics = self.hierarchical_knn_classification(
                    train_df, test_df, level=level, k=k
                )
                level_results["knn"] = {
                    "results": knn_results,
                    "metrics": knn_metrics
                }
            
            if "ood" in methods:
                print(f"\nRunning OOD detection at level {level}...")
                ood_results, ood_metrics = self.hierarchical_ood_detection(df, level=level, k=k)
                level_results["ood"] = {
                    "results": ood_results,
                    "metrics": ood_metrics
                }
            
            all_results[f"level_{level}"] = level_results
        
        # Store results
        self.level_results = all_results
        
        # Print summary
        self._print_evaluation_summary(all_results)
        
        return all_results

    def _print_evaluation_summary(self, results: Dict):
        """Print a summary of evaluation results across all levels."""
        print("\n" + "="*60)
        print("HIERARCHICAL EVALUATION SUMMARY")
        print("="*60)
        
        for level_name, level_results in results.items():
            level_num = level_name.split("_")[1]
            level_type = ["Category", "Subcategory", "Model"][int(level_num) - 1]
            
            print(f"\nLevel {level_num} ({level_type}):")
            print("-" * 30)
            
            for method_name, method_data in level_results.items():
                metrics = method_data["metrics"]
                print(f"  {method_name.upper()}:")
                
                if "accuracy" in metrics:
                    print(f"    Accuracy: {metrics['accuracy']:.3f}")
                if "f1" in metrics:
                    print(f"    F1 Score: {metrics['f1']:.3f}")
                if "roc_auc" in metrics:
                    print(f"    ROC AUC:  {metrics['roc_auc']:.3f}")

    def visualize_hierarchy_performance(
        self,
        results: Optional[Dict] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Visualize performance across hierarchy levels.
        
        Args:
            results: Results dictionary. If None, uses stored results.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure with performance visualization.
        """
        if results is None:
            results = self.level_results
        
        if not results:
            raise ValueError("No results available. Run evaluate_all_levels() first.")
        
        # Extract metrics for plotting
        levels = []
        methods = []
        accuracies = []
        f1_scores = []
        roc_aucs = []
        
        for level_name, level_results in results.items():
            level_num = int(level_name.split("_")[1])
            
            for method_name, method_data in level_results.items():
                metrics = method_data["metrics"]
                
                levels.append(level_num)
                methods.append(method_name)
                accuracies.append(metrics.get("accuracy", 0))
                f1_scores.append(metrics.get("f1", 0))
                roc_aucs.append(metrics.get("roc_auc", 0))
        
        # Create DataFrame for easier plotting
        plot_df = pd.DataFrame({
            "Level": levels,
            "Method": methods,
            "Accuracy": accuracies,
            "F1_Score": f1_scores,
            "ROC_AUC": roc_aucs
        })
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Hierarchical Aircraft Classification Performance", fontsize=16)
        
        # Plot 1: Accuracy by Level and Method
        ax1 = axes[0, 0]
        for method in plot_df["Method"].unique():
            method_data = plot_df[plot_df["Method"] == method]
            ax1.plot(method_data["Level"], method_data["Accuracy"], 
                    marker='o', label=method, linewidth=2, markersize=8)
        
        ax1.set_xlabel("Hierarchy Level")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy Across Hierarchy Levels")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels(["Category", "Subcategory", "Model"])
        
        # Plot 2: F1 Score by Level and Method
        ax2 = axes[0, 1]
        for method in plot_df["Method"].unique():
            method_data = plot_df[plot_df["Method"] == method]
            if method_data["F1_Score"].sum() > 0:  # Only plot if we have F1 scores
                ax2.plot(method_data["Level"], method_data["F1_Score"], 
                        marker='s', label=method, linewidth=2, markersize=8)
        
        ax2.set_xlabel("Hierarchy Level")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("F1 Score Across Hierarchy Levels")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(["Category", "Subcategory", "Model"])
        
        # Plot 3: ROC AUC for OOD Detection
        ax3 = axes[1, 0]
        ood_data = plot_df[plot_df["Method"] == "ood"]
        if len(ood_data) > 0:
            ax3.bar(ood_data["Level"], ood_data["ROC_AUC"], 
                   color=['skyblue', 'lightgreen', 'lightcoral'])
            ax3.set_xlabel("Hierarchy Level")
            ax3.set_ylabel("ROC AUC")
            ax3.set_title("OOD Detection Performance")
            ax3.set_xticks([1, 2, 3])
            ax3.set_xticklabels(["Category", "Subcategory", "Model"])
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Method Comparison at Each Level
        ax4 = axes[1, 1]
        level_labels = ["Category", "Subcategory", "Model"]
        method_colors = {'zero_shot': 'blue', 'knn': 'green', 'ood': 'red'}
        
        x = np.arange(len(level_labels))
        width = 0.25
        
        for i, method in enumerate(['zero_shot', 'knn', 'ood']):
            method_accs = []
            for level in [1, 2, 3]:
                method_data = plot_df[(plot_df["Level"] == level) & (plot_df["Method"] == method)]
                if len(method_data) > 0:
                    method_accs.append(method_data["Accuracy"].iloc[0])
                else:
                    method_accs.append(0)
            
            ax4.bar(x + i*width, method_accs, width, 
                   label=method, color=method_colors.get(method, 'gray'), alpha=0.7)
        
        ax4.set_xlabel("Hierarchy Level")
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Method Comparison Across Levels")
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(level_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def visualize_hierarchical_embeddings(
        self,
        df: pd.DataFrame,
        level: int = 1,
        method: str = "umap",
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Visualize embeddings colored by hierarchical labels.
        
        Args:
            df: DataFrame with hierarchical image data.
            level: Hierarchy level to visualize (1, 2, or 3).
            method: Dimensionality reduction method ("umap" or "tsne").
            figsize: Figure size.
            
        Returns:
            Matplotlib figure with embedding visualization.
        """
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(df)
        
        if 'level_1_category' not in df.columns:
            df = self.create_hierarchical_dataset(df)
        
        # Determine label column
        if level == 1:
            label_column = "level_1_category"
            title_suffix = "Category"
        elif level == 2:
            label_column = "level_2_subcategory"
            title_suffix = "Subcategory"
        else:
            label_column = "level_3_model"
            title_suffix = "Model"
        
        # Extract embeddings and labels
        embeddings = []
        labels = []
        filenames = []
        
        for _, row in df.iterrows():
            filename = row['filename']
            if filename in self.image_embeddings and label_column in row:
                embeddings.append(self.image_embeddings[filename].cpu().numpy())
                labels.append(row[label_column])
                filenames.append(filename)
        
        if not embeddings:
            raise ValueError(f"No embeddings found for level {level}")
        
        X = np.array(embeddings)
        
        # Apply dimensionality reduction
        if method == "umap":
            from umap import UMAP
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
        else:  # tsne
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_reduced = reducer.fit_transform(X)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # Limit to top 10 classes if too many
        if len(unique_labels) > 10:
            # Count occurrences and keep top 10
            label_counts = Counter(labels)
            top_labels = [label for label, _ in label_counts.most_common(10)]
            
            # Create mask for top labels
            mask = np.array([label in top_labels for label in labels])
            X_reduced_filtered = X_reduced[mask]
            labels_filtered = np.array(labels)[mask]
            
            unique_labels = sorted(set(labels_filtered))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels_filtered == label
                ax.scatter(X_reduced_filtered[mask, 0], X_reduced_filtered[mask, 1],
                          c=[colors[i]], label=label, alpha=0.7, s=50)
            
            ax.set_title(f"Aircraft Embeddings by {title_suffix} ({method.upper()}) - Top 10 Classes")
        else:
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                          c=[colors[i]], label=label, alpha=0.7, s=50)
            
            ax.set_title(f"Aircraft Embeddings by {title_suffix} ({method.upper()})")
        
        # Add text embeddings if available and level-appropriate
        if level <= 2 and self.hierarchy_embeddings:
            level_name = f"level_{level}"
            if level_name in self.hierarchy_embeddings:
                text_embeddings = self.hierarchy_embeddings[level_name]
                
                # Project text embeddings
                text_emb_array = np.array([emb.cpu().numpy() for emb in text_embeddings.values()])
                text_labels = list(text_embeddings.keys())
                
                if len(text_emb_array) > 0:
                    try:
                        text_reduced = reducer.transform(text_emb_array)
                        
                        # Plot text embeddings
                        ax.scatter(text_reduced[:, 0], text_reduced[:, 1],
                                  marker='X', s=200, c='black', 
                                  edgecolors='white', linewidth=2,
                                  label='Text Prompts', zorder=5)
                        
                        # Add labels
                        for i, label in enumerate(text_labels):
                            ax.annotate(label, xy=(text_reduced[i, 0], text_reduced[i, 1]),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=10, fontweight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", 
                                               facecolor="white", alpha=0.8))
                    except:
                        pass  # Skip text embedding projection if it fails
        
        ax.set_xlabel(f"{method.upper()} Dimension 1")
        ax.set_ylabel(f"{method.upper()} Dimension 2")
        
        # Handle legend
        if len(unique_labels) > 10:
            # Put legend outside the plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig

    def generate_hierarchical_report(
        self,
        results: Optional[Dict] = None,
        output_dir: str = "hierarchical_results"
    ) -> str:
        """
        Generate a comprehensive report of hierarchical classification results.
        
        Args:
            results: Results dictionary. If None, uses stored results.
            output_dir: Directory to save the report and visualizations.
            
        Returns:
            Path to the generated report.
        """
        if results is None:
            results = self.level_results
        
        if not results:
            raise ValueError("No results available. Run evaluate_all_levels() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report
        report_path = os.path.join(output_dir, "hierarchical_classification_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Hierarchical Aircraft Classification Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write hierarchy structure
            f.write("## Aircraft Hierarchy\n\n")
            f.write("```\n")
            f.write("Aircraft\n")
            for category, subcategories in self.hierarchy["Aircraft"].items():
                f.write(f"├── {category}\n")
                for i, (subcategory, models) in enumerate(subcategories.items()):
                    is_last_sub = i == len(subcategories) - 1
                    sub_prefix = "└──" if is_last_sub else "├──"
                    f.write(f"│   {sub_prefix} {subcategory}\n")
                    for j, model in enumerate(models[:3]):
                        is_last_model = j == len(models[:3]) - 1 and len(models) <= 3
                        model_prefix = "    └──" if is_last_model else "    ├──"
                        if not is_last_sub:
                            model_prefix = "│" + model_prefix
                        f.write(f"│   {model_prefix} {model}\n")
                    if len(models) > 3:
                        more_prefix = "    └── ..." if is_last_sub else "│   └── ..."
                        f.write(f"{more_prefix} ({len(models) - 3} more)\n")
            f.write("```\n\n")
            
            # Write performance summary
            f.write("## Performance Summary\n\n")
            
            # Create summary table
            f.write("| Level | Type | Zero-Shot Acc | KNN Acc | OOD ROC AUC |\n")
            f.write("|-------|------|---------------|---------|-------------|\n")
            
            level_types = ["Category", "Subcategory", "Model"]
            
            for i, (level_name, level_results) in enumerate(results.items()):
                level_num = int(level_name.split("_")[1])
                level_type = level_types[level_num - 1]
                
                # Extract metrics
                zs_acc = level_results.get("zero_shot", {}).get("metrics", {}).get("accuracy", 0)
                knn_acc = level_results.get("knn", {}).get("metrics", {}).get("accuracy", 0)
                ood_auc = level_results.get("ood", {}).get("metrics", {}).get("roc_auc", 0)
                
                f.write(f"| {level_num} | {level_type} | {zs_acc:.3f} | {knn_acc:.3f} | {ood_auc:.3f} |\n")
            
            f.write("\n")
            
            # Detailed results for each level
            for level_name, level_results in results.items():
                level_num = int(level_name.split("_")[1])
                level_type = level_types[level_num - 1]
                
                f.write(f"## Level {level_num}: {level_type} Classification\n\n")
                
                for method_name, method_data in level_results.items():
                    metrics = method_data["metrics"]
                    
                    f.write(f"### {method_name.replace('_', ' ').title()}\n\n")
                    
                    if "accuracy" in metrics:
                        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
                    if "precision" in metrics:
                        f.write(f"- **Precision**: {metrics['precision']:.4f}\n")
                    if "recall" in metrics:
                        f.write(f"- **Recall**: {metrics['recall']:.4f}\n")
                    if "f1" in metrics:
                        f.write(f"- **F1 Score**: {metrics['f1']:.4f}\n")
                    if "roc_auc" in metrics:
                        f.write(f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n")
                    
                    f.write("\n")
            
            # Add conclusions
            f.write("## Key Findings\n\n")
            
            # Find best performing level for each method
            best_zs_level = 0
            best_zs_acc = 0
            best_knn_level = 0
            best_knn_acc = 0
            best_ood_level = 0
            best_ood_auc = 0
            
            for level_name, level_results in results.items():
                level_num = int(level_name.split("_")[1])
                
                zs_acc = level_results.get("zero_shot", {}).get("metrics", {}).get("accuracy", 0)
                if zs_acc > best_zs_acc:
                    best_zs_acc = zs_acc
                    best_zs_level = level_num
                
                knn_acc = level_results.get("knn", {}).get("metrics", {}).get("accuracy", 0)
                if knn_acc > best_knn_acc:
                    best_knn_acc = knn_acc
                    best_knn_level = level_num
                
                ood_auc = level_results.get("ood", {}).get("metrics", {}).get("roc_auc", 0)
                if ood_auc > best_ood_auc:
                    best_ood_auc = ood_auc
                    best_ood_level = level_num
            
            f.write(f"1. **Best Zero-Shot Performance**: Level {best_zs_level} ({level_types[best_zs_level-1]}) with {best_zs_acc:.3f} accuracy\n")
            f.write(f"2. **Best KNN Performance**: Level {best_knn_level} ({level_types[best_knn_level-1]}) with {best_knn_acc:.3f} accuracy\n")
            f.write(f"3. **Best OOD Detection**: Level {best_ood_level} ({level_types[best_ood_level-1]}) with {best_ood_auc:.3f} ROC AUC\n\n")
            
            # Performance trends
            f.write("### Performance Trends\n\n")
            
            # Analyze how performance changes across levels
            zs_accs = []
            knn_accs = []
            for level_name in sorted(results.keys()):
                level_results = results[level_name]
                zs_accs.append(level_results.get("zero_shot", {}).get("metrics", {}).get("accuracy", 0))
                knn_accs.append(level_results.get("knn", {}).get("metrics", {}).get("accuracy", 0))
            
            if len(zs_accs) >= 2:
                if zs_accs[0] > zs_accs[-1]:
                    f.write("- Zero-shot performance **decreases** as hierarchy becomes more specific\n")
                elif zs_accs[0] < zs_accs[-1]:
                    f.write("- Zero-shot performance **increases** as hierarchy becomes more specific\n")
                else:
                    f.write("- Zero-shot performance remains **stable** across hierarchy levels\n")
            
            if len(knn_accs) >= 2:
                if knn_accs[0] > knn_accs[-1]:
                    f.write("- KNN performance **decreases** as hierarchy becomes more specific\n")
                elif knn_accs[0] < knn_accs[-1]:
                    f.write("- KNN performance **increases** as hierarchy becomes more specific\n")
                else:
                    f.write("- KNN performance remains **stable** across hierarchy levels\n")
        
        print(f"Hierarchical classification report saved to: {report_path}")
        return report_path

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file or create from TXT annotations if CSV doesn't exist.
        """
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"Loaded {len(df)} samples from {csv_path}")
            else:
                split = os.path.basename(csv_path).replace('.csv', '')
                df = self.create_dataset_from_txt(split)
                df.to_csv(csv_path, index=False)
                print(f"Created and saved {len(df)} samples to {csv_path}")
            
            # Ensure filename column has .jpg extension
            if 'filename' in df.columns and not df['filename'].str.endswith('.jpg').all():
                df['filename'] = df['filename'].apply(lambda x: f"{x}.jpg" if not str(x).endswith('.jpg') else x)
            
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame(columns=['filename', 'Classes'])

    def create_dataset_from_txt(self, split='train'):
        """
        Create a dataset DataFrame from the TXT annotation files.
        """
        annotation_file = os.path.join(self.annotation_dir, f"images_family_{split}.txt")
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found: {annotation_file}")
            return pd.DataFrame(columns=['filename', 'Classes'])
        
        rows = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    aircraft_class = " ".join(parts[1:])  # Join all parts after image_id
                    
                    rows.append({
                        'filename': f"{image_id}.jpg",
                        'Classes': aircraft_class
                    })
        
        df = pd.DataFrame(rows)
        print(f"Created {split} dataset with {len(df)} samples")
        
        if 'Classes' in df.columns:
            class_counts = df['Classes'].value_counts()
            print(f"Found {len(class_counts)} unique aircraft classes")
            for class_name, count in class_counts.head(10).items():
                print(f"  {class_name}: {count} samples")
            if len(class_counts) > 10:
                print(f"  ... and {len(class_counts) - 10} more classes")
        
        return df