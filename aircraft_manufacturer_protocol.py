"""
VLM Aircraft Protocol - Updated for Aircraft Model Names

A comprehensive protocol for evaluating aircraft model classification and 
manufacturer classification using vision-language model embeddings, with a 
focus on out-of-distribution detection for distinguishing Boeing, Airbus, 
and unknown manufacturers.
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
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm


class AircraftProtocol:
    """Protocol for aircraft classification and OOD detection using VLM embeddings."""
    
    def __init__(
        self, 
        clip_model_name: str = "openai/clip-vit-base-patch32",
        data_dir: str = "./data/fgvc-aircraft",
        device: Optional[str] = None,
        manufacturer_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the aircraft classification protocol.
        
        Args:
            clip_model_name: Name of the CLIP model to use.
            data_dir: Directory containing the FGVC-Aircraft dataset.
            device: Device to run the model on ('cuda' or 'cpu').
            manufacturer_mapping: Mapping from aircraft models to manufacturers.
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
            # Try to find the images directory
            for root, dirs, _ in os.walk(self.data_dir):
                if "images" in dirs:
                    self.image_dir = os.path.join(root, "images")
                    print(f"Found images directory at: {self.image_dir}")
                    break
        
        # Check if image directory exists
        if not os.path.isdir(self.image_dir):
            print(f"Warning: Images directory not found: {self.image_dir}")
        else:
            print(f"Using images directory: {self.image_dir}")
        
        # Find the data directory (containing annotation files)
        self.annotation_dir = os.path.dirname(self.image_dir) if os.path.isdir(self.image_dir) else None
        if not self.annotation_dir:
            # Try to find the annotation directory
            for root, _, files in os.walk(self.data_dir):
                if any("manufacturer" in f or "families" in f or "mailies" in f for f in files):
                    self.annotation_dir = root
                    print(f"Found annotation directory at: {self.annotation_dir}")
                    break
        
        # Load manufacturer mapping if provided, otherwise create a default
        self.manufacturer_mapping = manufacturer_mapping or self._create_mapping()
        
        # Initialize containers for embeddings
        self.image_embeddings = {}
        self.text_embeddings = {}
        self.class_embeddings = {}
        self.manufacturer_embeddings = {}
        
        # Store class and manufacturer lists
        self.classes = []
        self.manufacturers = set(["Boeing", "Airbus", "Unknown"])
        
        # Map for determining known vs unknown manufacturers
        self.known_manufacturers = set(["Boeing", "Airbus"])

    def _create_mapping(self) -> Dict[str, str]:
        """
        Create a mapping from aircraft model names to manufacturers.
        Loads model names from the dataset and maps them to Boeing, Airbus or Unknown.
        
        Returns:
            Dict mapping aircraft models to manufacturers.
        """
        mapping = {}
        
        # Try to load families.txt or manufacturers.txt from the dataset
        families_file = None
        for filename in ["families.txt", "manufacturers.txt", "mailies.txt"]:
            potential_path = os.path.join(self.annotation_dir, filename) if self.annotation_dir else None
            if potential_path and os.path.exists(potential_path):
                families_file = potential_path
                break
        
        if families_file:
            with open(families_file, 'r') as f:
                families = [line.strip() for line in f if line.strip()]
            
            # Map aircraft families to manufacturers
            for family in families:
                if family.startswith('7') or family.startswith('Boeing'):
                    mapping[family] = "Boeing"
                elif family.startswith('A3') or family.startswith('Airbus'):
                    mapping[family] = "Airbus"
                else:
                    mapping[family] = "Unknown"
            
            print(f"Loaded {len(families)} aircraft families from {families_file}")
        else:
            # Fallback mapping for common models
            # Boeing aircraft families
            boeing_models = ['707', '717', '727', '737', '747', '757', '767', '777', '787', 'Boeing']
            for model in boeing_models:
                mapping[model] = 'Boeing'
            
            # Airbus aircraft families
            airbus_models = ['A300', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330', 'A340', 'A350', 'A380', 'Airbus']
            for model in airbus_models:
                mapping[model] = 'Airbus'
            
            # Add prefixes for model variants
            for i in range(7):
                mapping[f'7{i}7'] = 'Boeing'
            for i in range(3, 5):
                mapping[f'A{i}'] = 'Airbus'
        
        print(f"Created mapping with {sum(1 for v in mapping.values() if v == 'Boeing')} Boeing models, "
              f"{sum(1 for v in mapping.values() if v == 'Airbus')} Airbus models, and "
              f"{sum(1 for v in mapping.values() if v == 'Unknown')} Unknown models")
        
        return mapping
    
    def get_manufacturer(self, model: str) -> str:
        """
        Get the manufacturer for a given aircraft model.
        """
        # Check direct mapping first
        if model in self.manufacturer_mapping:
            return self.manufacturer_mapping[model]
        
        # Try to infer manufacturer from model name patterns
        if model.startswith('7') or model.startswith('Boeing'):
            return 'Boeing'
        elif model.startswith('A3') or model.startswith('Airbus'):
            return 'Airbus'
        else:
            return 'Unknown'
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file or create from TXT annotations if CSV doesn't exist.
        """
        try:
            # Check if the CSV file exists
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"Loaded {len(df)} samples from {csv_path}")
                
                # Ensure filename column has .jpg extension if not already
                if 'filename' in df.columns and not df['filename'].str.endswith('.jpg').all():
                    df['filename'] = df['filename'].apply(lambda x: f"{x}.jpg" if not str(x).endswith('.jpg') else x)
                
                # Map Classes to manufacturer if needed
                if 'Classes' in df.columns and 'manufacturer' not in df.columns:
                    df['manufacturer'] = df['Classes'].apply(self.get_manufacturer)
                    print("Created 'manufacturer' column from 'Classes'")
                
                # Save original Classes as model name if not present
                if 'Classes' in df.columns and 'model' not in df.columns:
                    df['model'] = df['Classes']
                
                # Update list of classes
                if 'Classes' in df.columns:
                    self.classes = sorted(df['Classes'].unique())
                    print(f"Found {len(self.classes)} unique aircraft models")
                elif 'model' in df.columns:
                    self.classes = sorted(df['model'].unique())
                    print(f"Found {len(self.classes)} unique aircraft models")
                
                # Print stats on manufacturers
                if 'manufacturer' in df.columns:
                    manufacturer_counts = df['manufacturer'].value_counts()
                    for manufacturer, count in manufacturer_counts.items():
                        print(f"  {manufacturer}: {count} samples")
                
                return df
            else:
                # Extract split name from csv_path
                split = os.path.basename(csv_path).replace('.csv', '')
                
                # If CSV doesn't exist, try to create dataset from annotation files
                print(f"CSV file not found: {csv_path}")
                print(f"Returning empty DataFrame with required columns")
                return pd.DataFrame(columns=['filename', 'Classes', 'manufacturer', 'model'])
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['filename', 'Classes', 'manufacturer', 'model'])
        
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
            
        # Process images in batches
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
                
                # Process batch if it's full or at the end
                if len(batch_images) == batch_size or i == len(df) - 1:
                    if batch_images:
                        inputs = self.processor(images=batch_images, return_tensors="pt", padding=True).to(self.device)
                        batch_embeddings = self.model.get_image_features(**inputs)
                        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                        
                        for filename, embedding in zip(batch_filenames, batch_embeddings):
                            embeddings[filename] = embedding.cpu()
                    
                    # Reset batch
                    batch_images = []
                    batch_filenames = []
        
        self.image_embeddings.update(embeddings)
        print(f"Generated embeddings for {len(embeddings)} images")
        return embeddings
    
    def generate_text_embeddings(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate text embeddings for class names and manufacturers.
        This is the first step in our VLM pipeline.
        
        Returns:
            Dictionary with text embeddings for classes and manufacturers.
        """
        # Generate class embeddings
        with torch.no_grad():
            # Class prompt templates
            class_templates = [
                "a photo of a {} aircraft.",
                "an image of a {} airplane.",
                "a picture of a {} plane.",
                "a {} aircraft.",
                "an airplane model {}."
            ]
            
            # Manufacturer prompt templates
            manufacturer_templates = [
                "a photo of a {} aircraft.",
                "an image of a {} airplane.",
                "a picture of a {} plane.",
                "a {} aircraft.",
                "an airplane made by {}."
            ]
            
            # Generate embeddings for each class (aircraft model)
            print("Generating aircraft model text embeddings...")
            for class_name in tqdm(self.classes):
                prompts = [template.format(class_name) for template in class_templates]
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.class_embeddings[class_name] = text_features.mean(dim=0).cpu()
                
            # Generate embeddings for manufacturers
            print("Generating manufacturer text embeddings...")
            for manufacturer in ["Boeing", "Airbus", "Unknown"]:
                prompts = [template.format(manufacturer) for template in manufacturer_templates]
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.manufacturer_embeddings[manufacturer] = text_features.mean(dim=0).cpu()
        
        self.text_embeddings = {
            'classes': self.class_embeddings,
            'manufacturers': self.manufacturer_embeddings
        }
        
        print(f"Generated text embeddings for {len(self.class_embeddings)} aircraft models and {len(self.manufacturer_embeddings)} manufacturers")
        return self.text_embeddings
    
    def zero_shot_classification(
        self,
        df: pd.DataFrame,
        use_manufacturers: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform zero-shot classification by comparing image embeddings to text embeddings.
        This implements a direct comparison between image and text embeddings.
        
        Args:
            df: DataFrame with image data.
            use_manufacturers: Whether to use manufacturer embeddings (True) or class embeddings (False).
            
        Returns:
            DataFrame with zero-shot classification results and metrics dict.
        """
        # Ensure we have embeddings
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(df)
            
        if not self.text_embeddings:
            print("No text embeddings found. Generating text embeddings...")
            self.generate_text_embeddings()
        
        # Choose text embeddings to use
        if use_manufacturers:
            text_embeddings = self.manufacturer_embeddings
            label_key = 'manufacturer'
        else:
            text_embeddings = self.class_embeddings
            label_key = 'Classes' if 'Classes' in df.columns else 'model'
        
        # Classify each image
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot classification"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            # Get image embedding
            img_embedding = self.image_embeddings[filename]
            
            # Compare to each text embedding
            similarities = {}
            for label, text_embedding in text_embeddings.items():
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    img_embedding.unsqueeze(0), 
                    text_embedding.unsqueeze(0)
                ).item()
                similarities[label] = similarity
            
            # Get prediction (highest similarity)
            predicted = max(similarities, key=similarities.get)
            confidence = similarities[predicted]
            
            # Get ground truth
            if label_key in row:
                true_label = row[label_key]
                is_correct = predicted == true_label
            else:
                true_label = None
                is_correct = None
            
            # Store result
            result = {
                'filename': filename,
                f'true_{label_key}': true_label,
                'predicted': predicted,
                'confidence': confidence,
                'is_correct': is_correct
            }
            
            # Add similarities for each class
            for label, similarity in similarities.items():
                result[f'{label}_similarity'] = similarity
            
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        metrics = {}
        
        # Only calculate metrics if we have ground truth
        if label_key in df.columns and 'is_correct' in results_df.columns:
            accuracy = results_df['is_correct'].mean()
            metrics['accuracy'] = accuracy
            
            # Calculate precision, recall, f1 if possible
            if len(results_df) > 0 and not results_df[f'true_{label_key}'].isna().all():
                y_true = results_df[f'true_{label_key}'].values
                y_pred = results_df['predicted'].values
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0
                )
                
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
                
                # Create confusion matrix
                labels = sorted(set(y_true) | set(y_pred))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                metrics['confusion_matrix'] = cm
                metrics['labels'] = labels
        
        return results_df, metrics
    
    def knn_binary_classification(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        k: int = 5,
        classification_target: str = 'manufacturer'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform KNN classification.
        
        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            k: Number of neighbors to use.
            classification_target: Target column ('manufacturer' or 'Classes')
            
        Returns:
            DataFrame with classification results and metrics dict.
        """
        # Ensure we have embeddings
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(pd.concat([train_df, test_df]))
            
        # Extract embeddings for training samples
        train_embeddings = []
        train_labels = []
        train_filenames = []
        
        for _, row in train_df.iterrows():
            filename = row['filename']
            if filename in self.image_embeddings:
                train_embeddings.append(self.image_embeddings[filename].cpu().numpy())
                
                # Get label based on classification target
                if classification_target in row:
                    train_labels.append(row[classification_target])
                elif classification_target == 'manufacturer' and 'Classes' in row:
                    # Derive manufacturer from class if needed
                    train_labels.append(self.get_manufacturer(row['Classes']))
                else:
                    continue  # Skip if no valid label
                    
                train_filenames.append(filename)
        
        train_embeddings = np.array(train_embeddings)
        
        # Fit KNN model
        knn = NearestNeighbors(n_neighbors=min(k, len(train_embeddings)))
        knn.fit(train_embeddings)
        
        # Classify test samples
        results = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"KNN classification (k={k})"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            # Get image embedding
            img_embedding = self.image_embeddings[filename].cpu().numpy().reshape(1, -1)
            
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
                true_label = self.get_manufacturer(row['Classes'])
            else:
                true_label = None
            
            # Check if known class based on classification target
            if classification_target == 'manufacturer':
                known_class = true_label in self.known_manufacturers
            else:
                # For aircraft model classification, all test classes are considered known
                known_class = True
            
            # Store result
            results.append({
                'filename': filename,
                f'true_{classification_target}': true_label,
                'predicted': predicted,
                'confidence': confidence,
                'known_class': known_class,
                'is_correct': predicted == true_label,
                'neighbor_distances': distances[0].tolist(),
                'neighbor_labels': neighbor_labels
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Only calculate metrics on known classes for manufacturer classification
        if classification_target == 'manufacturer':
            known_results = results_df[results_df['known_class']]
        else:
            known_results = results_df
        
        # Calculate metrics
        metrics = {}
        
        if len(known_results) > 0:
            y_true = known_results[f'true_{classification_target}'].values
            y_pred = known_results['predicted'].values
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            
            # Create confusion matrix
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
            if classification_target == 'manufacturer':
                metrics['confusion_matrix'] = np.zeros((2, 2))
                metrics['classes'] = ["Boeing", "Airbus"]
            else:
                metrics['confusion_matrix'] = np.zeros((1, 1))
                metrics['classes'] = ["Unknown"]
        
        return results_df, metrics
    
    def knn_open_set_evaluation(
        self, 
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        k: int = 5,
        threshold: float = 0.7,
        classification_target: str = 'manufacturer'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform open-set evaluation using k-nearest neighbors with distance threshold.
        
        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame (including unknown types).
            k: Number of neighbors to use.
            threshold: Distance threshold for rejection.
            classification_target: Target for classification ('manufacturer' or 'Classes')
            
        Returns:
            DataFrame with classification results and metrics dict.
        """
        # Ensure we have embeddings
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(pd.concat([train_df, test_df]))
            
        # Extract embeddings for training samples
        train_embeddings = []
        train_labels = []
        train_filenames = []
        
        for _, row in train_df.iterrows():
            filename = row['filename']
            if filename in self.image_embeddings:
                # Get label based on classification target
                if classification_target in row:
                    label = row[classification_target]
                elif classification_target == 'manufacturer' and 'Classes' in row:
                    label = self.get_manufacturer(row['Classes'])
                else:
                    continue  # Skip if no valid label
                
                train_embeddings.append(self.image_embeddings[filename].cpu().numpy())
                train_labels.append(label)
                train_filenames.append(filename)
        
        train_embeddings = np.array(train_embeddings)
        
        # Fit KNN model
        knn = NearestNeighbors(n_neighbors=min(k, len(train_embeddings)))
        knn.fit(train_embeddings)
        
        # Define known categories based on classification target
        if classification_target == 'manufacturer':
            known_categories = self.known_manufacturers
        else:
            known_categories = set(train_labels)
        
        # Process test samples
        results = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Open-set evaluation (k={k})"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            # Get image embedding
            img_embedding = self.image_embeddings[filename].cpu().numpy().reshape(1, -1)
            
            # Find k nearest neighbors
            distances, indices = knn.kneighbors(img_embedding)
            mean_distance = distances[0].mean()
            
            # Get neighbor labels
            neighbor_labels = [train_labels[i] for i in indices[0]]
            
            # Get most common label (majority vote)
            from collections import Counter
            label_counts = Counter(neighbor_labels)
            predicted = label_counts.most_common(1)[0][0]
            confidence = label_counts[predicted] / k
            
            # Check if sample should be rejected as unknown
            is_rejected = mean_distance > threshold
            
            # Final prediction with rejection
            if classification_target == 'manufacturer':
                final_prediction = "Unknown" if is_rejected else predicted
            else:
                final_prediction = "Unknown Model" if is_rejected else predicted
            
            # Get true label based on classification target
            if classification_target in row:
                true_label = row[classification_target]
            elif classification_target == 'manufacturer' and 'Classes' in row:
                true_label = self.get_manufacturer(row['Classes'])
            else:
                true_label = None
            
            # Check if sample belongs to a known category
            is_known = true_label in known_categories
            
            # Store result
            results.append({
                'filename': filename,
                f'true_{classification_target}': true_label,
                'raw_prediction': predicted,
                'final_prediction': final_prediction,
                'confidence': confidence,
                'mean_distance': mean_distance,
                'is_rejected': is_rejected,
                'is_known': is_known,
                'is_correct': final_prediction == true_label,
                'neighbor_distances': distances[0].tolist(),
                'neighbor_labels': neighbor_labels
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate open-set metrics
        metrics = self._calculate_open_set_metrics(results_df, classification_target)
        
        return results_df, metrics
    
    def _calculate_open_set_metrics(self, results_df: pd.DataFrame, target: str = 'manufacturer') -> Dict:
        """
        Calculate metrics for open-set evaluation.
        
        Args:
            results_df: DataFrame with classification results.
            target: Classification target ('manufacturer' or 'Classes')
            
        Returns:
            Dictionary with metrics.
        """
        metrics = {}
        
        # Standard classification metrics on all samples
        y_true = results_df[f'true_{target}'].values
        y_pred = results_df['final_prediction'].values
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Open-set specific metrics
        
        # 1. Detection metrics (known vs unknown)
        true_known = results_df['is_known'].values
        pred_known = ~results_df['is_rejected'].values
        
        # Calculate detection accuracy
        metrics['detection_accuracy'] = accuracy_score(true_known, pred_known)
        
        # Detection rates
        if true_known.sum() > 0:
            # True positive rate for known samples
            metrics['known_detection_rate'] = (true_known & pred_known).sum() / true_known.sum()
        else:
            metrics['known_detection_rate'] = 0.0
            
        if (~true_known).sum() > 0:
            # True positive rate for unknown samples
            metrics['unknown_detection_rate'] = (~true_known & ~pred_known).sum() / (~true_known).sum()
        else:
            metrics['unknown_detection_rate'] = 0.0
        
        # 2. Classification metrics on correctly detected known samples
        correctly_detected_known = (true_known & pred_known)
        if correctly_detected_known.sum() > 0:
            known_true = results_df.loc[correctly_detected_known, f'true_{target}'].values
            known_pred = results_df.loc[correctly_detected_known, 'final_prediction'].values
            
            metrics['known_accuracy'] = accuracy_score(known_true, known_pred)
            metrics['known_precision'], metrics['known_recall'], metrics['known_f1'], _ = precision_recall_fscore_support(
                known_true, known_pred, average='weighted', zero_division=0
            )
        else:
            metrics['known_accuracy'] = 0.0
            metrics['known_precision'] = 0.0
            metrics['known_recall'] = 0.0
            metrics['known_f1'] = 0.0
        
        # 3. ROC metrics for detection (using mean_distance as score)
        if 'is_known' in results_df.columns and 'mean_distance' in results_df.columns:
            fpr, tpr, _ = roc_curve(~results_df['is_known'], results_df['mean_distance'])
            metrics['detection_roc_auc'] = auc(fpr, tpr)
            metrics['detection_fpr'] = fpr
            metrics['detection_tpr'] = tpr
        
        # 4. Normalized accuracy (average of accuracy on known and unknown)
        if true_known.sum() > 0 and (~true_known).sum() > 0:
            # Accuracy on known samples
            known_acc = accuracy_score(
                results_df.loc[true_known, f'true_{target}'], 
                results_df.loc[true_known, 'final_prediction']
            )
            
            # Accuracy on unknown samples
            if target == 'manufacturer':
                unknown_pred_correct = (results_df.loc[~true_known, 'final_prediction'] == "Unknown")
            else:
                unknown_pred_correct = (results_df.loc[~true_known, 'final_prediction'] == "Unknown Model")
                
            unknown_acc = unknown_pred_correct.mean()
            
            # Normalized accuracy (average of known and unknown accuracy)
            metrics['normalized_accuracy'] = (known_acc + unknown_acc) / 2
        else:
            metrics['normalized_accuracy'] = 0.0
        
        return metrics
    
    def knn_ood_detection(
        self, 
        df: pd.DataFrame, 
        k: int = 5,
        threshold: float = 0.7,
        class_type: str = 'manufacturer'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform OOD detection using k-nearest neighbors.
        
        Args:
            df: DataFrame with image data.
            k: Number of neighbors to use.
            threshold: Distance threshold for OOD detection.
            class_type: Type of class to use ('manufacturer' or 'model')
            
        Returns:
            DataFrame with OOD detection results and metrics dict.
        """
        # Ensure we have embeddings
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(df)
        
        # Get known samples
        if class_type == 'manufacturer':
            known_df = df[df['manufacturer'].isin(self.known_manufacturers)]
        else:
            # For model-level OOD, use all models in our class embeddings
            known_classes = list(self.class_embeddings.keys())
            known_df = df[df['Classes'].isin(known_classes) | df['model'].isin(known_classes)]
        
        # Extract embeddings for known samples
        known_embeddings = []
        known_filenames = []
        
        for _, row in known_df.iterrows():
            filename = row['filename']
            if filename in self.image_embeddings:
                known_embeddings.append(self.image_embeddings[filename].cpu().numpy())
                known_filenames.append(filename)
        
        known_embeddings = np.array(known_embeddings)
        
        # Fit KNN model on known embeddings
        knn = NearestNeighbors(n_neighbors=min(k, len(known_embeddings)))
        knn.fit(known_embeddings)
        
        # Process each image
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="KNN OOD detection"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            # Get image embedding
            img_embedding = self.image_embeddings[filename].cpu().numpy().reshape(1, -1)
            
            # Calculate distance to k nearest neighbors
            distances, indices = knn.kneighbors(img_embedding)
            mean_distance = distances[0].mean()
            
            # Get nearest neighbors information
            neighbor_files = [known_filenames[i] for i in indices[0]]
            neighbors_df = known_df[known_df['filename'].isin(neighbor_files)]
            
            # Get neighbor labels based on class type
            if class_type == 'manufacturer':
                neighbor_labels = neighbors_df['manufacturer'].values
                is_known = row['manufacturer'] in self.known_manufacturers
            else:
                # For model-level, use either 'Classes' or 'model' column
                if 'Classes' in neighbors_df.columns:
                    neighbor_labels = neighbors_df['Classes'].values
                    is_known = 'Classes' in row and row['Classes'] in self.class_embeddings
                else:
                    neighbor_labels = neighbors_df['model'].values
                    is_known = 'model' in row and row['model'] in self.class_embeddings
            
            # Most common label among neighbors
            if len(neighbor_labels) > 0:
                from collections import Counter
                labels, counts = np.unique(neighbor_labels, return_counts=True)
                best_match = labels[counts.argmax()]
            else:
                best_match = "Unknown"
            
            # Determine if in-distribution or OOD
            is_predicted_known = mean_distance <= threshold
            is_ood = not is_predicted_known
            
            results.append({
                'filename': filename,
                f'true_{class_type}': row.get(class_type, None),
                'best_match': best_match,
                'mean_distance': mean_distance,
                'is_true_known': is_known,
                'is_predicted_known': is_predicted_known,
                'is_ood': is_ood
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate OOD metrics
        y_true = ~results_df['is_true_known']
        y_scores = results_df['mean_distance']
        y_pred = results_df['is_ood']
        
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate other metrics
        accuracy = (y_true == y_pred).mean()
        true_ood = y_true & y_pred
        false_ood = ~y_true & y_pred
        missed_ood = y_true & ~y_pred
        
        ood_precision = true_ood.sum() / y_pred.sum() if y_pred.sum() > 0 else 0
        ood_recall = true_ood.sum() / y_true.sum() if y_true.sum() > 0 else 0
        ood_f1 = 2 * (ood_precision * ood_recall) / (ood_precision + ood_recall) if (ood_precision + ood_recall) > 0 else 0
        
        # Calculate normalized accuracy (average of accuracy on known and unknown)
        known_acc = ((~y_true) & (~y_pred)).sum() / (~y_true).sum() if (~y_true).sum() > 0 else 0
        unknown_acc = (y_true & y_pred).sum() / y_true.sum() if y_true.sum() > 0 else 0
        normalized_acc = (known_acc + unknown_acc) / 2
        
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': ood_precision,
            'recall': ood_recall,
            'f1': ood_f1,
            'normalized_accuracy': normalized_acc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'false_ood_rate': false_ood.sum() / (~y_true).sum() if (~y_true).sum() > 0 else 0,
            'missed_ood_rate': missed_ood.sum() / y_true.sum() if y_true.sum() > 0 else 0
        }
        
        return results_df, metrics
    
    def direct_embedding_comparison(
        self,
        image_df: pd.DataFrame,
        target: str = 'manufacturer',
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Directly compare image embeddings to text embeddings using cosine similarity.
        
        Args:
            image_df: DataFrame with image data.
            target: The target for comparison ('manufacturer' or 'model')
            normalize: Whether to normalize similarities to sum to 1.
            
        Returns:
            DataFrame with cosine similarities.
        """
        # Ensure we have embeddings
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(image_df)
            
        if not self.text_embeddings:
            print("No text embeddings found. Generating text embeddings...")
            self.generate_text_embeddings()
        
        # Get text embeddings for target
        if target == 'manufacturer':
            text_embeddings = {
                m: self.manufacturer_embeddings[m].numpy() 
                for m in self.manufacturer_embeddings
            }
        else:  # 'model' or 'Classes'
            text_embeddings = {
                c: self.class_embeddings[c].numpy() 
                for c in self.class_embeddings
            }
        
        # Calculate similarities
        results = []
        
        for _, row in tqdm(image_df.iterrows(), total=len(image_df), desc="Calculating similarities"):
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            # Get image embedding
            img_embedding = self.image_embeddings[filename].numpy()
            
            # Calculate similarities to each text embedding
            similarities = {}
            for label, text_emb in text_embeddings.items():
                similarity = 1 - cosine(img_embedding, text_emb)
                similarities[f"{label}_similarity"] = similarity
            
            # Normalize similarities if requested
            if normalize and similarities:
                total = sum(similarities.values())
                if total > 0:
                    for key in similarities:
                        similarities[key] = similarities[key] / total
            
            # Add to results
            result = {
                'filename': filename,
            }
            
            # Add true label based on target
            if target == 'manufacturer':
                result['true_manufacturer'] = row.get('manufacturer') if 'manufacturer' in row else None
            elif 'Classes' in row:
                result['true_model'] = row['Classes'] 
            elif 'model' in row:
                result['true_model'] = row['model']
            
            result.update(similarities)
            
            # Add best match
            if similarities:
                best_match = max(similarities.items(), key=lambda x: x[1])[0].replace("_similarity", "")
                result['best_match'] = best_match
                
                # Check if correct based on target
                if target == 'manufacturer' and 'manufacturer' in row:
                    result['is_correct'] = best_match == row['manufacturer']
                elif target == 'model' and 'Classes' in row:
                    result['is_correct'] = best_match == row['Classes']
                elif target == 'model' and 'model' in row:
                    result['is_correct'] = best_match == row['model']
            
            results.append(result)
        
        # Convert to DataFrame
        return pd.DataFrame(results)
    
    def compare_image_to_text_embeddings(
        self,
        df: pd.DataFrame,
        method: str = 'cosine',
        target: str = 'manufacturer',
        plot: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
        """
        Compare image embeddings to text embeddings and visualize results.
        
        Args:
            df: DataFrame with image data.
            method: Method for comparison ('cosine', 'euclidean', 'dot').
            target: Target for comparison ('manufacturer' or 'model')
            plot: Whether to generate a plot.
            figsize: Figure size.
            
        Returns:
            DataFrame with comparison results and optional figure.
        """
        # Ensure we have embeddings
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(df)
            
        if not self.text_embeddings:
            print("No text embeddings found. Generating text embeddings...")
            self.generate_text_embeddings()
        
        # Get text embeddings based on target
        if target == 'manufacturer':
            text_embeds = {
                manufacturer: embedding.numpy()
                for manufacturer, embedding in self.manufacturer_embeddings.items()
            }
            true_label_key = 'manufacturer'
        else:  # 'model' or 'class'
            text_embeds = {
                class_name: embedding.numpy()
                for class_name, embedding in self.class_embeddings.items()
                if class_name in df['Classes'].values or (
                    'model' in df.columns and class_name in df['model'].values
                )
            }
            true_label_key = 'Classes' if 'Classes' in df.columns else 'model'
        
        # Calculate distances/similarities
        comparison_results = []
        
        for _, row in df.iterrows():
            filename = row['filename']
            if filename not in self.image_embeddings:
                continue
                
            img_embedding = self.image_embeddings[filename].numpy()
            true_label = row.get(true_label_key, None)
            
            # Calculate distances/similarities to each text embedding
            distances = {}
            for text_label, text_embedding in text_embeds.items():
                if method == 'cosine':
                    # Cosine similarity (higher is more similar)
                    dist = 1 - cosine(img_embedding, text_embedding)
                elif method == 'euclidean':
                    # Euclidean distance (lower is more similar)
                    dist = np.linalg.norm(img_embedding - text_embedding)
                elif method == 'dot':
                    # Dot product (higher is more similar)
                    dist = np.dot(img_embedding, text_embedding)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                distances[text_label] = dist
            
            # Find best match
            if method in ['cosine', 'dot']:
                # For cosine and dot, higher is better
                best_match = max(distances.items(), key=lambda x: x[1])[0]
            else:
                # For euclidean, lower is better
                best_match = min(distances.items(), key=lambda x: x[1])[0]
            
            comparison_results.append({
                'filename': filename,
                f'true_{true_label_key}': true_label,
                'best_match': best_match,
                'is_correct': best_match == true_label if true_label else None,
                **{f"{m}_score": score for m, score in distances.items()}
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(comparison_results)
        
        # Generate plot if requested
        fig = None
        if plot and not results_df.empty and target == 'manufacturer':
            fig, ax = plt.subplots(figsize=figsize)
            
            # Group by manufacturer or model
            categories = results_df[f'true_{true_label_key}'].unique()
            
            # Limit to plotting only Boeing, Airbus and Unknown if target is manufacturer
            if target == 'manufacturer':
                plot_categories = [c for c in categories if c in ["Boeing", "Airbus", "Unknown"]]
            else:
                # For model-level, limit to a reasonable number
                plot_categories = categories[:10] if len(categories) > 10 else categories
            
            for category in plot_categories:
                if pd.isna(category):
                    continue
                    
                cat_df = results_df[results_df[f'true_{true_label_key}'] == category]
                
                # For manufacturer target, compare Boeing to Airbus
                if target == 'manufacturer':
                    # Get scores for this manufacturer's samples
                    x_scores = cat_df['Boeing_score'] if 'Boeing_score' in cat_df else cat_df['Boeing_similarity']
                    y_scores = cat_df['Airbus_score'] if 'Airbus_score' in cat_df else cat_df['Airbus_similarity']
                    
                    # Plot as scatter
                    ax.scatter(
                        x_scores, 
                        y_scores, 
                        label=category,
                        alpha=0.7
                    )
                    
                    ax.set_xlabel('Similarity to Boeing')
                    ax.set_ylabel('Similarity to Airbus')
                else:
                    # For model target, this is too complex to visualize effectively in 2D
                    # Instead, plot the score for true category vs best non-true
                    pass
            
            # Add diagonal line
            if target == 'manufacturer':
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])
                ]
                ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal similarity')
                
            # Add labels and legend
            ax.set_title(f'Image-Text Embedding Comparison ({method})')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
        
        return results_df, fig
    
    def visualize_embeddings_umap(
        self, 
        df: pd.DataFrame,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        figsize: Tuple[int, int] = (12, 10),
        color_by: str = 'manufacturer',  # 'manufacturer' or 'Classes'
        title: str = None
    ) -> plt.Figure:
        """
        Visualize embeddings using UMAP.
        
        Args:
            df: DataFrame with image data.
            n_neighbors: UMAP n_neighbors parameter.
            min_dist: UMAP min_dist parameter.
            n_components: Number of UMAP dimensions.
            figsize: Figure size.
            color_by: Column to use for coloring ('manufacturer' or 'Classes').
            title: Custom title for the plot.
            
        Returns:
            Matplotlib figure with visualization.
        """
        # Ensure we have embeddings
        if not self.image_embeddings:
            print("No image embeddings found. Generating embeddings...")
            self.generate_image_embeddings(df)
            
        # Get embeddings as numpy array
        embeddings = []
        filenames = []
        labels = []
        
        # Make sure the color_by column exists
        if color_by not in df.columns and color_by == 'manufacturer' and 'Classes' in df.columns:
            # Create manufacturer column from Classes
            df['manufacturer'] = df['Classes'].apply(self.get_manufacturer)
            print("Created manufacturer column from Classes for visualization")
        elif color_by not in df.columns and color_by == 'Classes' and 'model' in df.columns:
            # Use model column instead
            color_by = 'model'
        
        # Get embeddings and labels
        for _, row in df.iterrows():
            filename = row['filename']
            if filename in self.image_embeddings:
                embeddings.append(self.image_embeddings[filename].cpu().numpy())
                filenames.append(filename)
                
                # Get label based on color_by
                if color_by in row:
                    labels.append(row[color_by])
                else:
                    labels.append("Unknown")
        
        if not embeddings:
            raise ValueError("No embeddings found for the given data.")
        
        X = np.array(embeddings)
        
        # Apply UMAP
        print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=42
        )
        X_umap = reducer.fit_transform(X)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique labels and assign colors
        unique_labels = sorted(set(labels))
        
        # Limit the number of categories to plot (if too many)
        if len(unique_labels) > 20 and color_by == 'Classes':
            # If too many classes, focus on Boeing and Airbus models plus Unknown
            boeing_models = [l for l in unique_labels if 'Boeing' in l or l.startswith('7')]
            airbus_models = [l for l in unique_labels if 'Airbus' in l or l.startswith('A3')]
            other_models = [l for l in unique_labels if l not in boeing_models and l not in airbus_models]
            
            # Get top models by count
            label_counts = pd.Series(labels).value_counts()
            top_boeing = [m for m in boeing_models if m in label_counts.nlargest(5).index]
            top_airbus = [m for m in airbus_models if m in label_counts.nlargest(5).index]
            top_other = [m for m in other_models if m in label_counts.nlargest(3).index]
            
            plot_labels = top_boeing + top_airbus + top_other + ["Unknown"]
            legend_title = f"Top Aircraft Models (from {len(unique_labels)} total)"
        else:
            plot_labels = unique_labels
            legend_title = color_by
        
        # Create colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, len(plot_labels)))
        color_dict = {label: colors[i] for i, label in enumerate(plot_labels)}
        
        # Plot points in groups
        for label in plot_labels:
            mask = np.array(labels) == label
            if np.any(mask):  # Only plot if we have points with this label
                ax.scatter(
                    X_umap[mask, 0], 
                    X_umap[mask, 1],
                    label=label,
                    color=color_dict.get(label, 'gray'),
                    alpha=0.7,
                    edgecolors='none'
                )
        
        # Add other points as "Other" if we limited the labels
        if len(plot_labels) < len(unique_labels):
            other_mask = ~np.array([label in plot_labels for label in labels])
            if np.any(other_mask):
                ax.scatter(
                    X_umap[other_mask, 0],
                    X_umap[other_mask, 1],
                    label="Other Models",
                    color='gray',
                    alpha=0.4,
                    edgecolors='none'
                )
                
        # Add text embeddings if available and using 2D UMAP
        if n_components == 2 and self.text_embeddings:
            # Get relevant text embeddings
            if color_by == 'manufacturer':
                text_embs = self.manufacturer_embeddings
            else:
                # Only add text embeddings for plotted models
                text_embs = {k: v for k, v in self.class_embeddings.items() if k in plot_labels}
                
            # Only add text embedding visualization if we have a reasonable number
            if len(text_embs) <= len(plot_labels) + 5:
                text_embeddings = np.array([emb.cpu().numpy() for emb in text_embs.values()])
                text_labels = list(text_embs.keys())
                
                # Project text embeddings with UMAP
                if len(text_embeddings) > 0:
                    # Map text embeddings through the same UMAP transformation
                    text_umap = reducer.transform(text_embeddings)
                    
                    # Plot text embeddings
                    ax.scatter(
                        text_umap[:, 0], 
                        text_umap[:, 1],
                        marker='X',
                        s=100,
                        color='black',
                        alpha=1.0,
                        edgecolors='white',
                        label='Text Prompts'
                    )
                    
                    # Add text labels
                    for i, label in enumerate(text_labels):
                        ax.annotate(
                            label,
                            xy=(text_umap[i, 0], text_umap[i, 1]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                        )
                
        # Add labels, legend, and title
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        
        if title:
            ax.set_title(title)
        elif color_by == 'manufacturer':
            ax.set_title('Aircraft Embeddings by Manufacturer (UMAP)')
        else:
            ax.set_title('Aircraft Embeddings by Model (UMAP)')
            
        # Handle legend - if too many classes, create a separate legend
        if len(plot_labels) > 10:
            # Put legend outside the main plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=legend_title)
        else:
            ax.legend(title=legend_title)
        
        plt.tight_layout()
        return fig
    
    def visualize_confusion_matrix(
        self, 
        confusion_mat: np.ndarray,
        classes: List[str],
        figsize: Tuple[int, int] = (10, 8),
        normalize: bool = True,
        title: str = 'Confusion Matrix'
    ) -> plt.Figure:
        """
        Visualize confusion matrix.
        
        Args:
            confusion_mat: Confusion matrix array.
            classes: List of class names.
            figsize: Figure size.
            normalize: Whether to normalize the confusion matrix.
            title: Plot title.
            
        Returns:
            Matplotlib figure with visualization.
        """
        if normalize:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot confusion matrix
        im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        # Add labels
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        
        # Add values
        thresh = confusion_mat.max() / 2.
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                ax.text(j, i, format(confusion_mat[i, j], fmt),
                        ha="center", va="center",
                        color="white" if confusion_mat[i, j] > thresh else "black")
                
        plt.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
        return fig
    
    def visualize_roc_curve(
        self, 
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        figsize: Tuple[int, int] = (8, 8),
        title: str = 'Receiver Operating Characteristic'
    ) -> plt.Figure:
        """
        Visualize ROC curve for OOD detection.
        
        Args:
            fpr: False positive rates.
            tpr: True positive rates.
            roc_auc: Area under ROC curve.
            figsize: Figure size.
            title: Plot title.
            
        Returns:
            Matplotlib figure with visualization.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        return fig
    
    def analyze_by_manufacturer(
        self, 
        results_df: pd.DataFrame,
        metric: str = 'similarity',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Analyze and visualize performance by manufacturer.
        
        Args:
            results_df: DataFrame with classification or OOD results.
            metric: Metric to analyze ('similarity', 'accuracy', or 'ood').
            figsize: Figure size.
            
        Returns:
            Matplotlib figure with visualization.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by manufacturer
        grouped = results_df.groupby('true_manufacturer')
        
        if metric == 'similarity':
            # Calculate average similarity by manufacturer
            if 'max_similarity' in results_df.columns:
                avg_similarity = grouped['max_similarity'].mean()
                std_similarity = grouped['max_similarity'].std()
            else:
                # Try with mean_distance (inversed for similarity)
                avg_similarity = 1 - grouped['mean_distance'].mean()
                std_similarity = grouped['mean_distance'].std()
            
            # Sort by average similarity
            avg_similarity = avg_similarity.sort_values(ascending=False)
            std_similarity = std_similarity.reindex(avg_similarity.index)
            
            # Create bar chart
            x = np.arange(len(avg_similarity))
            ax.bar(x, avg_similarity, yerr=std_similarity, alpha=0.7)
            
            # Add manufacturer labels
            ax.set_xticks(x)
            ax.set_xticklabels(avg_similarity.index, rotation=45, ha='right')
            
            # Customize plot
            ax.set_ylabel('Average Similarity')
            ax.set_title('Average Similarity by Manufacturer')
            
        elif metric == 'accuracy':
            # Calculate accuracy by manufacturer
            accuracies = []
            for name, group in grouped:
                correct = (group['predicted'] == group['true_manufacturer']).mean()
                count = len(group)
                accuracies.append({'manufacturer': name, 'accuracy': correct, 'count': count})
            
            # Convert to DataFrame and sort
            acc_df = pd.DataFrame(accuracies).sort_values('accuracy', ascending=False)
            
            # Create bar chart
            x = np.arange(len(acc_df))
            ax.bar(x, acc_df['accuracy'], alpha=0.7)
            
            # Add manufacturer labels
            ax.set_xticks(x)
            ax.set_xticklabels(acc_df['manufacturer'], rotation=45, ha='right')
            
            # Customize plot
            ax.set_ylabel('Classification Accuracy')
            ax.set_title('Classification Accuracy by Manufacturer')
            ax.set_ylim(0, 1.05)
            
        elif metric == 'ood':
            # Calculate OOD detection performance by manufacturer
            ood_metrics = []
            for name, group in grouped:
                is_known = name in self.known_manufacturers
                if is_known:
                    # For known manufacturers, correct prediction is not OOD
                    if 'is_ood' in group.columns:
                        correct = (~group['is_ood']).mean()
                    elif 'is_rejected' in group.columns:
                        correct = (~group['is_rejected']).mean()
                    else:
                        correct = 0  # Default if column not found
                else:
                    # For unknown manufacturers, correct prediction is OOD
                    if 'is_ood' in group.columns:
                        correct = group['is_ood'].mean()
                    elif 'is_rejected' in group.columns:
                        correct = group['is_rejected'].mean()
                    else:
                        correct = 0  # Default if column not found
                    
                count = len(group)
                ood_metrics.append({
                    'manufacturer': name, 
                    'accuracy': correct, 
                    'count': count,
                    'is_known': is_known
                })
            
            # Convert to DataFrame and sort
            ood_df = pd.DataFrame(ood_metrics).sort_values(['is_known', 'accuracy'], ascending=[False, False])
            
            # Create bar chart with color coding
            x = np.arange(len(ood_df))
            bars = ax.bar(
                x, 
                ood_df['accuracy'], 
                alpha=0.7, 
                color=[('green' if is_known else 'red') for is_known in ood_df['is_known']]
            )
            
            # Add manufacturer labels
            ax.set_xticks(x)
            ax.set_xticklabels(ood_df['manufacturer'], rotation=45, ha='right')
            
            # Customize plot
            ax.set_ylabel('OOD Detection Accuracy')
            ax.set_title('OOD Detection Accuracy by Manufacturer')
            ax.set_ylim(0, 1.05)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.7, label='Known (Boeing/Airbus)'),
                Patch(facecolor='red', alpha=0.7, label='Unknown')
            ]
            ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        return fig