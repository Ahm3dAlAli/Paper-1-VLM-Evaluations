import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np

# Function to set up Kaggle credentials from the desktop file
def set_kaggle_credentials(credentials_path):
    try:
        # Read the credentials file
        with open(credentials_path) as f:
            creds = json.load(f)
            # Set environment variables for Kaggle
            os.environ['KAGGLE_USERNAME'] = creds['username']
            os.environ['KAGGLE_KEY'] = creds['key']
        print("Credentials loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return False

# Function to download the dataset
def download_fgvc_aircraft_dataset(dataset_path="./data"):
    try:
        # Create the data directory if it doesn't exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            
        # Initialize and authenticate the Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download the FGVC-Aircraft dataset
        print(f"Downloading FGVC-Aircraft dataset to {dataset_path}...")
        api.dataset_download_files(
            'seryouxblaster764/fgvc-aircraft', 
            path=dataset_path,
            unzip=True
        )
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

# Function to explore the dataset structure
# Function to explore the dataset structure with more detailed output
def explore_dataset(dataset_path="./data"):
    try:
        print("\n=== DETAILED DATASET STRUCTURE ANALYSIS ===")
        
        # Dictionary to store important paths
        important_paths = {
            "image_directories": [],
            "annotation_files": [],
            "csv_files": [],
            "potential_data_dirs": []
        }
        
        # Walk through the directory and analyze structure
        print("\nDirectory structure:")
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(dataset_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            
            # Count image files in this directory
            image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                important_paths["image_directories"].append(root)
                print(f"{indent}    [FOUND {len(image_files)} IMAGES]")
            
            # Look for annotation files
            annotation_files = [f for f in files if f.endswith(('.txt', '.json', '.xml'))]
            if annotation_files:
                for f in annotation_files:
                    important_paths["annotation_files"].append(os.path.join(root, f))
                print(f"{indent}    [FOUND {len(annotation_files)} ANNOTATION FILES]")
            
            # Look for CSV files that might contain labels
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                for f in csv_files:
                    important_paths["csv_files"].append(os.path.join(root, f))
                print(f"{indent}    [FOUND {len(csv_files)} CSV FILES]")
            
            # Check if directory might be a data directory
            if "data" in root.lower() or "images" in root.lower() or "aircraft" in root.lower():
                important_paths["potential_data_dirs"].append(root)
        
        # Report important directories and files
        print("\n=== IMPORTANT PATHS FOR YOUR AIRCRAFT CLASSIFICATION CODE ===")
        
        # Image directories
        print("\nImage directories:")
        for i, path in enumerate(important_paths["image_directories"]):
            img_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {i+1}. {path} ({img_count} images)")
        
        # Annotation files
        print("\nAnnotation files:")
        if important_paths["annotation_files"]:
            for i, path in enumerate(important_paths["annotation_files"]):
                print(f"  {i+1}. {path}")
                # Peek at first few lines of annotation files
                try:
                    with open(path, 'r') as f:
                        first_lines = [next(f) for _ in range(3)]
                        print(f"     Preview: {first_lines}")
                except:
                    print("     (Could not read file)")
        else:
            print("  No annotation files found")
        
        # CSV files
        print("\nCSV files (potential label files):")
        if important_paths["csv_files"]:
            for i, path in enumerate(important_paths["csv_files"]):
                print(f"  {i+1}. {path}")
        else:
            print("  No CSV files found")
        
        # Recommended data directory
        print("\nRecommended data_dir for your code:")
        best_dir = None
        
        # First, look for directories containing both images and annotation files
        for dir_path in important_paths["potential_data_dirs"]:
            has_images = any(dir_path in img_dir for img_dir in important_paths["image_directories"])
            has_annotations = any(dir_path in anno_file for anno_file in important_paths["annotation_files"])
            
            if has_images and has_annotations:
                best_dir = dir_path
                break
        
        # If no ideal directory found, just recommend the first image directory's parent
        if not best_dir and important_paths["image_directories"]:
            best_dir = os.path.dirname(important_paths["image_directories"][0])
        
        if best_dir:
            print(f"  {best_dir}")
            print(f"\nSuggested setting for your code:")
            print(f"  --data_dir \"{best_dir}\"")
        else:
            print("  Could not determine best data directory")
        
        return important_paths
    except Exception as e:
        print(f"Error exploring dataset: {e}")
        return None

# Function to display sample images
def display_sample_images(dataset_path="./data", num_samples=4):
    try:
        # Find image files
        image_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print("No image files found in the dataset.")
            return False
        
        # Select random samples
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        print(f"\nDisplaying {len(samples)} sample images:")
        for img_path in samples:
            print(f"- {os.path.basename(img_path)}")
        
        print("\nTo view these images, run this script in an environment that supports matplotlib display.")
        
        # Code to display images (will work in Jupyter or with plt.show())
        fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
        if len(samples) == 1:
            axes = [axes]
        
        for i, img_path in enumerate(samples):
            img = Image.open(img_path)
            axes[i].imshow(np.array(img))
            axes[i].set_title(os.path.basename(img_path))
            axes[i].axis('off')
        
        plt.tight_layout()
        # plt.show()  # Uncomment this line if running in an environment that supports display
        
        return True
    except Exception as e:
        print(f"Error displaying sample images: {e}")
        return False

# Main function
def main():
    # Set paths
    credentials_path = r"C:\Users\ahal-\Desktop\kaggle.json"
    dataset_path = r"./data/fgvc-aircraft"
    
    # Set up Kaggle credentials
    if set_kaggle_credentials(credentials_path):
        # Download the dataset
        if download_fgvc_aircraft_dataset(dataset_path):
            # Explore the dataset
            explore_dataset(dataset_path)
            # Display sample images
            display_sample_images(dataset_path)

if __name__ == "__main__":
    main()