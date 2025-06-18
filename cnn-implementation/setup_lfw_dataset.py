#!/usr/bin/env python3
"""
LFW Dataset Setup and Verification Script
=========================================

This script helps you download, extract, and verify the Labeled Faces in the Wild (LFW) dataset
for face recognition training with the CNN implementation.

Usage:
    python setup_lfw_dataset.py

Author: CNN Implementation Team
"""

import os
import sys
import urllib.request
import tarfile
import shutil
from pathlib import Path

def download_lfw_dataset(data_dir="./data"):
    """Download the LFW dataset."""
    print("LFW Dataset Setup")
    print("=" * 50)
    
    # URLs for LFW dataset
    lfw_urls = {
        "lfw": "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
        "lfw-deepfunneled": "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    }
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Data directory: {os.path.abspath(data_dir)}")
    print()
    
    # Check if already exists
    lfw_path = os.path.join(data_dir, "lfw")
    if os.path.exists(lfw_path):
        print(f"✅ LFW dataset already exists at {lfw_path}")
        return verify_lfw_dataset(lfw_path)
    
    print("Available LFW versions:")
    print("1. lfw - Original dataset (173MB)")
    print("2. lfw-deepfunneled - Better aligned faces (173MB)")
    print()
    
    choice = input("Choose version (1 or 2) [default: 1]: ").strip()
    if choice == "2":
        dataset_key = "lfw-deepfunneled"
    else:
        dataset_key = "lfw"
    
    url = lfw_urls[dataset_key]
    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    
    print(f"Downloading {dataset_key} from {url}")
    print("This may take a few minutes...")
    
    try:
        # Download with progress
        urllib.request.urlretrieve(url, filepath, reporthook=download_progress)
        print(f"\n✅ Downloaded: {filepath}")
        
        # Extract
        print("Extracting dataset...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        print(f"✅ Extracted to: {lfw_path}")
        
        # Clean up
        os.remove(filepath)
        print("✅ Cleaned up download file")
        
        return verify_lfw_dataset(lfw_path)
        
    except Exception as e:
        print(f"❌ Error downloading/extracting dataset: {e}")
        return False

def download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        print(f"\rProgress: {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")

def verify_lfw_dataset(lfw_path):
    """Verify the LFW dataset structure."""
    print("\nVerifying LFW dataset...")
    print("-" * 30)
    
    if not os.path.exists(lfw_path):
        print(f"❌ LFW directory not found: {lfw_path}")
        return False
    
    # Count people and images
    people_dirs = [d for d in os.listdir(lfw_path) 
                   if os.path.isdir(os.path.join(lfw_path, d))]
    
    if not people_dirs:
        print(f"❌ No person directories found in {lfw_path}")
        return False
    
    total_images = 0
    people_with_multiple_images = 0
    
    for person_dir in people_dirs[:10]:  # Check first 10 for verification
        person_path = os.path.join(lfw_path, person_dir)
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        if len(images) > 1:
            people_with_multiple_images += 1
    
    print(f"✅ Found {len(people_dirs)} people")
    print(f"✅ Checked {min(10, len(people_dirs))} people directories")
    print(f"✅ Found {total_images} images in first 10 people")
    print(f"✅ {people_with_multiple_images} people have multiple images")
    
    # Show some examples
    print("\nExample people:")
    for person in people_dirs[:5]:
        person_path = os.path.join(lfw_path, person)
        num_images = len([f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  • {person}: {num_images} images")
    
    return True

def test_lfw_loading():
    """Test loading LFW dataset with our data loader."""
    print("\nTesting LFW data loading...")
    print("-" * 30)
    
    try:
        # Add parent directory to path for imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from utils.data_loader import load_lfw_dataset
        
        print("Loading LFW dataset (this may take a moment)...")
        X_train, X_test, y_train, y_test = load_lfw_dataset(
            data_dir='./data/lfw',
            target_size=(112, 112),
            normalize=True,
            test_size=0.2
        )
        
        print(f"✅ Training set: {X_train.shape}")
        print(f"✅ Test set: {X_test.shape}")
        print(f"✅ Training labels: {y_train.shape}")
        print(f"✅ Test labels: {y_test.shape}")
        print(f"✅ Number of people: {len(set(y_train)) + len(set(y_test))}")
        print(f"✅ Image shape: {X_train[0].shape}")
        print(f"✅ Pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the CNN implementation directory")
        return False
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def create_face_demo():
    """Create a simple face recognition demo."""
    print("\nCreating face recognition demo...")
    print("-" * 30)
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from utils.data_loader import load_lfw_dataset, create_face_pairs
        from models.mobilefacenet import MobileFaceNet
        import numpy as np
        
        # Load small subset for demo
        print("Loading subset of LFW data...")
        X_train, X_test, y_train, y_test = load_lfw_dataset(
            data_dir='./data/lfw',
            target_size=(112, 112),
            normalize=True,
            test_size=0.8  # Use small training set for demo
        )
        
        # Create face pairs for verification
        print("Creating face pairs for verification...")
        pairs, pair_labels = create_face_pairs(X_train, y_train, num_pairs=100)
        
        print(f"✅ Created {len(pairs)} face pairs")
        print(f"✅ Same person pairs: {np.sum(pair_labels)}")
        print(f"✅ Different person pairs: {len(pair_labels) - np.sum(pair_labels)}")
        
        # Test MobileFaceNet
        print("Testing MobileFaceNet architecture...")
        mobilefacenet = MobileFaceNet(embedding_size=128)
        
        # Test forward pass
        sample_faces = X_train[:4]  # Test with 4 faces
        embeddings = mobilefacenet.forward(sample_faces)
        
        print(f"✅ MobileFaceNet input: {sample_faces.shape}")
        print(f"✅ MobileFaceNet output: {embeddings.shape}")
        print(f"✅ Embedding dimension: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in face demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("LFW Dataset Setup for CNN Implementation")
    print("=" * 50)
    print()
    
    print("This script will help you:")
    print("1. Download the LFW dataset (if not present)")
    print("2. Verify the dataset structure")
    print("3. Test loading with our data loader")
    print("4. Run a simple face recognition demo")
    print()
    
    # Check if PIL is available
    try:
        from PIL import Image
        print("✅ PIL (Pillow) is available for image loading")
    except ImportError:
        print("❌ PIL (Pillow) not found. Please install it:")
        print("   pip install Pillow")
        return
    
    print()
    response = input("Do you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    print()
    
    # Step 1: Download/verify dataset
    if not download_lfw_dataset():
        print("❌ Dataset setup failed")
        return
    
    # Step 2: Test loading
    if not test_lfw_loading():
        print("❌ Data loading test failed")
        return
    
    # Step 3: Face demo
    demo_response = input("\nDo you want to run the face recognition demo? (y/N): ")
    if demo_response.lower() == 'y':
        create_face_demo()
    
    print("\n" + "=" * 50)
    print("LFW Dataset Setup Complete!")
    print("=" * 50)
    print()
    print("You can now use the LFW dataset with:")
    print("• Face recognition training")
    print("• MobileFaceNet and FaceNet models")
    print("• Face verification tasks")
    print()
    print("Dataset location: ./data/lfw/")
    print("Use load_lfw_dataset() in utils/data_loader.py to load the data")

if __name__ == "__main__":
    main() 