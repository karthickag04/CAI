"""
Dataset Utilities for Custom OCR Training
==========================================

This module provides dataset classes and utilities for loading and preprocessing
OCR training data. Supports various dataset formats and includes data augmentation.

Features:
- Custom dataset loading from CSV files
- Image preprocessing and normalization
- Data augmentation for training
- CTC-compatible label encoding
- Support for variable-length sequences

Author: OCR Project Extended
Date: July 2025
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import string


class CharacterMapping:
    """
    Character mapping utility for converting between characters and indices.
    
    Handles the conversion between text strings and numerical indices required
    for training neural networks with CTC loss.
    """
    
    def __init__(self, characters=None, include_blank=True):
        """
        Initialize character mapping.
        
        Args:
            characters (str): String of all possible characters
            include_blank (bool): Whether to include blank character for CTC
        """
        if characters is None:
            # Default character set: letters, digits, and common punctuation
            characters = string.ascii_lowercase + string.digits + " .,!?'-"
        
        self.characters = characters
        self.include_blank = include_blank
        
        # Create character to index mapping
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Blank character (for CTC) is typically index 0
        if include_blank:
            self.char_to_idx['<BLANK>'] = 0
            self.idx_to_char[0] = '<BLANK>'
            start_idx = 1
        else:
            start_idx = 0
        
        # Map each character to an index
        for i, char in enumerate(characters):
            self.char_to_idx[char] = i + start_idx
            self.idx_to_char[i + start_idx] = char
        
        self.num_classes = len(self.char_to_idx)
    
    def encode(self, text):
        """
        Convert text string to list of indices.
        
        Args:
            text (str): Input text string
            
        Returns:
            list: List of character indices
        """
        indices = []
        for char in text.lower():  # Convert to lowercase
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Skip unknown characters or handle as needed
                pass
        return indices
    
    def decode(self, indices):
        """
        Convert list of indices to text string.
        
        Args:
            indices (list): List of character indices
            
        Returns:
            str: Decoded text string
        """
        text = ""
        for idx in indices:
            if idx in self.idx_to_char and self.idx_to_char[idx] != '<BLANK>':
                text += self.idx_to_char[idx]
        return text
    
    def ctc_decode(self, indices):
        """
        Decode CTC output by removing blanks and repeated characters.
        
        Args:
            indices (list): Raw CTC output indices
            
        Returns:
            str: Decoded text string
        """
        # Remove consecutive duplicates and blanks
        decoded = []
        prev_idx = None
        
        for idx in indices:
            if idx != prev_idx and idx != 0:  # 0 is blank
                if idx in self.idx_to_char:
                    decoded.append(self.idx_to_char[idx])
            prev_idx = idx
        
        return ''.join(decoded)


class OCRDataset(Dataset):
    """
    PyTorch Dataset class for OCR training data.
    
    Loads images and corresponding text labels from CSV file,
    applies preprocessing and augmentation.
    """
    
    def __init__(self, csv_file, image_dir, char_mapping, 
                 img_height=32, img_width=128, 
                 transforms=None, is_training=True):
        """
        Initialize OCR dataset.
        
        Args:
            csv_file (str): Path to CSV file with image names and labels
            image_dir (str): Directory containing images
            char_mapping (CharacterMapping): Character mapping utility
            img_height (int): Target image height
            img_width (int): Target image width
            transforms (albumentations.Compose): Data augmentation transforms
            is_training (bool): Whether this is training data
        """
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.char_mapping = char_mapping
        self.img_height = img_height
        self.img_width = img_width
        self.transforms = transforms
        self.is_training = is_training
        
        # Load dataset
        self.data = pd.read_csv(csv_file)
        
        # Validate dataset
        self._validate_dataset()
        
        # Default transforms if none provided
        if self.transforms is None:
            self.transforms = self._get_default_transforms()
    
    def _validate_dataset(self):
        """Validate that all required columns exist in the dataset."""
        required_columns = ['imagename', 'label']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        print(f"Dataset loaded: {len(self.data)} samples")
        print(f"Sample labels: {self.data['label'].head().tolist()}")
    
    def _get_default_transforms(self):
        """Get default image transforms based on training/validation mode."""
        if self.is_training:
            # Training transforms with augmentation
            transforms = A.Compose([
                A.Resize(self.img_height, self.img_width),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.CLAHE(clip_limit=2.0, p=0.3),
                ], p=0.2),
                A.Rotate(limit=2, p=0.3),
                A.Perspective(scale=(0.02, 0.05), p=0.2),
                A.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
                ToTensorV2(),
            ])
        else:
            # Validation/test transforms without augmentation
            transforms = A.Compose([
                A.Resize(self.img_height, self.img_width),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        
        return transforms
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image_tensor, label_indices, label_length)
        """
        # Get image and label
        row = self.data.iloc[idx]
        image_name = row['imagename']
        label_text = row['label']
        
        # Load image
        image_path = os.path.join(self.image_dir, image_name)
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        # Encode label
        label_indices = self.char_mapping.encode(label_text)
        label_length = len(label_indices)
        
        return {
            'image': image,
            'label': torch.tensor(label_indices, dtype=torch.long),
            'label_length': torch.tensor(label_length, dtype=torch.long),
            'text': label_text  # Keep original text for reference
        }
    
    def _load_image(self, image_path):
        """
        Load and preprocess image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image
        """
        try:
            # Try loading with PIL first
            image = Image.open(image_path)
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array
            image = np.array(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return blank image as fallback
            return np.zeros((self.img_height, self.img_width), dtype=np.uint8)


def ctc_collate_fn(batch):
    """
    Collate function for CTC training.
    
    Handles variable-length sequences by padding and creating input/target lengths.
    
    Args:
        batch (list): List of dataset samples
        
    Returns:
        dict: Batched data with proper padding and lengths
    """
    images = []
    labels = []
    label_lengths = []
    texts = []
    
    for sample in batch:
        images.append(sample['image'])
        labels.extend(sample['label'].tolist())
        label_lengths.append(sample['label_length'])
        texts.append(sample['text'])
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Convert labels and lengths to tensors
    labels = torch.tensor(labels, dtype=torch.long)
    label_lengths = torch.stack(label_lengths, dim=0)
    
    # Input lengths (sequence length from CNN output)
    # This depends on your CNN architecture - adjust as needed
    input_lengths = torch.full((len(batch),), 31, dtype=torch.long)  # Assuming CNN outputs 31 time steps
    
    return {
        'images': images,
        'labels': labels,
        'label_lengths': label_lengths,
        'input_lengths': input_lengths,
        'texts': texts
    }


def create_dataloaders(train_csv, val_csv, train_dir, val_dir, 
                      char_mapping, batch_size=32, num_workers=4,
                      img_height=32, img_width=128):
    """
    Create training and validation data loaders.
    
    Args:
        train_csv (str): Path to training CSV file
        val_csv (str): Path to validation CSV file
        train_dir (str): Training images directory
        val_dir (str): Validation images directory
        char_mapping (CharacterMapping): Character mapping utility
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes
        img_height (int): Target image height
        img_width (int): Target image width
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = OCRDataset(
        csv_file=train_csv,
        image_dir=train_dir,
        char_mapping=char_mapping,
        img_height=img_height,
        img_width=img_width,
        is_training=True
    )
    
    val_dataset = OCRDataset(
        csv_file=val_csv,
        image_dir=val_dir,
        char_mapping=char_mapping,
        img_height=img_height,
        img_width=img_width,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_sample_dataset(output_dir, num_samples=100):
    """
    Create a sample dataset for testing purposes.
    
    Args:
        output_dir (str): Output directory for sample data
        num_samples (int): Number of sample images to create
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample texts
    texts = [
        "hello world", "machine learning", "deep learning", "computer vision",
        "artificial intelligence", "neural networks", "pytorch", "python",
        "data science", "text recognition", "optical character", "training data"
    ]
    
    data = []
    
    for i in range(num_samples):
        # Choose random text
        text = texts[i % len(texts)]
        
        # Create simple text image
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.1, 0.5, text, fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save image
        image_name = f"sample_{i:04d}.png"
        image_path = os.path.join(output_dir, image_name)
        plt.savefig(image_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Add to dataset
        data.append({
            'id': i,
            'imagename': image_name,
            'label': text
        })
    
    # Save CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Created sample dataset with {num_samples} images at {output_dir}")
    print(f"CSV file saved at {csv_path}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing OCR Dataset Utilities")
    print("=" * 40)
    
    # Test character mapping
    char_mapping = CharacterMapping()
    print(f"Character mapping created with {char_mapping.num_classes} classes")
    
    # Test encoding/decoding
    test_text = "hello world"
    encoded = char_mapping.encode(test_text)
    decoded = char_mapping.decode(encoded)
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Test CTC decoding
    ctc_output = [1, 1, 2, 0, 3, 3, 0, 4]  # Example CTC output
    ctc_decoded = char_mapping.ctc_decode(ctc_output)
    print(f"CTC output: {ctc_output}")
    print(f"CTC decoded: '{ctc_decoded}'")
    
    print("\n✅ Dataset utilities test completed successfully!")
