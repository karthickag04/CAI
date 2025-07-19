"""
Prediction/Inference Script for Custom OCR Model
================================================

This script loads a trained OCR model and performs inference on new images.
Supports single image prediction and batch processing.

Features:
- Load trained model from checkpoint
- Single image inference
- Batch image processing
- Confidence scores
- Visualization of results
- Export predictions to various formats

Usage:
    python predict.py --model models/best_model.pth --image path/to/image.jpg
    python predict.py --model models/best_model.pth --batch_dir path/to/images/
    python predict.py --model models/best_model.pth --csv path/to/test_data.csv

Author: OCR Project Extended
Date: July 2025
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.custom_model import create_model
from utils.dataset import CharacterMapping, OCRDataset
from utils.metrics import calculate_detailed_metrics, print_metrics_report
import albumentations as A
from albumentations.pytorch import ToTensorV2


class OCRPredictor:
    """
    OCR model predictor for inference on new images.
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path (str): Path to trained model checkpoint
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.config = self.checkpoint['config']
        self.char_mapping = self.checkpoint['char_mapping']
        
        # Create and load model
        self.model = create_model(
            num_classes=self.char_mapping.num_classes,
            img_height=self.config['img_height'],
            img_width=self.config['img_width'],
            lstm_hidden_size=self.config['lstm_hidden_size'],
            lstm_num_layers=self.config['lstm_num_layers']
        )
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup image preprocessing
        self.transforms = A.Compose([
            A.Resize(self.config['img_height'], self.config['img_width']),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"📊 Model trained for {self.checkpoint['epoch']} epochs")
        print(f"🎯 Best validation accuracy: {self.checkpoint['best_val_accuracy']:.4f}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        image = np.array(image)
        
        # Apply transforms
        transformed = self.transforms(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict_single(self, image_path, return_confidence=False):
        """
        Predict text from a single image.
        
        Args:
            image_path (str): Path to image file
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            str or tuple: Predicted text, optionally with confidence
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)  # [1, seq_len, num_classes]
            
            if return_confidence:
                # Calculate average confidence (probability of predicted characters)
                probabilities = torch.softmax(outputs, dim=2)
                predictions = torch.argmax(outputs, dim=2)
                
                # Get confidence for each predicted character
                confidences = []
                for i in range(predictions.shape[1]):
                    pred_char_idx = predictions[0, i].item()
                    char_confidence = probabilities[0, i, pred_char_idx].item()
                    confidences.append(char_confidence)
                
                avg_confidence = np.mean(confidences)
            
            # Decode prediction
            predicted_indices = torch.argmax(outputs, dim=2)[0].cpu().numpy()
            predicted_text = self.char_mapping.ctc_decode(predicted_indices)
        
        if return_confidence:
            return predicted_text, avg_confidence
        else:
            return predicted_text
    
    def predict_batch(self, image_paths, batch_size=32, show_progress=True):
        """
        Predict text from multiple images.
        
        Args:
            image_paths (list): List of image file paths
            batch_size (int): Batch size for processing
            show_progress (bool): Whether to show progress bar
            
        Returns:
            list: List of predicted texts
        """
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), 
                     desc="Processing batches", disable=not show_progress):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
                    batch_tensors.append(torch.zeros(1, 1, self.config['img_height'], self.config['img_width']))
            
            # Stack batch
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                batch_predictions = torch.argmax(outputs, dim=2).cpu().numpy()
                
                # Decode each prediction
                for pred_indices in batch_predictions:
                    pred_text = self.char_mapping.ctc_decode(pred_indices)
                    predictions.append(pred_text)
        
        return predictions
    
    def predict_from_csv(self, csv_path, image_dir, output_path=None):
        """
        Predict text from images listed in CSV file.
        
        Args:
            csv_path (str): Path to CSV file with image names
            image_dir (str): Directory containing images
            output_path (str): Path to save predictions CSV
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Get image paths
        image_paths = [os.path.join(image_dir, img_name) for img_name in df['imagename']]
        
        # Get predictions
        predictions = self.predict_batch(image_paths)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        # If ground truth labels are available, calculate metrics
        if 'label' in df.columns:
            targets = df['label'].tolist()
            metrics = calculate_detailed_metrics(predictions, targets)
            
            print_metrics_report(metrics, "Batch Prediction Results")
            
            # Add individual metrics to dataframe
            df['correct'] = [pred.lower().strip() == target.lower().strip() 
                           for pred, target in zip(predictions, targets)]
        
        # Save results if output path specified
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"💾 Results saved to {output_path}")
        
        return df
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize image with prediction overlay.
        
        Args:
            image_path (str): Path to image file
            save_path (str): Path to save visualization
        """
        # Get prediction with confidence
        prediction, confidence = self.predict_single(image_path, return_confidence=True)
        
        # Load original image
        image = Image.open(image_path)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.imshow(image, cmap='gray' if image.mode == 'L' else None)
        ax.set_title(f"Prediction: '{prediction}' (Confidence: {confidence:.3f})", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"📊 Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return prediction, confidence
    
    def export_predictions(self, predictions, image_paths, output_format='json', output_path='predictions'):
        """
        Export predictions to various formats.
        
        Args:
            predictions (list): List of predicted texts
            image_paths (list): List of image paths
            output_format (str): Output format ('json', 'csv', 'txt')
            output_path (str): Output file path (without extension)
        """
        # Prepare data
        data = [
            {
                'image_path': path,
                'image_name': os.path.basename(path),
                'prediction': pred,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            for path, pred in zip(image_paths, predictions)
        ]
        
        if output_format == 'json':
            output_file = f"{output_path}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif output_format == 'csv':
            output_file = f"{output_path}.csv"
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
        
        elif output_format == 'txt':
            output_file = f"{output_path}.txt"
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(f"{item['image_name']}: {item['prediction']}\\n")
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        print(f"📁 Predictions exported to {output_file}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='OCR Model Prediction')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image', help='Path to single image for prediction')
    parser.add_argument('--batch_dir', help='Directory containing images for batch prediction')
    parser.add_argument('--csv', help='CSV file with image names for prediction')
    parser.add_argument('--image_dir', help='Directory containing images (for CSV mode)')
    parser.add_argument('--output', help='Output path for results')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--format', default='csv', choices=['csv', 'json', 'txt'], 
                       help='Output format for batch predictions')
    parser.add_argument('--device', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = OCRPredictor(args.model, args.device)
    
    if args.image:
        # Single image prediction
        print(f"\\n🔍 Predicting text from: {args.image}")
        
        if args.visualize:
            prediction, confidence = predictor.visualize_prediction(
                args.image, 
                args.output if args.output else None
            )
            print(f"📝 Prediction: '{prediction}' (Confidence: {confidence:.3f})")
        else:
            prediction = predictor.predict_single(args.image)
            print(f"📝 Prediction: '{prediction}'")
    
    elif args.batch_dir:
        # Batch directory prediction
        print(f"\\n📁 Processing images from directory: {args.batch_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(Path(args.batch_dir).glob(f"*{ext}"))
            image_paths.extend(Path(args.batch_dir).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        print(f"Found {len(image_paths)} images")
        
        if image_paths:
            predictions = predictor.predict_batch(image_paths)
            
            # Export results
            output_path = args.output if args.output else 'batch_predictions'
            predictor.export_predictions(predictions, image_paths, args.format, output_path)
            
            # Show sample predictions
            print(f"\\n📋 Sample predictions:")
            for i, (path, pred) in enumerate(zip(image_paths[:5], predictions[:5])):
                print(f"  {os.path.basename(path)}: '{pred}'")
    
    elif args.csv:
        # CSV-based prediction
        if not args.image_dir:
            raise ValueError("--image_dir required when using --csv")
        
        print(f"\\n📊 Processing images from CSV: {args.csv}")
        
        output_path = args.output if args.output else 'csv_predictions.csv'
        results_df = predictor.predict_from_csv(args.csv, args.image_dir, output_path)
        
        print(f"\\n📋 Processed {len(results_df)} images")
        if 'correct' in results_df.columns:
            accuracy = results_df['correct'].mean()
            print(f"🎯 Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    else:
        print("❌ Please specify one of: --image, --batch_dir, or --csv")
        parser.print_help()


if __name__ == "__main__":
    main()
