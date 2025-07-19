"""
Training Script for Custom OCR Model
=====================================

This script handles the complete training pipeline for the custom OCR model,
including data loading, model training, validation, and model saving.

Features:
- CTC loss training
- Learning rate scheduling
- Model checkpointing
- Training metrics tracking
- Validation evaluation
- TensorBoard logging

Usage:
    python train.py --config config.yaml
    or
    python train.py (uses default parameters)

Author: OCR Project Extended
Date: July 2025
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.custom_model import create_model, count_parameters
from utils.dataset import CharacterMapping, create_dataloaders
from utils.metrics import calculate_cer, calculate_wer, calculate_accuracy


class OCRTrainer:
    """
    Main training class for OCR model.
    
    Handles all aspects of training including data loading, model optimization,
    validation, and checkpointing.
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config (dict): Training configuration parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.create_directories()
        
        # Initialize character mapping
        self.char_mapping = CharacterMapping()
        
        # Create model
        self.model = self.create_model()
        
        # Create data loaders
        self.train_loader, self.val_loader = self.create_data_loaders()
        
        # Initialize optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Initialize loss function
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        # Initialize metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Initialize TensorBoard logging
        self.writer = SummaryWriter(log_dir=self.config['log_dir'])
        
        # Save configuration
        self.save_config()
    
    def create_directories(self):
        """Create necessary output directories."""
        directories = [
            self.config['checkpoint_dir'],
            self.config['log_dir'],
            self.config['output_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_model(self):
        """Create and initialize the OCR model."""
        model = create_model(
            num_classes=self.char_mapping.num_classes,
            img_height=self.config['img_height'],
            img_width=self.config['img_width'],
            lstm_hidden_size=self.config['lstm_hidden_size'],
            lstm_num_layers=self.config['lstm_num_layers']
        )
        
        model = model.to(self.device)
        
        print(f"Model created with {count_parameters(model):,} trainable parameters")
        
        return model
    
    def create_data_loaders(self):
        """Create training and validation data loaders."""
        train_loader, val_loader = create_dataloaders(
            train_csv=self.config['train_csv'],
            val_csv=self.config['val_csv'],
            train_dir=self.config['train_dir'],
            val_dir=self.config['val_dir'],
            char_mapping=self.char_mapping,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            img_height=self.config['img_height'],
            img_width=self.config['img_width']
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_optimizer(self):
        """Create optimizer based on configuration."""
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['lr_step_size'],
                gamma=self.config['lr_gamma']
            )
        elif self.config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs']
            )
        else:
            scheduler = None
        
        return scheduler
    
    def save_config(self):
        """Save training configuration."""
        config_path = os.path.join(self.config['output_dir'], 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_lengths = batch['label_lengths'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)  # [batch_size, seq_len, num_classes]
            
            # Transpose for CTC loss: [seq_len, batch_size, num_classes]
            outputs = outputs.transpose(0, 1)
            
            # Calculate CTC loss
            loss = self.criterion(outputs, labels, input_lengths, label_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to TensorBoard
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Training/Loss', loss.item(), global_step)
            self.writer.add_scalar('Training/Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch):
        """
        Validate the model for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            tuple: (average_loss, accuracy, cer, wer)
        """
        self.model.eval()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                label_lengths = batch['label_lengths'].to(self.device)
                input_lengths = batch['input_lengths'].to(self.device)
                texts = batch['texts']
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                outputs_for_loss = outputs.transpose(0, 1)
                loss = self.criterion(outputs_for_loss, labels, input_lengths, label_lengths)
                epoch_loss += loss.item()
                
                # Get predictions
                predictions = self.model.predict(images)
                
                # Decode predictions and targets
                for i, pred in enumerate(predictions):
                    pred_text = self.char_mapping.ctc_decode(pred.cpu().numpy())
                    target_text = texts[i]
                    
                    all_predictions.append(pred_text)
                    all_targets.append(target_text)
        
        # Calculate metrics
        avg_loss = epoch_loss / len(self.val_loader)
        accuracy = calculate_accuracy(all_predictions, all_targets)
        cer = calculate_cer(all_predictions, all_targets)
        wer = calculate_wer(all_predictions, all_targets)
        
        # Log to TensorBoard
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Validation/CER', cer, epoch)
        self.writer.add_scalar('Validation/WER', wer, epoch)
        
        # Log some example predictions
        self.log_predictions(all_predictions[:5], all_targets[:5], epoch)
        
        return avg_loss, accuracy, cer, wer
    
    def log_predictions(self, predictions, targets, epoch):
        """Log example predictions to TensorBoard."""
        text_log = ""
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            text_log += f"Example {i+1}:\\n"
            text_log += f"  Target:     '{target}'\\n"
            text_log += f"  Prediction: '{pred}'\\n"
            text_log += f"  Match:      {'✓' if pred == target else '✗'}\\n\\n"
        
        self.writer.add_text('Validation/Predictions', text_log, epoch)
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'char_mapping': self.char_mapping
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with accuracy: {self.best_val_accuracy:.4f}")
    
    def train(self):
        """Run complete training loop."""
        print("🚀 Starting training...")
        print(f"Training for {self.config['num_epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss, val_accuracy, cer, wer = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Check if this is the best model
            is_best = val_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            print(f"\\nEpoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy:.4f}")
            print(f"  CER:        {cer:.4f}")
            print(f"  WER:        {wer:.4f}")
            print(f"  Best Acc:   {self.best_val_accuracy:.4f}")
            print("-" * 50)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\\n🎉 Training completed in {total_time/3600:.2f} hours")
        print(f"📊 Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()


def get_default_config():
    """Get default training configuration."""
    return {
        # Data paths
        'train_csv': 'data/train/dataset.csv',
        'val_csv': 'data/val/dataset.csv',
        'train_dir': 'data/train',
        'val_dir': 'data/val',
        
        # Model parameters
        'img_height': 32,
        'img_width': 128,
        'lstm_hidden_size': 256,
        'lstm_num_layers': 2,
        
        # Training parameters
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'scheduler': 'step',
        'lr_step_size': 30,
        'lr_gamma': 0.1,
        
        # Other parameters
        'num_workers': 4,
        'save_every': 10,
        
        # Output directories
        'checkpoint_dir': 'models/checkpoints',
        'log_dir': 'models/logs',
        'output_dir': 'models/outputs'
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Custom OCR Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
        print("Using default configuration")
    
    # Create trainer
    trainer = OCRTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.best_val_accuracy = checkpoint['best_val_accuracy']
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
