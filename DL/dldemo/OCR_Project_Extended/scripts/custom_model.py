"""
Custom OCR Model Architecture - CNN + LSTM + CTC
===================================================

This module defines a deep learning architecture for Optical Character Recognition
using Convolutional Neural Networks (CNN) for feature extraction, 
Long Short-Term Memory (LSTM) networks for sequence modeling, 
and Connectionist Temporal Classification (CTC) loss for training.

Architecture Overview:
1. CNN Feature Extractor: Extracts visual features from input images
2. LSTM Sequence Processor: Models sequential dependencies in text
3. CTC Head: Handles variable-length sequences without explicit alignment

Author: OCR Project Extended
Date: July 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNNFeatureExtractor(nn.Module):
    """
    CNN backbone for extracting visual features from text images.
    
    Architecture inspired by VGG and ResNet designs, optimized for text recognition.
    """
    
    def __init__(self, input_channels=1):
        """
        Initialize CNN feature extractor.
        
        Args:
            input_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(CNNFeatureExtractor, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x128 -> 16x64
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x64 -> 8x32
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 8x32 -> 4x32
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 4x32 -> 2x32
        )
        
        # Final convolutional layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)  # 2x32 -> 1x31
        )
        
    def forward(self, x):
        """
        Forward pass through CNN feature extractor.
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Feature maps [batch_size, features, sequence_length]
        """
        # Apply convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Reshape for LSTM: [B, C, H, W] -> [B, C*H, W] -> [B, W, C*H]
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels * height, width)
        x = x.permute(0, 2, 1)  # [B, W, C*H] - sequence_length is width
        
        return x


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence modeling.
    
    Processes sequences in both forward and backward directions to capture
    contextual information from both past and future characters.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize bidirectional LSTM.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of LSTM hidden state
            output_size (int): Size of output features
        """
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        
    def forward(self, x):
        """
        Forward pass through bidirectional LSTM.
        
        Args:
            x (torch.Tensor): Input sequences [batch_size, sequence_length, input_size]
            
        Returns:
            torch.Tensor: Output sequences [batch_size, sequence_length, output_size]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply linear transformation
        output = self.linear(lstm_out)
        
        return output


class CRNNModel(nn.Module):
    """
    Complete CRNN (Convolutional Recurrent Neural Network) model for OCR.
    
    Combines CNN feature extraction, LSTM sequence modeling, and CTC loss
    for end-to-end text recognition from images.
    """
    
    def __init__(self, img_height=32, img_width=128, num_classes=80, 
                 cnn_output_height=1, cnn_output_width=31, lstm_hidden_size=256,
                 lstm_num_layers=2):
        """
        Initialize CRNN model.
        
        Args:
            img_height (int): Input image height
            img_width (int): Input image width
            num_classes (int): Number of character classes (including blank)
            cnn_output_height (int): Height of CNN output feature maps
            cnn_output_width (int): Width of CNN output feature maps
            lstm_hidden_size (int): LSTM hidden state size
            lstm_num_layers (int): Number of LSTM layers
        """
        super(CRNNModel, self).__init__()
        
        # Store configuration
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(input_channels=1)  # Grayscale images
        
        # Calculate CNN output feature size
        # After CNN: 512 channels * 1 height = 512 features per time step
        cnn_output_size = 512 * cnn_output_height
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            BidirectionalLSTM(cnn_output_size, lstm_hidden_size, lstm_hidden_size)
        )
        
        # Additional LSTM layers
        for _ in range(lstm_num_layers - 1):
            self.lstm_layers.append(
                BidirectionalLSTM(lstm_hidden_size, lstm_hidden_size, lstm_hidden_size)
            )
        
        # Final classification layer
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for param in module.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
        
    def forward(self, x):
        """
        Forward pass through complete CRNN model.
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Character probabilities [batch_size, sequence_length, num_classes]
        """
        # CNN feature extraction
        cnn_features = self.cnn(x)  # [B, W, C*H]
        
        # LSTM sequence processing
        lstm_out = cnn_features
        for lstm_layer in self.lstm_layers:
            lstm_out = lstm_layer(lstm_out)
        
        # Final classification
        output = self.classifier(lstm_out)  # [B, W, num_classes]
        
        # Apply log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def predict(self, x):
        """
        Make predictions without gradients (for inference).
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Predicted character indices [batch_size, sequence_length]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            # Get most likely character at each time step
            _, predicted = torch.max(output, dim=2)
        return predicted


def create_model(num_classes, img_height=32, img_width=128, lstm_hidden_size=256, 
                 lstm_num_layers=2):
    """
    Factory function to create CRNN model with specified parameters.
    
    Args:
        num_classes (int): Number of character classes (including blank)
        img_height (int): Input image height
        img_width (int): Input image width
        lstm_hidden_size (int): LSTM hidden state size
        lstm_num_layers (int): Number of LSTM layers
        
    Returns:
        CRNNModel: Configured CRNN model
    """
    model = CRNNModel(
        img_height=img_height,
        img_width=img_width,
        num_classes=num_classes,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers
    )
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage and model testing
    print("Testing CRNN Model Architecture")
    print("=" * 40)
    
    # Create model with example parameters
    num_classes = 80  # 26 letters + 10 digits + special chars + blank
    model = create_model(num_classes=num_classes)
    
    # Print model information
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test with example input
    batch_size = 4
    img_height = 32
    img_width = 128
    
    # Create random input tensor (simulating batch of images)
    x = torch.randn(batch_size, 1, img_height, img_width)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected: [batch_size={batch_size}, sequence_length≈31, num_classes={num_classes}]")
    
    # Test prediction function
    predictions = model.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    
    print("\n✅ Model architecture test completed successfully!")
