# Configuration file for Custom OCR Model Training
# This file contains all the configuration parameters for training and evaluation

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data configuration
DATA_CONFIG = {
    # Data directories
    "train_dir": DATA_DIR / "train",
    "val_dir": DATA_DIR / "val", 
    "test_dir": DATA_DIR / "test",
    
    # CSV files
    "train_csv": DATA_DIR / "train" / "dataset.csv",
    "val_csv": DATA_DIR / "val" / "dataset.csv",
    "test_csv": DATA_DIR / "test" / "dataset.csv",
    
    # Data generation
    "num_train_samples": 1000,
    "num_val_samples": 200,
    "num_test_samples": 200,
    
    # Image properties
    "img_height": 32,
    "img_width": 128,
    "img_channels": 1,  # Grayscale
}

# Model configuration
MODEL_CONFIG = {
    # Architecture parameters
    "img_height": 32,
    "img_width": 128,
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
    "dropout": 0.2,
    
    # Character set (will be dynamically determined from data)
    "characters": None,  # Auto-detected from training data
    "max_text_length": 25,  # Maximum expected text length
    
    # Model saving
    "model_name": "custom_ocr_model",
    "save_best_only": True,
}

# Training configuration
TRAINING_CONFIG = {
    # Training parameters
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "optimizer": "adam",  # Options: "adam", "sgd", "rmsprop"
    
    # Learning rate scheduling
    "use_scheduler": True,
    "scheduler_type": "step",  # Options: "step", "cosine", "plateau"
    "scheduler_step_size": 20,
    "scheduler_gamma": 0.5,
    
    # Early stopping
    "early_stopping": True,
    "patience": 15,
    "min_delta": 0.001,
    
    # Gradient clipping
    "gradient_clipping": True,
    "max_grad_norm": 1.0,
    
    # Data loading
    "num_workers": 4,
    "pin_memory": True,
    "shuffle": True,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    # Image augmentations (using Albumentations)
    "use_augmentation": True,
    "augmentation_prob": 0.5,
    
    # Rotation
    "rotation_limit": 5,  # degrees
    
    # Perspective transform
    "perspective_scale": 0.05,
    
    # Gaussian noise
    "noise_var_limit": (10, 50),
    
    # Blur
    "blur_limit": 3,
    
    # Brightness/Contrast
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    
    # Elastic transform
    "elastic_alpha": 1,
    "elastic_sigma": 50,
    "elastic_alpha_affine": 50,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    # Metrics to calculate
    "calculate_cer": True,
    "calculate_wer": True,
    "calculate_bleu": True,
    "calculate_accuracy": True,
    
    # Confidence thresholds
    "confidence_threshold": 0.5,
    "high_confidence_threshold": 0.8,
    
    # Visualization
    "save_predictions": True,
    "num_samples_to_visualize": 20,
    "save_attention_maps": False,  # If attention is implemented
}

# Logging configuration
LOGGING_CONFIG = {
    # TensorBoard logging
    "use_tensorboard": True,
    "log_dir": LOGS_DIR,
    "log_every_n_steps": 10,
    
    # Console logging
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    
    # Checkpointing
    "save_checkpoint_every": 5,  # epochs
    "keep_last_n_checkpoints": 3,
}

# Hardware configuration
HARDWARE_CONFIG = {
    # Device settings
    "device": "auto",  # Options: "auto", "cuda", "cpu"
    "gpu_id": 0,
    
    # Memory settings
    "mixed_precision": True,  # Use automatic mixed precision
    "gradient_checkpointing": False,  # Save memory during training
    
    # Parallel processing
    "use_data_parallel": False,
    "use_distributed": False,
}

# Text generation configuration (for synthetic data)
TEXT_GENERATION_CONFIG = {
    # Character sets
    "include_letters": True,
    "include_numbers": True,
    "include_symbols": True,
    "include_spaces": True,
    
    # Custom character sets
    "custom_chars": "",  # Additional characters to include
    "exclude_chars": "",  # Characters to exclude
    
    # Text properties
    "min_text_length": 3,
    "max_text_length": 20,
    "avg_text_length": 8,
    
    # Language patterns
    "use_dictionary_words": True,
    "dictionary_ratio": 0.3,  # Ratio of dictionary words vs random text
    "common_phrases": [
        "hello world",
        "test image",
        "sample text",
        "deep learning",
        "computer vision",
        "optical character recognition"
    ],
}

# Font configuration (for synthetic data generation)
FONT_CONFIG = {
    # Font settings
    "font_sizes": [16, 18, 20, 22, 24],
    "font_families": [
        "Arial",
        "Times New Roman", 
        "Courier New",
        "Helvetica",
        "Calibri"
    ],
    
    # Text styling
    "use_bold": True,
    "use_italic": False,
    "bold_probability": 0.1,
    "italic_probability": 0.05,
    
    # Colors (grayscale values)
    "text_colors": [0, 50, 100, 150, 200],  # 0 = black, 255 = white
    "background_colors": [200, 220, 240, 255],
}

# Image generation configuration
IMAGE_GENERATION_CONFIG = {
    # Background
    "background_color": 255,  # White background
    "add_noise": True,
    "noise_intensity": 0.1,
    
    # Text positioning
    "text_margin": 5,  # pixels
    "center_text": True,
    "random_position": False,
    
    # Image effects
    "add_artifacts": True,
    "artifact_probability": 0.1,
    "jpeg_compression": False,
    "compression_quality": 85,
}

# Paths for external resources
RESOURCE_PATHS = {
    # Pretrained models (if using transfer learning)
    "pretrained_model_path": None,
    
    # Font files
    "fonts_dir": None,  # If None, will use system fonts
    
    # Dictionary files
    "dictionary_path": None,  # Path to word dictionary file
    
    # Background images
    "backgrounds_dir": None,  # For realistic background textures
}

# Model comparison configuration
COMPARISON_CONFIG = {
    # External OCR engines to compare with
    "compare_with_tesseract": True,
    "compare_with_easyocr": True,
    "compare_with_paddleocr": False,
    
    # Comparison settings
    "num_comparison_samples": 100,
    "save_comparison_results": True,
    "generate_comparison_report": True,
}

# Export configuration as a single dictionary
CONFIG = {
    "data": DATA_CONFIG,
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "augmentation": AUGMENTATION_CONFIG,
    "evaluation": EVALUATION_CONFIG,
    "logging": LOGGING_CONFIG,
    "hardware": HARDWARE_CONFIG,
    "text_generation": TEXT_GENERATION_CONFIG,
    "font": FONT_CONFIG,
    "image_generation": IMAGE_GENERATION_CONFIG,
    "resources": RESOURCE_PATHS,
    "comparison": COMPARISON_CONFIG,
}

# Helper functions
def get_config(section=None):
    """Get configuration section or entire config."""
    if section is None:
        return CONFIG
    return CONFIG.get(section, {})

def update_config(section, key, value):
    """Update a specific configuration value."""
    if section in CONFIG:
        CONFIG[section][key] = value
    else:
        raise ValueError(f"Configuration section '{section}' not found")

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        DATA_DIR,
        DATA_DIR / "train",
        DATA_DIR / "val", 
        DATA_DIR / "test",
        MODELS_DIR,
        MODELS_DIR / "checkpoints",
        RESULTS_DIR,
        LOGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Created project directories")

def print_config():
    """Print current configuration."""
    import json
    print("📋 Current Configuration:")
    print("=" * 50)
    for section, config_dict in CONFIG.items():
        print(f"\n[{section.upper()}]")
        for key, value in config_dict.items():
            if isinstance(value, Path):
                value = str(value)
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Create directories when script is run directly
    create_directories()
    print_config()
