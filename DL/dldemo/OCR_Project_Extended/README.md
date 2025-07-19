# Custom OCR Model Training Project

This project implements a custom deep learning-based OCR (Optical Character Recognition) system using PyTorch. It features a CNN-LSTM architecture with CTC (Connectionist Temporal Classification) loss for end-to-end text recognition.

## 🏗️ Architecture Overview

The OCR model consists of three main components:
1. **CNN Feature Extractor**: Extracts visual features from input images
2. **Bidirectional LSTM**: Models sequential dependencies in the extracted features
3. **CTC Head**: Handles variable-length sequences without requiring character-level alignment

## 📁 Project Structure

```
OCR_Project_Extended/
├── scripts/
│   ├── custom_model.py      # Model architecture definition
│   ├── train.py            # Training pipeline
│   ├── predict.py          # Inference and prediction
│   └── evaluate.py         # Model evaluation
├── utils/
│   ├── dataset.py          # Dataset handling and preprocessing
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Visualization utilities
├── notebooks/
│   ├── train_model.ipynb   # Interactive training notebook
│   └── evaluate_model.ipynb # Interactive evaluation notebook
├── data/
│   ├── train/              # Training data
│   ├── val/                # Validation data
│   └── test/               # Test data
├── models/
│   ├── checkpoints/        # Model checkpoints
│   └── logs/              # Training logs
├── results/               # Evaluation results and reports
├── config.py             # Configuration file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository (if using git)
git clone <repository-url>
cd OCR_Project_Extended

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

#### Option A: Use Sample Data Generator
```python
from utils.dataset import create_sample_dataset

# Create sample training data
create_sample_dataset("data/train", num_samples=1000)
create_sample_dataset("data/val", num_samples=200)
create_sample_dataset("data/test", num_samples=200)
```

#### Option B: Use Your Own Data
Organize your data in the following CSV format:
```csv
imagename,label
image_001.jpg,hello world
image_002.jpg,sample text
...
```

### 3. Training the Model

#### Using Python Script:
```bash
python scripts/train.py --config config.py --epochs 50 --batch-size 32
```

#### Using Jupyter Notebook:
Open `notebooks/train_model.ipynb` and run all cells for interactive training.

### 4. Model Evaluation

#### Using Python Script:
```bash
python scripts/evaluate.py --model models/checkpoints/best_model.pth --test-data data/test
```

#### Using Jupyter Notebook:
Open `notebooks/evaluate_model.ipynb` for comprehensive evaluation analysis.

### 5. Making Predictions

#### Single Image:
```bash
python scripts/predict.py --model models/checkpoints/best_model.pth --image path/to/image.jpg
```

#### Batch Processing:
```bash
python scripts/predict.py --model models/checkpoints/best_model.pth --input-dir path/to/images/ --output results/predictions.csv
```

#### Using Python API:
```python
from scripts.predict import OCRPredictor

predictor = OCRPredictor("models/checkpoints/best_model.pth")
result = predictor.predict_single("image.jpg")
print(f"Predicted text: {result['text']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📊 Configuration

The project uses a comprehensive configuration system defined in `config.py`. Key configuration sections include:

- **Data Configuration**: Dataset paths, image dimensions, augmentation settings
- **Model Configuration**: Architecture parameters, hidden sizes, layers
- **Training Configuration**: Learning rate, batch size, epochs, optimization settings
- **Evaluation Configuration**: Metrics, visualization options
- **Hardware Configuration**: GPU settings, memory optimization

### Example Configuration:
```python
from config import get_config

# Get training configuration
train_config = get_config('training')
print(f"Learning rate: {train_config['learning_rate']}")
print(f"Batch size: {train_config['batch_size']}")

# Update configuration
from config import update_config
update_config('training', 'learning_rate', 0.0005)
```

## 🎯 Model Performance

### Training Metrics
- **Architecture**: CNN + Bidirectional LSTM + CTC
- **Parameters**: ~2.5M trainable parameters
- **Input Size**: 32x128 grayscale images
- **Character Set**: Configurable (letters, numbers, symbols)

### Expected Performance (on sample data)
- **Accuracy**: 85-95% (depending on data complexity)
- **Character Error Rate (CER)**: 5-15%
- **Word Error Rate (WER)**: 10-25%
- **Inference Speed**: ~10-50ms per image (GPU)

## 📈 Evaluation Metrics

The project provides comprehensive evaluation metrics:

1. **Accuracy**: Exact string match percentage
2. **Character Error Rate (CER)**: Character-level edit distance
3. **Word Error Rate (WER)**: Word-level edit distance  
4. **BLEU Score**: Sequence similarity metric
5. **Confidence Analysis**: Model prediction confidence
6. **Performance by Text Length**: Accuracy across different text lengths

## 🔧 Advanced Usage

### Custom Model Architecture

Modify the model architecture in `scripts/custom_model.py`:

```python
def create_model(num_classes, img_height=32, img_width=128, 
                lstm_hidden_size=256, lstm_num_layers=2):
    # Customize CNN layers
    cnn_layers = [
        # Add your custom CNN layers
    ]
    
    # Customize LSTM layers
    lstm = nn.LSTM(
        input_size=cnn_output_size,
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers,
        bidirectional=True,
        batch_first=True
    )
```

### Custom Data Augmentation

Add custom augmentations in `utils/dataset.py`:

```python
def get_training_transforms():
    return A.Compose([
        A.Rotate(limit=5, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        # Add your custom augmentations
        A.Normalize(mean=[0.5], std=[0.5])
    ])
```

### Transfer Learning

Use a pretrained model as starting point:

```python
# Load pretrained weights
pretrained_model = torch.load("pretrained_model.pth")
model.load_state_dict(pretrained_model['model_state_dict'], strict=False)

# Freeze early layers
for param in model.cnn_feature_extractor.parameters():
    param.requires_grad = False
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Enable gradient checkpointing
   - Use mixed precision training

2. **CTC Loss becomes NaN**
   - Check input/target length compatibility
   - Ensure proper sequence length calculation
   - Verify character mapping consistency

3. **Poor Model Performance**
   - Increase training data size
   - Adjust learning rate
   - Add more data augmentation
   - Verify data quality and labels

4. **Slow Training**
   - Enable mixed precision training
   - Increase batch size (if memory allows)
   - Use data loading optimizations

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Dependencies

Core dependencies:
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- PIL (Pillow) >= 8.3.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0
- tensorboard >= 2.7.0

Optional dependencies for comparison:
- easyocr >= 1.6.0
- pytesseract >= 0.3.8

## 🔬 Experiments and Research

### Baseline Experiments
1. **Architecture Comparison**: CNN-LSTM vs CNN-Transformer vs CRNN
2. **Loss Function Analysis**: CTC vs Attention-based seq2seq
3. **Data Augmentation Impact**: Effect of different augmentation strategies
4. **Character Set Optimization**: Performance with different character sets

### Advanced Experiments
1. **Attention Mechanisms**: Add attention to improve long sequence handling
2. **Multi-scale Features**: Combine features from different CNN layers
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Domain Adaptation**: Fine-tune for specific document types

## 📊 Comparison with Other OCR Solutions

The project includes comparison tools with popular OCR engines:

| Method | Accuracy | Speed | Pros | Cons |
|--------|----------|-------|------|------|
| Custom CRNN | 85-95% | Fast | Customizable, End-to-end | Requires training data |
| EasyOCR | 80-95% | Medium | Pre-trained, Multi-language | Limited customization |
| Pytesseract | 70-90% | Fast | Mature, Configurable | Poor on complex layouts |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Look through existing GitHub issues
3. Create a new issue with detailed information
4. Include configuration, error messages, and system information

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Albumentations for data augmentation utilities
- Contributors to open-source OCR research and datasets
- Community feedback and contributions

## 🎓 Educational Resources

### Understanding OCR Components

1. **CNN Feature Extraction**
   - Convolutional layers extract visual patterns
   - Pooling layers reduce spatial dimensions
   - Batch normalization for training stability

2. **LSTM Sequence Modeling**
   - Bidirectional LSTM captures context from both directions
   - Hidden states encode sequence information
   - Dropout prevents overfitting

3. **CTC Loss Function**
   - Handles variable-length sequences
   - No need for character-level alignment
   - Allows multiple paths through sequence

### Training Best Practices

1. **Data Quality**
   - Ensure high-quality annotations
   - Balance text length distribution
   - Include diverse fonts and styles

2. **Hyperparameter Tuning**
   - Start with reasonable defaults
   - Use learning rate scheduling
   - Monitor validation metrics

3. **Regularization**
   - Apply dropout in LSTM layers
   - Use data augmentation
   - Implement early stopping

## 🚀 Future Improvements

### Planned Features
- [ ] Transformer-based architecture option
- [ ] Multi-language support
- [ ] Real-time video OCR
- [ ] Web interface for easy testing
- [ ] Docker containerization
- [ ] Model quantization for mobile deployment

### Research Directions
- [ ] Few-shot learning for new domains
- [ ] Adversarial training for robustness
- [ ] Neural architecture search
- [ ] Self-supervised pre-training

---

**Happy OCR Training! 🎉**

For the latest updates and discussions, check out our GitHub repository.
