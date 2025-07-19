# OCR Project - Optical Character Recognition

This project demonstrates how to extract text from images using Python OCR libraries (Pytesseract and EasyOCR) in a Jupyter Notebook environment.

## 📁 Project Structure

```
OCR_Project/
│── images/                    # Folder containing images to process
│── data/
│   ├── imagedataset.csv      # CSV file with id,imagename columns
│── notebooks/
│   ├── ocr_extraction.ipynb  # Jupyter Notebook for testing OCR
│── results/
│   ├── extracted_texts/      # Folder to store extracted text files
│── scripts/
│   ├── process_images.py     # Batch processing script
│── requirements.txt          # List of required libraries
│── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Tesseract OCR** (for Windows users):
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install and add to system PATH, or note the installation path

### Installation

1. **Clone or download this project**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Windows users - Configure Tesseract path:**
   - If Tesseract is not in your PATH, you'll need to specify the path in the code
   - Common installation path: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`

## 📖 Usage

### Option 1: Interactive Notebook (Recommended for beginners)

1. **Open VSCode and navigate to the project folder**
2. **Open the notebook:**
   ```
   notebooks/ocr_extraction.ipynb
   ```
3. **Run cells step by step** to understand the OCR process
4. **Test with single images** before processing in batch

### Option 2: Batch Processing Script

1. **Add your images** to the `images/` folder
2. **Update the CSV file** `data/imagedataset.csv` with your image names
3. **Run the batch processing script:**
   ```bash
   python scripts/process_images.py
   ```

## 📋 Features

- ✅ **Pytesseract OCR** - Traditional OCR with Tesseract engine
- ✅ **EasyOCR** - Modern deep learning-based OCR
- ✅ **Batch processing** - Process multiple images automatically
- ✅ **CSV-based management** - Organize images using CSV file
- ✅ **Text file output** - Save extracted text to individual files
- ✅ **Error handling** - Robust error handling for missing files
- ✅ **Progress tracking** - Visual progress bars for batch processing

## 🔧 Configuration

### Tesseract Configuration (Windows)

If you encounter "tesseract not found" error, add this line to your code:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- GIF (.gif)

## 📊 Output

- **Text files**: Extracted text saved in `results/extracted_texts/`
- **Naming convention**: `{image_id}_{imagename}_extracted.txt`
- **Processing logs**: Console output showing progress and any errors

## 🆘 Troubleshooting

### Common Issues:

1. **"tesseract not found"**
   - Install Tesseract OCR and add to PATH
   - Or specify the path in code (see Configuration section)

2. **"No such file or directory"**
   - Check that image files exist in the `images/` folder
   - Verify image names in the CSV file match actual filenames

3. **Poor OCR results**
   - Try preprocessing images (resize, denoise, adjust contrast)
   - EasyOCR often works better for handwritten text
   - Ensure images have good quality and contrast

4. **Memory issues with EasyOCR**
   - EasyOCR uses GPU if available, CPU otherwise
   - For large batches, consider processing in smaller chunks

## 📚 Learning Resources

- [Pytesseract Documentation](https://pypi.org/project/pytesseract/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying libraries.
