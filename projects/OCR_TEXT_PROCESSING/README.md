# OCR Project (Image Text Extraction)

This project extracts text from images using Pytesseract and EasyOCR.

## 📂 Folder Structure
- `images/` → Stores input images
- `data/imagedataset.csv` → Contains image filenames
- `results/extracted_texts/` → Stores extracted text files
- `ocr_extraction.ipynb` → Jupyter Notebook for testing and  for batch processing

## 🚀 Usage
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

2. Run Jupyter Notebook:  
   ```bash
   jupyter notebook ocr_extraction.ipynb
   ```

## ⚙️ Requirements
- Python 3.8+
- Tesseract-OCR (if using Pytesseract)
- EasyOCR (for alternative processing)
