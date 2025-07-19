#!/usr/bin/env python3
"""
OCR Batch Processing Script

This script processes multiple images listed in a CSV file and extracts text
using both Pytesseract and EasyOCR libraries.

Author: OCR Project
Date: July 2025
"""

import os
import sys
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm

# Image processing libraries
from PIL import Image
import cv2

# OCR libraries
import pytesseract
import easyocr

# Add project root to path for potential imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class OCRProcessor:
    """Main class for batch OCR processing."""
    
    def __init__(self, project_root=None):
        """
        Initialize the OCR processor.
        
        Args:
            project_root (Path): Path to project root directory
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
            
        # Define folder paths
        self.images_folder = self.project_root / "images"
        self.data_folder = self.project_root / "data"
        self.results_folder = self.project_root / "results" / "extracted_texts"
        
        # Create results folder if it doesn't exist
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR readers
        self.easy_reader = None
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed_successfully': 0,
            'failed_processing': 0,
            'start_time': None,
            'end_time': None
        }
    
    def configure_tesseract(self):
        """Configure Tesseract path for Windows if needed."""
        try:
            # Test if Tesseract is accessible
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
            return True
        except:
            print("⚠️  Tesseract not found in PATH. Trying common Windows paths...")
            
            # Common Tesseract installation paths on Windows
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    try:
                        version = pytesseract.get_tesseract_version()
                        print(f"✅ Found Tesseract at: {path}")
                        print(f"✅ Tesseract version: {version}")
                        return True
                    except:
                        continue
            
            print("❌ Tesseract not found. Please install Tesseract OCR:")
            print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
    
    def initialize_easyocr(self):
        """Initialize EasyOCR reader."""
        try:
            print("🔄 Initializing EasyOCR reader...")
            self.easy_reader = easyocr.Reader(['en'])
            print("✅ EasyOCR reader initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize EasyOCR: {str(e)}")
            return False
    
    def extract_text_pytesseract(self, image_path):
        """
        Extract text from image using Pytesseract.
        
        Args:
            image_path (Path): Path to the image file
        
        Returns:
            str: Extracted text
        """
        try:
            # Open image using PIL
            image = Image.open(image_path)
            
            # Extract text using Pytesseract
            # You can customize OCR settings here
            text = pytesseract.image_to_string(image, lang='eng')
            
            return text.strip()
        
        except Exception as e:
            return f"Error processing image with Pytesseract: {str(e)}"
    
    def extract_text_easyocr(self, image_path):
        """
        Extract text from image using EasyOCR.
        
        Args:
            image_path (Path): Path to the image file
        
        Returns:
            str: Extracted text
        """
        try:
            if self.easy_reader is None:
                return "EasyOCR reader not initialized"
            
            # Extract text using EasyOCR
            results = self.easy_reader.readtext(str(image_path))
            
            # Combine all detected text
            text = ' '.join([result[1] for result in results])
            
            return text.strip()
        
        except Exception as e:
            return f"Error processing image with EasyOCR: {str(e)}"
    
    def preprocess_image(self, image_path):
        """
        Basic image preprocessing to improve OCR accuracy.
        
        Args:
            image_path (Path): Path to the image file
        
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
        
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def save_extracted_text(self, image_id, image_name, pytesseract_text, easyocr_text):
        """
        Save extracted text to a file.
        
        Args:
            image_id (int): Image ID from CSV
            image_name (str): Image filename
            pytesseract_text (str): Text extracted by Pytesseract
            easyocr_text (str): Text extracted by EasyOCR
        """
        # Create filename for the text file
        base_name = Path(image_name).stem  # Remove file extension
        output_filename = f"{image_id}_{base_name}_extracted.txt"
        output_path = self.results_folder / output_filename
        
        # Prepare content
        content = f"""OCR EXTRACTION RESULTS
========================
Image ID: {image_id}
Image Name: {image_name}
Extraction Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Processing Script: process_images.py

PYTESSERACT RESULTS:
--------------------
{pytesseract_text if pytesseract_text else 'No text detected'}

EASYOCR RESULTS:
----------------
{easyocr_text if easyocr_text else 'No text detected'}

PROCESSING NOTES:
-----------------
- Both Pytesseract and EasyOCR were used for comparison
- Results may vary based on image quality and content type
- For better results, consider image preprocessing

END OF EXTRACTION
=================="""
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"❌ Error saving {output_filename}: {str(e)}")
            return False
    
    def process_single_image(self, image_id, image_name):
        """
        Process a single image and extract text.
        
        Args:
            image_id (int): Image ID from CSV
            image_name (str): Image filename
        
        Returns:
            bool: True if successful, False otherwise
        """
        image_path = self.images_folder / image_name
        
        # Check if image exists
        if not image_path.exists():
            print(f"❌ Image not found: {image_name}")
            return False
        
        try:
            # Extract text using both methods
            pytesseract_text = self.extract_text_pytesseract(image_path)
            easyocr_text = self.extract_text_easyocr(image_path)
            
            # Save results to file
            success = self.save_extracted_text(image_id, image_name, pytesseract_text, easyocr_text)
            
            if success:
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error processing {image_name}: {str(e)}")
            return False
    
    def load_dataset(self):
        """
        Load the image dataset from CSV.
        
        Returns:
            pd.DataFrame: Dataset or None if failed
        """
        csv_path = self.data_folder / "imagedataset.csv"
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ Successfully loaded {len(df)} records from CSV")
            return df
        
        except FileNotFoundError:
            print(f"❌ CSV file not found at: {csv_path}")
            print("Please ensure the imagedataset.csv file exists in the data folder.")
            return None
        
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            return None
    
    def run_batch_processing(self):
        """
        Run batch processing on all images in the dataset.
        """
        print("🚀 Starting OCR Batch Processing")
        print("=" * 50)
        
        # Record start time
        self.stats['start_time'] = time.time()
        
        # Configure Tesseract
        if not self.configure_tesseract():
            print("❌ Cannot proceed without Tesseract. Exiting.")
            return
        
        # Initialize EasyOCR
        if not self.initialize_easyocr():
            print("⚠️  Proceeding without EasyOCR (only Pytesseract will be used)")
        
        # Load dataset
        df = self.load_dataset()
        if df is None:
            print("❌ Cannot proceed without dataset. Exiting.")
            return
        
        self.stats['total_images'] = len(df)
        
        # Print processing info
        print(f"\\n📊 Processing Information:")
        print(f"Images to process: {len(df)}")
        print(f"Images folder: {self.images_folder}")
        print(f"Results folder: {self.results_folder}")
        print(f"\\nStarting processing...\\n")
        
        # Process each image with progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            image_id = row['id']
            image_name = row['imagename']
            
            success = self.process_single_image(image_id, image_name)
            
            if success:
                self.stats['processed_successfully'] += 1
                tqdm.write(f"✅ Processed: {image_name}")
            else:
                self.stats['failed_processing'] += 1
                tqdm.write(f"❌ Failed: {image_name}")
        
        # Record end time
        self.stats['end_time'] = time.time()
        
        # Print final statistics
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print final processing statistics."""
        processing_time = self.stats['end_time'] - self.stats['start_time']
        
        print("\\n" + "=" * 50)
        print("📊 PROCESSING COMPLETE")
        print("=" * 50)
        
        print(f"Total images: {self.stats['total_images']}")
        print(f"Successfully processed: {self.stats['processed_successfully']}")
        print(f"Failed to process: {self.stats['failed_processing']}")
        print(f"Success rate: {(self.stats['processed_successfully'] / self.stats['total_images'] * 100):.1f}%")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        if self.stats['processed_successfully'] > 0:
            avg_time = processing_time / self.stats['processed_successfully']
            print(f"Average time per image: {avg_time:.2f} seconds")
        
        print(f"\\n📁 Results saved to: {self.results_folder}")
        
        # List generated files
        result_files = list(self.results_folder.glob("*.txt"))
        print(f"\\n📄 Generated {len(result_files)} text files:")
        for file in result_files[:5]:  # Show first 5 files
            print(f"  • {file.name}")
        if len(result_files) > 5:
            print(f"  ... and {len(result_files) - 5} more files")


def main():
    """Main function to run the batch processing."""
    print("🎯 OCR Batch Processing Script")
    print("===============================")
    print("This script will process all images listed in imagedataset.csv")
    print("and extract text using both Pytesseract and EasyOCR.\\n")
    
    # Ask user for confirmation
    response = input("Do you want to start batch processing? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes']:
        print("❌ Processing cancelled by user.")
        return
    
    # Initialize and run processor
    processor = OCRProcessor()
    processor.run_batch_processing()
    
    print("\\n🎉 Batch processing complete!")
    print("📁 Check the results/extracted_texts/ folder for output files.")


if __name__ == "__main__":
    main()
