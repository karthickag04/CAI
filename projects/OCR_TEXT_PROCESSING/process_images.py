# Import necessary libraries
import pandas as pd
import pytesseract
from PIL import Image
import os

# Set up Tesseract-OCR path (modify if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Load the CSV file containing image names
csv_path = "data/imagedataset.csv"
df = pd.read_csv(csv_path)

# Ensure the results folder exists
os.makedirs("result", exist_ok=True)

# Iterate through the CSV file and process each image
for _, row in df.iterrows():
    image_name = row['imagename']  # Get the image filename
    image_path = os.path.join("images", image_name)

    if os.path.exists(image_path):  # Check if the image exists
        image = Image.open(image_path)  # Open image
        text = pytesseract.image_to_string(image)  # Extract text

        # Save the extracted text to a file
        output_file = os.path.join("result", f"{image_name}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Extracted text saved for {image_name}")
    else:
        print(f"Warning: Image {image_name} not found!")
