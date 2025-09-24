"""Drawing Shapes & Contours
Includes:
- Line, Rectangle, Circle, Ellipse, Text
- Finding and drawing contours

Run: python 05_drawing_contours.py
"""
import cv2
import numpy as np

# Drawing
canvas = np.zeros((400,400,3), dtype='uint8')
cv2.line(canvas, (50,50), (350,50), (255,0,0), 2)
cv2.rectangle(canvas, (50,100), (200,200), (0,255,0), 2)
cv2.circle(canvas, (300,300), 50, (0,0,255), -1)
cv2.ellipse(canvas, (200,300), (80,40), 0, 0, 360, (255,255,0), 2)
cv2.putText(canvas, 'OpenCV', (100,380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Contours
IMG_PATH = 'images/image2.png'
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)

cv2.imshow('Shapes & Text', canvas)
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
