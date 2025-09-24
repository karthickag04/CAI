"""Filtering, Blurring, Edges, Morphology
Includes:
- Average, Gaussian, Median, Bilateral
- Canny edges, Sobel, Laplacian
- Erosion, Dilation

Run: python 04_filtering_edges_morphology.py
"""
import cv2

IMG_PATH = 'images/image2.png'
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')

blur = cv2.blur(img, (5,5))
gaussian = cv2.GaussianBlur(img, (5,5), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

edges = cv2.Canny(img, 100, 200)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
laplacian = cv2.Laplacian(img, cv2.CV_64F)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
eroded = cv2.erode(mask, None, iterations=2)
dilated = cv2.dilate(mask, None, iterations=2)

cv2.imshow('Blur', blur)
cv2.imshow('Gaussian', gaussian)
cv2.imshow('Median', median)
cv2.imshow('Bilateral', bilateral)
cv2.imshow('Canny', edges)
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('threshold', mask)

cv2.imshow('Eroded', eroded)
cv2.imshow('Dilated', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
