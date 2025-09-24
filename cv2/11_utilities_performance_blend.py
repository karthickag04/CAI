"""Utilities: Blending & Timing
- addWeighted blending
- Measuring operation time with tick counts

Run: python 11_utilities_performance_blend.py
"""
import cv2

IMG_PATH = 'images/image2.png'
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')

blend = cv2.addWeighted(img, 0.7, img, 0.3, 0)

start = cv2.getTickCount()
_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
end = cv2.getTickCount()
seconds = (end - start) / cv2.getTickFrequency()
print(f'cvtColor time: {seconds:.6f}s')

cv2.imshow('Blended', blend)
cv2.waitKey(0)
cv2.destroyAllWindows()
