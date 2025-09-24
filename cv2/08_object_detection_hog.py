"""Pedestrian Detection (HOG + SVM)
- Uses default people detector

Run: python 08_object_detection_hog.py
Replace IMG_PATH with an image containing people.
"""
import cv2

IMG_PATH = 'images/image3.PNG'
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(rects, weights) = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
for (x,y,w,h) in rects:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv2.imshow('Pedestrian Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
