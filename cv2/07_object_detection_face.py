"""Face Detection (Haar Cascade)
- Uses built-in frontal face Haar cascade

Run: python 07_object_detection_face.py
Use your own face image by replacing IMG_PATH.
"""
import cv2

IMG_PATH = 'images/image2.png'  # Replace with a real face image for better results
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
