import cv2
import numpy as np

pictPath = r"C:\Users\User\Downloads\opencv-master\data\haarcascades\haarcascade_frontalcatface_extended.xml"
face_cascade = cv2.CascadeClassifier (pictPath)

img = cv2.imread("dogcat.jpg")
faces = face_cascade.detectMultiScale(img, scaleFactor=1.009, minNeighbors = 10, minSize=(20,20))

img1 = cv2.imread("fatcat.jpg")
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced_gray = clahe.apply(gray)
blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
faces1 = face_cascade.detectMultiScale(blurred, scaleFactor=1.05, minNeighbors=5, minSize=(20,20))

img2 = cv2.imread("dogcat1.jpg")
faces2 = face_cascade.detectMultiScale(img2, scaleFactor=1.01, minNeighbors = 15, minSize=(20,20))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    
for (x,y,w,h) in faces1:
    cv2.rectangle(img1, (x,y), (x+w,y+h), (255,0,0),2)
    
for (x,y,w,h) in faces2:
    cv2.rectangle(img2, (x,y), (x+w,y+h), (255,0,0),2)
    
cv2.imshow("cat", img)
cv2.imshow("cat1", img1)
cv2.imshow("cat2", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()