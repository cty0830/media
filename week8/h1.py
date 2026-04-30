import cv2
import numpy as np

pictPath = r'C:/Users/User/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier (pictPath)
img = cv2.imread("Solvay1927.jpg")
faces = face_cascade.detectMultiScale(img, scaleFactor=1.089, minNeighbors = 5, minSize=(20,20))

cv2.rectangle(img, (img.shape[1]-140, img.shape[0]-20), (img.shape[1],img.shape[0]), (0,255,255), -1)
cv2.putText(img, "Finding " + str(len(faces)) + " face", (img.shape[1]-135, img.shape[0]-5),
cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    
cv2.imshow("Face", img)

cv2.waitKey(0)
cv2.destroyAllWindows()