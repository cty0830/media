import cv2

pictPath = r"C:\Users\User\Downloads\opencv-master\data\haarcascades\haarcascade_russian_plate_number.xml"
face_cascade = cv2.CascadeClassifier (pictPath)
img = cv2.imread("car1.jpg")
faces = face_cascade.detectMultiScale(img, scaleFactor=1.089, minNeighbors = 5, minSize=(20,20))

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
    
cv2.imshow("car", img)

cv2.waitKey(0)
cv2.destroyAllWindows()