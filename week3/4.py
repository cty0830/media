import cv2

image = cv2.imread('lena.jpg')
img = cv2.imread('dollar.jpg')
cv2.imshow('dollar', img)

face = image[150:240,140:210]
face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
face_gray_3ch = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
h, w = face_gray_3ch.shape[:2]

y = 110
x = 300

img[y : y + h, x : x + w] = face_gray_3ch

cv2.imshow('Modified Dollar', img)

cv2.waitKey(0)
cv2.destroyAllWindows()