import cv2
import numpy as np

img = cv2.imread('Baboon256.bmp')
h, w, c = img.shape 

img1 = np.zeros((h // 2, w // 2, c), dtype = np.uint8)
for i in range(h // 2):
    for j in range(w // 2):
        img1[i, j] = img[i * 2, j * 2]

img2 = img
for i in range(h // 2):
    for j in range(w // 2):
        px = img1[i, j]
        img2[i * 2, j * 2] = px
        img2[i * 2 + 1, j * 2] = px
        img2[i * 2, j * 2 + 1] = px
        img2[i * 2 + 1, j * 2 + 1] = px

cv2.imshow('1. Original', img)
cv2.imshow('2. Shrunk', img1)
cv2.imshow('3. Enlarged', img2)

cv2.waitKey(0) 
cv2.destroyAllWindows()