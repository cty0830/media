import cv2
import numpy as np
 
image = cv2.imread('colorbar.jpg')
cv2.imshow('bgr', image)
blue, green, red = cv2.split(image)

zeros = np.zeros(image.shape[:2], dtype = "uint8")

blueimg = cv2.merge([blue, zeros, zeros])
greenimg = cv2.merge([zeros, green, zeros])
redimg = cv2.merge([zeros, zeros, red])

cv2.imshow('blue', blueimg)
cv2.imshow('green', greenimg)
cv2.imshow('red', redimg)

cv2.waitKey(0)
cv2.destroyAllWindows()