import cv2
import numpy as np
 
image = cv2.imread('view.jpg')
cv2.imshow('view', image)

blue, green, red = cv2.split(image)
bgr_image = cv2.merge([blue, green, red])
rgb_image = cv2.merge([red, green, blue])

cv2.imshow('bgr_image', bgr_image)
cv2.imshow('rgb_image', rgb_image)

cv2.waitKey(0)
cv2.destroyAllWindows()