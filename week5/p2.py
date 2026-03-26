import cv2
import numpy as np

img = cv2.imread("numbers.jpg")
img1 = cv2.imread("number.jpg")
maxval = 255 

ret, dst1 = cv2.threshold(img, 127, maxval, cv2.THRESH_BINARY)
ret, dst2 = cv2.threshold(img, 10, maxval, cv2.THRESH_BINARY)
ret, dst3 = cv2.threshold(img1, 2, maxval, cv2.THRESH_BINARY_INV)
ret, dst4 = cv2.threshold(img1, 2, maxval, cv2.THRESH_BINARY)

cv2.imshow("src", img)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.imshow("dst3", dst3)
cv2.imshow("dst4", dst4)

cv2.waitKey()
cv2.destroyAllWindows()