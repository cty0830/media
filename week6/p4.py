import cv2
import numpy as np

src = cv2.imread('hung_pepper.jpg')

dst = cv2.medianBlur(src, 3)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()  