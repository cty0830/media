import cv2
import numpy as np

img = cv2.imread("antarctic.jpg")
cv2.putText(img, 'Antarctic', (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 12, cv2.LINE_8,False)

cv2.imshow("antarctic", img) 
cv2.waitKey(0)
cv2.destroyAllWindows()