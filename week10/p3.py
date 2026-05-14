import cv2
import numpy as np

src = cv2.imread('hw15_1.jpg')
cv2.imshow('src', src)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    output = np.zeros_like(src)
    
    cv2.drawContours(output, [cnt], -1, (0, 255, 255), thickness=-1)
    
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        cv2.circle(output, (cX, cY), 5, (255, 0, 0), -1)
    
    cv2.imshow(f'contours{i}', output)

cv2.waitKey(0)
cv2.destroyAllWindows()