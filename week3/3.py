import cv2
import numpy as np
 
image = cv2.imread('street.png', cv2.IMREAD_UNCHANGED)
cv2.imshow('before modify', image)

print(f"顯示資訊")
print(f"修改前img[10,50] = {image[10, 50]}")
print(f"修改前img[50,99] = {image[50, 99]}")
print("-" * 40)

image[0:200,0:200,3] = 128

print(f"修改後img[10,50] = {image[10, 50]}")
print(f"修改後img[50,99] = {image[50, 99]}")
    
cv2.imwrite("street128.png", image)


cv2.waitKey(0)
cv2.destroyAllWindows()