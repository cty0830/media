import cv2
import numpy as np

src = cv2.imread('geneva.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = src.shape

sobelx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
dst_opencv = cv2.magnitude(sobelx, sobely)
dst_opencv = np.uint8(np.clip(dst_opencv, 0, 255))

dstx = cv2.Scharr(src, cv2.CV_32F, 1, 0) # 計算 x 軸影像梯度
dsty = cv2.Scharr(src, cv2.CV_32F, 0, 1) # 計算 y 軸影像梯度
dstx = cv2.convertScaleAbs(dstx) # 將負值轉正值
dsty = cv2.convertScaleAbs(dsty) # 將負值轉正值
dst_scharr = cv2.addWeighted(dstx, 0.5,dsty, 0.5, 0) 

dstla = cv2.Laplacian(src, cv2.CV_64F, ksize=3,scale=0.5)
dstla = cv2.convertScaleAbs(dstla)

cv2.imshow("src", src)
cv2.imshow("scharr", dst_scharr)
cv2.imshow("sobel", dst_opencv)
cv2.imshow("la", dstla)
cv2.waitKey(0)
cv2.destroyAllWindows()  