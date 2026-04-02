import cv2
import numpy as np

src = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = src.shape

mapx = np.ones(src.shape, np.float32)
mapy = np.ones(src.shape, np.float32)

for r in range(rows):
    for c in range(cols):
       mapx[r, c] = c
       mapy[r, c] = r

dst = cv2.remap(src, mapx, mapy, cv2.INTER_NEAREST)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()  