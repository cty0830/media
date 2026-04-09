import cv2
import numpy as np

src = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = src.shape
padded = cv2.copyMakeBorder(src, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)

Gy = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32)

dst = np.zeros((rows, cols), dtype=np.uint8)

for r in range(rows):
    for c in range(cols):
        patch = padded[r:r+3, c:c+3].astype(np.float32)
        
        gx = np.sum(patch * Gx)
        gy = np.sum(patch * Gy)
        
        g = np.sqrt(gx**2 + gy**2)
        
        if g > 255:
            g = 255
            
        dst[r, c] = np.uint8(g)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()  