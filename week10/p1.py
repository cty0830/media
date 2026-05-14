import cv2
import numpy as np

src = cv2.imread("easy.jpg")
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

ret, dst_binary = cv2.threshold(src_gray,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(dst_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    single_contour_img = np.zeros_like(src)
    x, y, w, h = cv2.boundingRect(cnt)
    
    cv2.drawContours(single_contour_img, [cnt], -1, (255, 255, 255), 2)
    cv2.rectangle(single_contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(single_contour_img, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(f"contours{i}", single_contour_img)

    cv2.putText(src, str(i), (x+40, y+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imshow("Original", src)

cv2.waitKey(0)
cv2.destroyAllWindows()