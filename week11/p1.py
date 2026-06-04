import cv2

src = cv2.imread("hand2.jpg")
cv2.imshow("src", src)

src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
ret, dst_binary = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(dst_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)        
    hull = cv2.convexHull(contour)
    dst = cv2.polylines(src, [hull], True, (0,255,0),2)

cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()