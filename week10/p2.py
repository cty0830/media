import cv2

src = cv2.imread("lake.jpg")
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(src_gray,150,255,cv2.THRESH_BINARY)
dst_result = cv2.bitwise_and(src, src, mask=binary)

result = src.copy()
result[binary==255]=(255,255,255)

cv2.imshow("src", src)           
cv2.imshow("binary", binary)      
cv2.imshow("dst result", dst_result) 
cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()