import cv2
import numpy as np

def_low = 50
def_high = 150
def_blur = 1 
def_method = 0

def nothing(x):
    pass

WINDOW_NAME = 'edge_tool'
cv2.namedWindow(WINDOW_NAME)

cv2.createTrackbar('Low', WINDOW_NAME, def_low, 255, nothing)
cv2.createTrackbar('High', WINDOW_NAME, def_high, 255, nothing)
cv2.createTrackbar('Blur', WINDOW_NAME, def_blur, 15, nothing)
cv2.createTrackbar('Method', WINDOW_NAME, def_method, 2, nothing)

img_base = cv2.imread('lena.jpg')

while True:
    low = cv2.getTrackbarPos('Low', WINDOW_NAME)
    high = cv2.getTrackbarPos('High', WINDOW_NAME)
    blur = cv2.getTrackbarPos('Blur', WINDOW_NAME)
    method = cv2.getTrackbarPos('Method', WINDOW_NAME)

    gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    
    k_size = blur * 2 + 1
    blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)

    method_name = ""
    if method == 0:
        method_name = "Canny edge"
        edge_result = cv2.Canny(blurred, low, high)
    elif method == 1:
        method_name = "Sobel"
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edge_result = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    else:
        method_name = "Laplacian"
        edge_result = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        edge_result = cv2.convertScaleAbs(edge_result)

    disp_orig = img_base.copy()
    disp_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    disp_blur = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    disp_edge = cv2.cvtColor(edge_result, cv2.COLOR_GRAY2BGR)

    cv2.putText(disp_orig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(disp_gray, "Grayscale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(disp_blur, f"Blurred (K={k_size})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(disp_edge, f"{method_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    top = np.hstack((disp_orig, disp_gray))
    bottom = np.hstack((disp_blur, disp_edge))
    combined = np.vstack((top, bottom))

    info_panel_width = 250
    total_h = combined.shape[0]
    info_panel = np.full((total_h, info_panel_width, 3), 255, np.uint8)
    
    cv2.putText(info_panel, "Status Panel", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(info_panel, f"Method: {method_name}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(info_panel, f"Low: {low}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(info_panel, f"High: {high}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(info_panel, f"Blur (K): {k_size}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(info_panel, "[q] Quit", (20, total_h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(info_panel, "[s] Save", (20, total_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(info_panel, "[r] Reset", (20, total_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    final_view = np.hstack((combined, info_panel))
    cv2.imshow(WINDOW_NAME, final_view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.imwrite('edge_output.jpg', edge_result)
    elif key == ord('r') or key == ord('R'):
        cv2.setTrackbarPos('Low', WINDOW_NAME, def_low)
        cv2.setTrackbarPos('High', WINDOW_NAME, def_high)
        cv2.setTrackbarPos('Blur', WINDOW_NAME, def_blur)
        cv2.setTrackbarPos('Method', WINDOW_NAME, def_method)

cv2.destroyAllWindows()