import cv2
import random

radius_max = 30    
direction = 1      
is_running = True

while True:
    if is_running:
        img = cv2.imread("antarctic.jpg")
        
        cy = int(img.shape[0] / 2 ) #高
        cx = int(img.shape[1] / 2 ) #寬        
        cv2.circle(img, (300, 250), 30, (0, 0, 255), -1) #紅色圓
        
        for r in range(45, radius_max, 15): # 動畫圓
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.circle(img, (300, 250), r, color, 2)

        radius_max += 10 * direction #往內或向外
        
        if radius_max > min(cx, cy) or radius_max <= 30: #是否碰到邊界
                direction *= -1
            
    cv2.imshow("My Draw", img)

    key = cv2.waitKey(500) & 0xFF
    if key == ord('q'): break     
    if key == ord(' '): is_running = False 
    if key == ord('r'): is_running = True  
    
    if cv2.getWindowProperty("My Draw", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()