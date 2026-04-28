import cv2
import numpy as np

drawing = False      
ix, iy = -1, -1      
mode = 6             
ctrl_pressed = False  
canvas_w, canvas_h = 800, 600     

undo_stack = []
redo_stack = []

def nothing(x):
    pass

img = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
cv2.namedWindow('Drawing_Tool')

cv2.createTrackbar('Red', 'Drawing_Tool', 0, 255, nothing)
cv2.createTrackbar('Green', 'Drawing_Tool', 0, 255, nothing)
cv2.createTrackbar('Blue', 'Drawing_Tool', 0, 255, nothing)
cv2.createTrackbar('Radius', 'Drawing_Tool', 4, 60, nothing)  
cv2.setTrackbarMin('Radius', 'Drawing_Tool', 4)   
cv2.createTrackbar('Thickness', 'Drawing_Tool', 1, 20, nothing)  
cv2.setTrackbarMin('Thickness', 'Drawing_Tool', 1)

def save_state():
    global undo_stack, redo_stack
    undo_stack.append(img.copy())
    redo_stack.clear() 
    if len(undo_stack) > 20: 
        undo_stack.pop(0)

def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, img, mode, ctrl_pressed

    r = cv2.getTrackbarPos('Red', 'Drawing_Tool')
    g = cv2.getTrackbarPos('Green', 'Drawing_Tool')
    b = cv2.getTrackbarPos('Blue', 'Drawing_Tool')
    rad = cv2.getTrackbarPos('Radius', 'Drawing_Tool')
    thick = cv2.getTrackbarPos('Thickness', 'Drawing_Tool')
    color = (b, g, r)

    is_ctrl = flags & cv2.EVENT_FLAG_CTRLKEY
    ctrl_pressed = bool(is_ctrl) 

    if event == cv2.EVENT_LBUTTONDOWN:
        save_state()
        drawing = True
        ix, iy = x, y
        
        if is_ctrl: 
            cv2.rectangle(img, (x-rad, y-rad), (x+rad, y+rad), color, thick)
        elif mode == 1: 
            cv2.circle(img, (x, y), rad, color, thick)
        elif mode == 2: 
            cv2.circle(img, (x, y), rad, color, -1)
        elif mode == 3: 
            cv2.rectangle(img, (x-rad, y-rad), (x+rad, y+rad), color, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and not is_ctrl:
            if mode == 4: 
                cv2.circle(img, (x, y), rad, (255, 255, 255), -1)
            elif mode == 6: 
                cv2.line(img, (ix, iy), (x, y), color, thick)
                ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            if mode == 5 and not is_ctrl: 
                cv2.line(img, (ix, iy), (x, y), color, thick)
            drawing = False

cv2.setMouseCallback('Drawing_Tool', draw_shape)

while True:
    info_panel = np.full((canvas_h, 250, 3), 240, np.uint8)
    
    r = cv2.getTrackbarPos('Red', 'Drawing_Tool')
    g = cv2.getTrackbarPos('Green', 'Drawing_Tool')
    b = cv2.getTrackbarPos('Blue', 'Drawing_Tool')
    rad = cv2.getTrackbarPos('Radius', 'Drawing_Tool')
    thick = cv2.getTrackbarPos('Thickness', 'Drawing_Tool')

    mode_names = {1:"Hollow Circle", 2:"Solid Circle", 3:"Solid Square", 4:"Eraser", 5:"Line", 6:"Free Curve"}
    display_mode = "Quick Square (Ctrl)" if ctrl_pressed else mode_names[mode]
    
    cv2.putText(info_panel, "INFO PANEL", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)
    cv2.putText(info_panel, f"Mode: {display_mode}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.rectangle(info_panel, (20, 110), (200, 140), (b, g, r), -1)
    cv2.putText(info_panel, f"R:{r} G:{g} B:{b}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(info_panel, f"Radius: {rad}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    cv2.putText(info_panel, f"Thickness: {thick}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    shortcuts = [("1", "Hollow Circle"), ("2", "Solid Circle"), ("3", "Solid Square"), ("4", "Eraser"), ("5", "Line"), ("6", "Free Curve"), ("U", "Undo"), ("R", "Redo"), ("Ctrl+L", "Quick Square")]
    for i, (k, v) in enumerate(shortcuts):
        cv2.putText(info_panel, f"[{k}] {v}", (20, 300 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

    final_frame = np.hstack((img, info_panel))
    cv2.imshow('Drawing_Tool', final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif ord('1') <= key <= ord('6'):
        mode = int(chr(key))
    elif key == ord('u') or key == ord('U'):
        if undo_stack:
            redo_stack.append(img.copy())
            img = undo_stack.pop()
    elif key == ord('r') or key == ord('R'): 
        if redo_stack:
            undo_stack.append(img.copy())
            img = redo_stack.pop()

cv2.destroyAllWindows()