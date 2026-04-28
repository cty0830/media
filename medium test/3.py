import cv2
import numpy as np
import random

WINDOW_NAME = "Puzzle_D1348859_ChenTingYao"
IMG_SIZE = 399  
GRID_SIZE = 3
BLOCK_W = IMG_SIZE // GRID_SIZE
INFO_W = 250   
CANVAS_W = IMG_SIZE + INFO_W
CANVAS_H = 450  

img_blocks = []      
current_order = []   
selected_idx = -1    
move_count = 0
is_success = False

def shuffle_puzzle():
    global current_order, move_count, is_success, selected_idx
    order = list(range(GRID_SIZE * GRID_SIZE))
    while True:
        random.shuffle(order)
        if order != list(range(GRID_SIZE * GRID_SIZE)):
            break
    current_order = order
    move_count = 0
    is_success = False
    selected_idx = -1

def check_success():
    return current_order == list(range(GRID_SIZE * GRID_SIZE))

def on_mouse(event, x, y, flags, param):
    global selected_idx, move_count, is_success

    if is_success or event != cv2.EVENT_LBUTTONDOWN:
        return
    
    if x < IMG_SIZE and y < IMG_SIZE:
        col = x // BLOCK_W
        row = y // BLOCK_W
        clicked_idx = row * GRID_SIZE + col

        if selected_idx == -1:
            selected_idx = clicked_idx
        else:
            if selected_idx != clicked_idx:
                current_order[selected_idx], current_order[clicked_idx] = current_order[clicked_idx], current_order[selected_idx]
                move_count += 1
                if check_success():
                    is_success = True
            selected_idx = -1

original_img = cv2.imread('lena.jpg')
img_resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
ref_img = cv2.resize(img_resized, (150, 150)) 

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        block = img_resized[i*BLOCK_W : (i+1)*BLOCK_W, j*BLOCK_W : (j+1)*BLOCK_W]
        img_blocks.append(block)

shuffle_puzzle()

canvas = np.ones((CANVAS_H, CANVAS_W, 3), np.uint8) * 240
cv2.imshow(WINDOW_NAME, canvas)

cv2.setMouseCallback(WINDOW_NAME, on_mouse)

while True:
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), np.uint8) * 240

    for i in range(GRID_SIZE * GRID_SIZE):
        row = i // GRID_SIZE
        col = i % GRID_SIZE
        block_to_draw = img_blocks[current_order[i]].copy()
        
        if i == selected_idx:
            cv2.rectangle(block_to_draw, (0, 0), (BLOCK_W-1, BLOCK_W-1), (0, 255, 0), 5)
        else:
            cv2.rectangle(block_to_draw, (0, 0), (BLOCK_W-1, BLOCK_W-1), (0, 0, 0), 1)

        canvas[row*BLOCK_W : (row+1)*BLOCK_W, col*BLOCK_W : (col+1)*BLOCK_W] = block_to_draw

    start_x = IMG_SIZE + 50
    canvas[20:170, start_x : start_x+150] = ref_img

    cv2.putText(canvas, "Reference", (start_x+30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, f"Moves: {move_count}", (start_x, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(canvas, "R: Restart", (start_x, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    cv2.putText(canvas, "Q: Quit", (start_x, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    if is_success:
        cv2.putText(canvas, "SUCCESS!", (110, 215), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow(WINDOW_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('r') or key == ord('R'):
        shuffle_puzzle()

cv2.destroyAllWindows()