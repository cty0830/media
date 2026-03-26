import cv2
import numpy as np
import random

width, height = 640, 480
paddle_w, paddle_h = 100, 20
r = 15

def reset():
    global ball_x, ball_y, ball_dx, ball_dy, paddle_x, score, game_over, paused, r
    ball_x = int(width / 2)
    ball_y = int(height / 2)
    ball_dx = random.choice([-5, 5])
    ball_dy = -5
    paddle_x = int((width - paddle_w) / 2)
    score = 0
    game_over = False
    paused = False
    r = 15 
reset()

cv2.namedWindow("Bouncing Ball Game")

while True:
    img = np.ones((height, width, 3), np.uint8) * 255

    key = cv2.waitKey(20)
    if key == 27: 
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('r') or key == ord('R'): 
        reset()
    elif key == ord('a') or key == ord('A'): 
        paddle_x = max(0, paddle_x - 40)
    elif key == ord('d') or key == ord('D'): 
        paddle_x = min(width - paddle_w, paddle_x + 40)

    if not paused and not game_over:
        ball_x += ball_dx
        ball_y += ball_dy

        if ball_x <= r:
            ball_x = r
            ball_dx = abs(ball_dx)
            r = random.randint(10, 30) 
            
        if ball_x >= width - r:
            ball_x = width - r
            ball_dx = -abs(ball_dx)
            r = random.randint(10, 30)
            
        if ball_y <= r + 50: 
            ball_y = r + 50
            ball_dy = abs(ball_dy)
            r = random.randint(10, 30) 

        if (ball_y + r >= height - paddle_h - 10) and (paddle_x <= ball_x <= paddle_x + paddle_w):
            ball_y = height - paddle_h - 10 - r
            ball_dy = -abs(ball_dy)
            score += 1
            r = random.randint(10, 30) 
                
        if ball_y > height:
            game_over = True

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "A:Left  D:Right  SPACE:Pause  R:Restart  ESC:Exit", (10, 30), font, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(img, (0, 50), (width, 50), (200, 200, 200), 1) 
    
    cv2.putText(img, f"Score: {score}", (20, 90), font, 1.2, (0, 0, 0), 2)

    cv2.circle(img, (int(ball_x), int(ball_y)), r, (255, 0, 0), -1)
    cv2.rectangle(img, (paddle_x, height - paddle_h - 10), (paddle_x + paddle_w, height - 10), (0, 200, 0), -1)

    if game_over:
        cv2.putText(img, "GAME OVER!", (200, int(height / 2)), font, 1.5, (0, 0, 255), 3)
        cv2.putText(img, "Press R to Restart", (210, int(height / 2) + 50), font, 0.8, (50, 50, 50), 2)
    elif paused:
        cv2.putText(img, "PAUSED", (240, int(height / 2)), font, 1.5, (255, 0, 0), 3)

    cv2.imshow("Bouncing Ball Game", img)
    
cv2.destroyAllWindows()