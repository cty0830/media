import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2chinesetext(img, text, left, top, color, font):
    if(isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
     
    draw = ImageDraw.Draw(img)
    fontext = ImageFont.truetype(
        "C:\Windows\mingliu.ttc",
        font,
        encoding = "utf-8"
    )
    
    draw.text((left, top), text, color, font = fontext)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        
img = cv2.imread("antarctic.jpg")
img = cv2chinesetext(img, '逢甲大學', 120, 150, (255, 0, 0), 50)

cv2.imshow("antarctic", img) 
cv2.waitKey(0)
cv2.destroyAllWindows()