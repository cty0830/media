import cv2
import numpy as np

# 1. 讀取影像
img = cv2.imread('Baboon256.bmp')

# 防呆機制：確保檔案讀取成功
if img is None:
    print("錯誤：找不到 Baboon256.bmp 檔案，請確認圖片與程式碼在同一資料夾！")
else:
    h, w, c = img.shape 
    print(f"成功讀取影像，尺寸為: {w}x{h}，通道數: {c}")

    # --- 步驟二：縮小影像 (256x256 -> 128x128) ---
    img_shrunk = np.zeros((h // 2, w // 2, c), dtype=np.uint8)

    for i in range(h // 2):
        for j in range(w // 2):
            # 取 2x2 區塊的左上角像素值
            img_shrunk[i, j] = img[i * 2, j * 2]

    # --- 步驟三：反向放大影像 (128x128 -> 256x256) ---
    img_enlarged = np.zeros((h, w, c), dtype=np.uint8)

    for i in range(h // 2):
        for j in range(w // 2):
            pixel_val = img_shrunk[i, j]
            # 將該值複製到 2x2 區塊的四個像素
            img_enlarged[i * 2, j * 2]         = pixel_val
            img_enlarged[i * 2 + 1, j * 2]     = pixel_val
            img_enlarged[i * 2, j * 2 + 1]     = pixel_val
            img_enlarged[i * 2 + 1, j * 2 + 1] = pixel_val

    # --- 步驟四：差異比較 ---
    # 計算原圖與還原圖的絕對差值
    diff = cv2.absdiff(img, img_enlarged)

    # --- 顯示結果 ---
    cv2.imshow('1. Original', img)
    cv2.imshow('2. Shrunk', img_shrunk)
    cv2.imshow('3. Enlarged (Reconstructed)', img_enlarged)
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()