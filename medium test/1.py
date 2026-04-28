import cv2
import numpy as np
import pandas as pd

df_codebook = pd.read_excel('codebook1.xlsx', header=None)
codebook = df_codebook.values.astype(np.float32)

img_org = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img_org.shape
idx_h, idx_w = h // 4, w // 4

index_map = np.zeros((idx_h, idx_w), np.int32)
for i in range(idx_h):
    for j in range(idx_w):
        block = img_org[i*4:(i+1)*4, j*4:(j+1)*4].flatten().astype(np.float32)
        dist = np.sum((codebook - block)**2, axis=1)
        index_map[i, j] = np.argmin(dist)

img_dec = np.zeros((h, w), np.uint8)
for i in range(idx_h):
    for j in range(idx_w):
        idx = index_map[i, j]
        img_dec[i*4:(i+1)*4, j*4:(j+1)*4] = codebook[idx].reshape(4, 4).astype(np.uint8)

psnr_val = cv2.PSNR(img_org, img_dec)
mse_val = np.mean((img_org.astype(np.float64) - img_dec.astype(np.float64)) ** 2)

img_org_bgr = cv2.cvtColor(img_org, cv2.COLOR_GRAY2BGR)
img_dec_bgr = cv2.cvtColor(img_dec, cv2.COLOR_GRAY2BGR)
index_bgr = cv2.cvtColor(index_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)
index_bgr = cv2.resize(index_bgr, (w, h), cv2.INTER_NEAREST)

info_panel = np.full((h, w, 3), 255, np.uint8)

cv2.putText(img_org_bgr, "1. Original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(index_bgr, "2. Index Map", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(img_dec_bgr, "3. Decompressed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(info_panel, "4. Quality Info", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(info_panel, f"MSE: {mse_val:.2f}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.putText(info_panel, f"PSNR: {psnr_val:.2f} dB", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.putText(info_panel, "Formula:", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(info_panel, "PSNR = 10 * log10((255^2) / MSE ))", (1, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

top_row = np.hstack((img_org_bgr, index_bgr))
bottom_row = np.hstack((img_dec_bgr, info_panel))
final_canvas = np.vstack((top_row, bottom_row))

window_name = "VQ Compression Result"
cv2.imshow(window_name, final_canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()