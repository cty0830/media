import cv2
import numpy as np

with np.load('knn_digit.npz') as data:
    train = data['train']
    train_labels = data['train_labels']

img = cv2.imread('8.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)
img = cv2.resize(img, (20, 20)).reshape ((1, 400))
data = img.astype (np.float32)

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret, result, neighbours, dist = knn.findNearest(data, k = 5)
print(f"識別的數字是 = {int(result[0,0])}")

img1 = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img1', img1)
img1 = cv2.resize(img1, (20, 20)).reshape ((1, 400))
data = img1.astype (np.float32)

knn1 = cv2.ml.KNearest_create()
knn1.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret1, result1, neighbours1, dist1 = knn1.findNearest(data, k = 5)
print(f"識別的數字是 = {int(result1[0,0])}")
cv2.waitKey(0)
cv2.destroyAllWindows()