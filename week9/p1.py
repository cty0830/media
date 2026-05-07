import numpy as np
import matplotlib.pyplot as plt

num = 50
trains = np.random.randint(0, 500, size=(150, 2))

labels = np.array([0]*num + [1]*num + [2]*num)

test_point = np.array([200, 300])

distances = np.sqrt(np.sum((trains - test_point)**2, axis=1))

k = 7
nearest_indices = np.argsort(distances)[:k]
nearest_labels = labels[nearest_indices]

counts = np.bincount(nearest_labels)
prediction = np.argmax(counts)

label_names = ['藍色 X', '黃色 O', '黑色 V']
result_name = label_names[prediction]

plt.figure(figsize=(9, 9))

plt.scatter(trains[0:50, 0], trains[0:50, 1], s=50, c='b', marker='x', label='Blue X')
plt.scatter(trains[50:100, 0], trains[50:100, 1], s=50, c='y', marker='o', label='Yellow O')
plt.scatter(trains[100:150, 0], trains[100:150, 1], s=50, c='k', marker='v', label='Black V')
plt.scatter(test_point[0], test_point[1], s=100, c='r', marker='s', label='Target (200,300)')

plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f"在 K=7 的情況下，座標 (200, 300) 附近的標籤分別為: {nearest_labels}")
print(f"根據投票結果，該點屬於: {result_name}")