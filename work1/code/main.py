import csv

import numpy as np
from matplotlib import pyplot as plt

from kmeans import kmeans


def show(X, label, mean, k):
    colors = ['hotpink', 'aqua', 'black']
    for i in range(k):
        idx = label[:, 0] == i
        point = np.array(X[idx, :], dtype=np.float32)
        plt.scatter(point[:, 0], point[:, 1], color=colors[i])
        # plt.scatter(mean[i][0], mean[i][1], s=100, color=colors[i])
    plt.show()


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
              [2, 1], [1, 2], [2, 2], [3, 2],
              [6, 6], [7, 6], [8, 6], [7, 7],
              [8, 7], [9, 7], [7, 8], [8, 8],
              [9, 8], [8, 9], [9, 9]], dtype=np.float32)
print(X.shape)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

k = 2
point_mean, point_label = kmeans(X, k)
show(X, point_label, point_mean, k)

with open('Sample.csv') as f:
    csv_data = list(csv.reader(f))
X = np.array(csv_data, dtype=np.float32)

k = 3
point_mean, point_label = kmeans(X, k)


def get_number_img(data, shape=(28, 28)):
    image = np.reshape(data, shape)
    image = np.uint8(image)
    return image


for i in range(k):
    idx = (point_label[:, 0] == i)
    images = X[idx, :]
    for j in range(200):
        img = get_number_img(images[j])
        plt.subplot(20, 10, j+1)
        plt.imshow(img, cmap='gray')
    plt.show()
