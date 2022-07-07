from matplotlib import pyplot as plt
import numpy as np


def lmse(data, labels, epsilon=0.001):
    sample_num, feature_num = data.shape

    b = np.ones(sample_num)

    data = np.insert(data, 0, [1], axis=1)
    data = data * np.expand_dims(labels, axis=1)
    data_trans = np.transpose(data)
    data_trans_data = data_trans.dot(data)

    if np.linalg.det(data_trans_data) == 0:
        data_trans_data += np.identity(feature_num+1) * epsilon

    data_inv = np.linalg.inv(data_trans_data).dot(data_trans)
    return data_inv.dot(b)


if __name__ == "__main__":
    X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [
                 1, 0], [0, 1]], dtype=np.float32)
    Y = np.array([1, 1, 1, -1, -1, -1], dtype=np.float32)

    result = lmse(X, Y)
    print(result)

    x1 = [0, -result[0]/result[2]]
    x2 = [-result[0]/result[1], 0]
    plt.plot([x1[0], x2[0]], [x1[1], x2[1]])

    plt.scatter(X[0:3, 0], X[0:3, 1])
    plt.scatter(X[3:6, 0], X[3:6, 1])
    plt.show()
