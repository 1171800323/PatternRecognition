import numpy as np
from matplotlib import pyplot as plt


def perceptron(data, labels, learning_rate=0.1):
    sample_num, feature_num = data.shape
    weight = np.random.normal(0, 1, feature_num+1)

    while True:
        break_flag = True
        for i in range(sample_num):
            if labels[i]*(weight[0]+np.sum(data[i] * weight[1:])) < 0:
                weight[0] += labels[i] * learning_rate
                weight[1:] += labels[i] * data[i] * learning_rate
                break_flag = False

        if break_flag:
            break
    return weight


if __name__ == "__main__":
    X = np.array([[1, 1], [2, 2], [2, 0], [0, 0], [
                 1, 0], [0, 1]], dtype=np.float32)
    Y = np.array([1, 1, 1, -1, -1, -1], dtype=np.float32)
    result = perceptron(X, Y, learning_rate=0.1)
    print(result)

    x1 = [0, -result[0]/result[2]]
    x2 = [-result[0]/result[1], 0]
    plt.plot([x1[0], x2[0]], [x1[1], x2[1]])

    plt.scatter(X[0:3, 0], X[0:3, 1])
    plt.scatter(X[3:6, 0], X[3:6, 1])
    plt.show()
