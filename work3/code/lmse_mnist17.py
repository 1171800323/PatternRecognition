import csv

import numpy as np

from lmse import lmse


def get_csv_data(filename):
    with open(filename) as f:
        return np.array(list(csv.reader(f)), dtype=np.float64)


def cal_label(data, weight):
    data = np.insert(data, 0, [1], axis=1)
    label = data.dot(weight)
    return label


if __name__ == '__main__':
    train_samples = get_csv_data('MNIST-17/MNIST-Train-Samples-17.csv')
    train_labels = get_csv_data('MNIST-17/MNIST-Train-Labels-17.csv')

    test_samples = get_csv_data('MNIST-17/MNIST-Test-Samples-17.csv')
    test_labels = get_csv_data('MNIST-17/MNIST-Test-Labels-17.csv')

    label_num = 10
    train_num, _ = train_labels.shape
    test_num, _ = test_labels.shape

    test_result = []

    for i in range(label_num):
        idx = (train_labels[:, 0] != i)
        labels = np.ones(train_num)
        labels[idx] = -1
        result = lmse(train_samples, labels)

        test_result.append(cal_label(test_samples, result))

    test_result = np.array(test_result, dtype=np.float64)
    test_result = np.argmax(test_result, axis=0)

    print(test_result[:20])
    print(np.int0(test_labels[:20, 0]))
    print(np.sum(test_result[:20] == test_labels[:20, 0]))

    test_right_num = np.sum(test_result[:] == test_labels[:, 0])
    print('right number: ', test_right_num)
    print('accuracy: ', test_right_num / test_num)
