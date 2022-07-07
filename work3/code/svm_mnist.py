import csv

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


def get_csv_data(filename):
    with open(filename) as f:
        return np.array(list(csv.reader(f)), dtype=np.float64)


if __name__ == '__main__':
    train_samples = get_csv_data('MNIST/MNIST-Train-Samples.csv')
    train_labels = get_csv_data('MNIST/MNIST-Train-Labels.csv')

    test_samples = get_csv_data('MNIST/MNIST-Test-Samples.csv')
    test_labels = get_csv_data('MNIST/MNIST-Test-Labels.csv')

    train_num, _ = train_labels.shape
    test_num, _ = test_labels.shape

    # clf = svm.SVC(C=1, kernel='linear')
    # clf = svm.SVC(C=1, kernel='poly', gamma='scale')
    # clf = svm.SVC(C=1, kernel='rbf', gamma='auto')
    clf = svm.SVC(C=0.01, kernel='rbf', gamma='scale')

    use_train_num = int(train_num)
    use_test_num = int(test_num)

    clf.fit(train_samples[:use_train_num], train_labels[:use_train_num, 0])
    y_hat = clf.predict(test_samples[:use_test_num])
    acc = accuracy_score(test_labels[:use_test_num, 0], y_hat)
    print(acc)
