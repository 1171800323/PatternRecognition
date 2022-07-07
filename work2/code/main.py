from gmm import gmm
import numpy as np
import csv
from scipy.stats import multivariate_normal

# X = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
#               [2, 1], [1, 2], [2, 2], [3, 2],
#               [6, 6], [7, 6], [8, 6], [7, 7],
#               [8, 7], [9, 7], [7, 8], [8, 8],
#               [9, 8], [8, 9], [9, 9]], dtype=np.float32)

# result = gmm(X, 2)
# print(result)


def get_csv_data(filename):
    with open(filename) as f:
        return np.array(list(csv.reader(f)), dtype=np.float64)


emu_train_samples = get_csv_data('Emu/Emu-Train-Samples.csv')
emu_train_labels = get_csv_data('Emu/Emu-Train-Labels.csv')

samples_label0_idx = (emu_train_labels[:, 0] == 0)
samples_label1_idx = (emu_train_labels[:, 0] == 1)

samples_label0 = emu_train_samples[samples_label0_idx]
samples_label1 = emu_train_samples[samples_label1_idx]

gmm0 = gmm(samples_label0, 2)
print('alpha: ', gmm0['alpha'])
print('mu: ', gmm0['mu'])
print('sigma: ', gmm0['sigma'])

gmm1 = gmm(samples_label1, 2)
print('alpha: ', gmm1['alpha'])
print('mu: ', gmm1['mu'])
print('sigma: ', gmm1['sigma'])


def cal_probability(X, k, alpha, mu, sigma):
    data_num, _ = X.shape
    px = np.zeros((data_num, k))
    for i in range(k):
        px[:, i] = \
            alpha[i][0] * multivariate_normal.pdf(X, mu[i], sigma[i])
    return np.sum(px, axis=1).reshape(-1, 1)


emu_test_samples = get_csv_data('Emu/Emu-Test-Samples.csv')
emu_test_labels = get_csv_data('Emu/Emu-Test-Labels.csv')
prob0 = cal_probability(emu_test_samples, 2,
                        gmm0['alpha'], gmm0['mu'], gmm0['sigma'])
prob1 = cal_probability(emu_test_samples, 2,
                        gmm1['alpha'], gmm1['mu'], gmm1['sigma'])


test_num, _ = emu_test_samples.shape
all_result = np.zeros((test_num, 2))
all_result[:, 0] = prob0[:, 0]
all_result[:, 1] = prob1[:, 0]
print(all_result)
predict_test_labels = np.argmax(all_result, axis=1).reshape(-1, 1)
print(np.sum(predict_test_labels == emu_test_labels))

from matplotlib import pyplot as plt
plt.scatter(emu_test_samples[:,0], emu_test_samples[:,1])
plt.show()