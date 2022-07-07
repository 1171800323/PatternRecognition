import csv

import numpy as np
from scipy.stats import multivariate_normal

from gmm import gmm


def get_csv_data(filename):
    with open(filename) as f:
        return np.array(list(csv.reader(f)), dtype=np.float64)


def cal_probability(X, k, alpha, mu, sigma):
    data_num, _ = X.shape
    px = np.zeros((data_num, k))
    for i in range(k):
        px[:, i] = \
            alpha[i][0] * multivariate_normal.pdf(X, mu[i], sigma[i])
    return np.sum(px, axis=1).reshape(-1, 1)

def max_min_normalization(X):
    X_min = np.min(X, axis=1).reshape(-1,1)
    X_max = np.max(X, axis=1).reshape(-1,1)
    X = (X - X_min) / (X_min - X_max)
    return X

train_samples = get_csv_data('MNIST/MNIST-Train-Samples.csv')
train_labels = get_csv_data('MNIST/MNIST-Train-Labels.csv')

test_samples = get_csv_data('MNIST/MNIST-Test-Samples.csv')
test_labels = get_csv_data('MNIST/MNIST-Test-Labels.csv')

# train_samples = max_min_normalization(train_samples)
# test_samples = max_min_normalization(test_samples)

# 类别数目，MNIST为0-9
label_num = 10
# 高斯数，每个类别的数据由k个高斯成分组成
k = 2
test_num, _ = test_samples.shape
all_result = np.zeros((test_num, label_num))

for i in range(label_num):
    print('Train GMM{}---------------------------------------'.format(i))
    
    # 使用第i类的数据进行训练，得到gmm_i
    idx = (train_labels[:, 0] == i)
    train_data = train_samples[idx]
    result = gmm(train_data, k)
    
    # 根据gmm_i来估计每个测试数据产生的概率
    all_result[:, i] = cal_probability(test_samples, k, result['alpha'],
                          result['mu'], result['sigma'])[:, 0]

# 对每个样本划定类别，取每一行概率最大值的索引为类别标签
predict_test_labels = np.argmax(all_result, axis=1).reshape(-1, 1)

# 将预测标签和ground truth对比，计算标签一样的样本数目
print(np.sum(predict_test_labels == test_labels))