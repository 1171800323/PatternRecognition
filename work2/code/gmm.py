import numpy as np
from scipy.stats import multivariate_normal


def gaussian(X, mu_i, sigma_i):
    """
    多维高斯分布
    """
    data_num, _ = X.shape
    result = np.zeros(data_num)
    for i in range(data_num):
        # norm_factor = 1.0 / np.clip(np.linalg.det(sigma_i), a_min=1e-10, a_max=1e10)
        norm_factor = 1.0 / np.linalg.det(sigma_i)
        sgm = norm_factor * \
            np.exp(-0.5 * np.transpose(X[i] -
                                       mu_i).dot(np.linalg.pinv(sigma_i)).dot(X[i] - mu_i))
        result[i] = sgm
    return result


def gmm(X, k):
    data_num, feature_num = X.shape
    # 初始化alpha
    alpha = np.ones((k, 1)) / k

    # 初始化均值mu，从数据中随机挑选
    mu_idx = np.random.choice(data_num, k, replace=False)
    mu = np.array([X[i, :] for i in mu_idx], dtype=np.float64)

    # 初始化协方差矩阵sigma
    sigma = np.zeros((k, feature_num, feature_num))
    for i in range(k):
        var = np.var(X[mu_idx[i], :])
        sigma[i] = var * np.identity(feature_num)
    
    px = np.zeros((data_num, k))

    prev_LLD = 0
    iter = 0
    while True:
        iter += 1
        # E步，计算每个样本属于每个高斯成分的后验概率
        for i in range(k):
            px[:, i] = \
                alpha[i][0] * multivariate_normal.pdf(X, mu[i], sigma[i])

        tmp = np.sum(px, axis=1).reshape(-1, 1)
        tmp[tmp==0] = 1    # 防止除0错误
        gamma = px / tmp

        # 更新标签
        index = np.zeros(data_num)
        for i in range(data_num):
            index[i] = np.argmax(gamma[i, :])

        # M步，更新模型参数，使得LLD最大
        Nk = np.sum(gamma, axis=0)
        temp = np.transpose(gamma).dot(X)
        
        alpha[:, 0] = Nk / data_num
        
        for i in range(k):
            mu[i] = temp[i, :] / Nk[i]
        
        for i in range(k):
            XminusMu = X - mu[i]
            sigma[i] = np.transpose(XminusMu).dot(np.diag(
                gamma[:, i])).dot(XminusMu) / Nk[i]

        # 停止条件，如果LLD增加很少，则停止
        LLD = np.sum(np.log(np.transpose(alpha).dot(np.transpose(px))))

        if np.abs(LLD - prev_LLD) < 1e-20:
            print('total iter:', iter)
            return {'label': index, 'alpha': alpha, 'mu': mu, 'sigma': sigma}
        prev_LLD = LLD
