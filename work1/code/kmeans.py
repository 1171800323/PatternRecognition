import numpy as np


def kmeans(X, k):
    point_num, feature_num = X.shape

    # 初始换k个均值向量
    point_mean_idx = np.random.choice(point_num, k, replace=False)
    point_mean = np.array([X[i, :] for i in point_mean_idx], dtype=np.float32)
    pre_point_mean = np.zeros((k, feature_num))

    point_label = np.zeros((point_num, 1))
    iter = 0
    while True:
        iter += 1
        # 对每一个样本找离它最近的中心点，E步
        distance = np.zeros((k, 1))
        for i in range(point_num):
            for j in range(k):
                distance[j][0] = np.linalg.norm(X[i, :] - point_mean[j, :]) # 计算矩阵的2-范数
            
            # 对样本i划分聚类簇
            point_label[i][0] = np.argmin(distance[:, 0])
        
        # 更新k个均值向量，M步
        for i in range(k):
            idx = (point_label[:, 0] == i)
            point_mean[i] = np.sum(X[idx, :], axis=0) / np.sum(idx)

        # 如果均值向量没有更新，跳出循环
        if np.sum(np.abs(pre_point_mean - point_mean)) == 0:
            return point_mean, point_label

        pre_point_mean = np.copy(point_mean)
