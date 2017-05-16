# -*- coding:utf-8 -*-

from sklearn import datasets
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class KMeans:
    """
     KMeans算法流程: 
     1.随机分配k个重心
     
     2.迭代至重心不变或者到了最大迭代次数
       2.1 基于定义的距离函数，将所有点分配至最近的重心，形成K个簇
       2.2 对每个簇重新计算重心 
     
     3.每个簇就是一个类别，输出
        
     KMeans 简单高效，蕴含了EM的思想   
    """

    def __init__(self, k, max_iter=20):
        self.k = k
        self.max_iter = max_iter

    def _random_assign_k_centroids(self, X):
        """ 随机分配k个重心 """
        n_samples, _ = np.shape(X)
        return X[np.random.randint(n_samples, size=self.k)]

    def _nearest_centroid_id(self, centroids, sample):
        """ 返回距离最近的重心的id """
        return np.argmin([distance.euclidean(sample, centroid) for centroid in centroids])

    def _recompute_centroids(self, clusters, X):
        """ 重新计算重心 """
        _, n_features = np.shape(X)

        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(X[clusters[i]], axis=0)
        return centroids

    def _assign_cluster(self, X, centroids):
        """ 分配至最近的重心 """

        clusters = [[] for _ in range(self.k)]

        for sample_id, sample in enumerate(X):
            nearest_centroid_id = self._nearest_centroid_id(centroids, sample)
            clusters[nearest_centroid_id].append(sample_id)

        return clusters

    def _assign_label(self, clusters, X):
        """ 分配标签 """
        y_pred = np.zeros(np.shape(X)[0])

        for cluster_i, cluster in enumerate(clusters):
            y_pred[cluster] = cluster_i
        return y_pred

    def predict(self, X):
        #step 1
        centroids = self._random_assign_k_centroids(X)

        for _ in range(self.max_iter):
            #step 2
            clusters = self._assign_cluster(X, centroids)
            old_centroids = centroids
            #step 3
            centroids = self._recompute_centroids(clusters, X)

            if not (centroids - old_centroids).any():
                break

        return self._assign_label(clusters, X)

    @staticmethod
    def plot(X, y, title):
        """
          简单绘图
        """
        kind = np.unique(y)
        cmap = plt.get_cmap('viridis')

        colors = [cmap(i) for i in np.linspace(0, 1, len(kind))]

        X1 = X[:, 0]
        X2 = X[:, 1]

        for i, l in enumerate(kind):
            _X1 = X1[y == l]
            _X2 = X2[y == l]
            plt.scatter(_X1, _X2, color=colors[i])

        plt.title(title)
        plt.show()


if __name__ == '__main__':
    X, y = datasets.make_blobs()

    clf = KMeans(k=3)
    y_pred = clf.predict(X)

    KMeans.plot(X, y_pred, "KMeans Cluster")
    KMeans.plot(X, y, "Real Cluster")


