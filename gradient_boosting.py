# -*- coding:utf-8 -*-

from __future__ import division
import numpy as np

class LeastSquaresLoss(object):

    def __init__(self):
        pass

    #计算ls残差
    def grad(self, y, y_pred):
        return -1 * (y - y_pred)

class GBDT(object):

    """梯度提升决策树
       实现了gbdt回归功能
       
       参数
       ----------
       n_estimators : int(default=100) 
                   boosting stages执行的数目, Gradient boosting对于防止拟合比较健壮，
                   所以推荐选择一个较大的数目，会产生更好的表现
                   
       learning_rate : int(default=0.1) 
                   learning_rate 控制着每颗树的贡献，在n_estimators和learning_rate
                   间需要权衡一下(trade-off)
       
       max_depth : int(default=3) 
                   每颗树的最大深度，它限制了树的结点数 
                        
       待加入                 
       subsample : float(default=1.0)
                   选取部分样本来拟合弱学习器，把subsample设为<1.0会使方差减少，使偏差上升
                     
    """
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.2,
                 max_depth=3,
                 min_samples_split=2):
        self.loss = LeastSquaresLoss()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.y_shape = None

    def one_hot_shot(self, x):
        """
            对类别进行编码    
        """
        cate = np.unique(x)
        index = dict((v, k) for k, v in enumerate(cate))
        dumpy = np.zeros(len(x), len(cate))
        for i in range(x):
            dumpy[i, index[x[i]]] = 1

        return dumpy

    def fit(self, X, y):
        self.y_shape = np.shape(y)
        y_pred = np.zeros(self.y_shape)
        for i in range(self.n_estimators):
            #残差
            residual = self.loss.grad(y, y_pred)
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth)
            #拟合残差
            tree.fit(X, residual)
            y_pred -= self.learning_rate * tree.predict(X)
            #本轮stage完毕
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(self.y_shape)
        for i in range(self.n_estimators):
            y_pred -= self.learning_rate * self.trees[i].predict(X)

        return y_pred


class RegressionTree(object):
    """回归树
    """
    def __init__(self, min_samples_split=2, max_depth=10):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        feature_is = np.array(range(np.shape(X)[1]))
        self.tree = self._build_tree(X, y, feature_is)

    def _build_tree(self, X, y, feature_is, depth=0):
        min_mse = float('inf')
        best_mask = None
        best_feature = None

        if len(feature_is) >= 1 and np.shape(X)[0] >= self.min_samples_split\
                and depth <= self.max_depth:
            for feature_i in feature_is:
                #第i个特征的值集合
                feature_i_vs = np.unique(X[:, feature_i])

                for feature_i_v in feature_i_vs:
                    #基于特征二分(回归树为二叉树)
                    mask = X[:, feature_i] <= feature_i_v
                    y1 = y[mask]
                    y2 = y[~mask]
                    mse = self._mse(y1) + self._mse(y2)
                    if mse < min_mse:
                        min_mse = mse
                        best_mask = mask
                        best_feature = (feature_i, feature_i_v)

            #如果分属于不同的类别
            if not np.all(best_mask):
                feature_is = feature_is[feature_is != best_feature[0]]
                left = self._build_tree(X[best_mask], y[best_mask], feature_is, depth + 1)
                right = self._build_tree(X[~best_mask], y[~best_mask], feature_is, depth + 1)
                return Node(feature_i=best_feature[0], feature_i_v=best_feature[1], left=left, right=right)

        return Node(leaf_value=np.mean(y))

    def _friedman_mse(self, yl, yr):
        #gbdt论文 公式35 (wl*wr / wl + wr) *(yl_mean - yr_mean)^2
        #权重均为1
        wl = len(yl)
        wr = len(yr)
        yl_mean = np.mean(yl)
        yr_mean = np.mean(yr)

        return (wl * wr / (wl + wr)) * ((yl_mean - yr_mean) ** 2)

    def _mse(self, x):
        if len(x) == 0:
            return 0
        mean = np.mean(x)
        return np.mean((x - mean) ** 2)

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict(x, self.tree))
        return np.asarray(y_pred)

    def _predict(self, x, tree):
        if tree.leaf_value is not None:
            return tree.leaf_value

        if x[tree.feature_i] <= tree.feature_i_v:
            return self._predict(x, tree.left)

        return self._predict(x, tree.right)

class Node(object):
    """结点
    
       leaf_value ： 记录叶子结点值
       feature_i ：特征i
       feature_i_v ： 特征i的值
       left ： 左子树
       right ： 右子树
    """

    def __init__(self, leaf_value=None, feature_i=None, feature_i_v=None, left=None, right=None):
        self.leaf_value = leaf_value
        self.feature_i = feature_i
        self.feature_i_v = feature_i_v
        self.left = left
        self.right = right
