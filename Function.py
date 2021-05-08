import numpy as np
import random


class TestFunction(object):
    def __init__(self, n: int, p:int, dataset:np.ndarray):
        self.n = n
        self.p = p
        self.f_k = []
        self.norm_g_k = []
        self.dataset = dataset
        self.R_index = []

    def function(self, x: np.ndarray):
        raise NotImplementedError

    def abs_diff(self, x:np.ndarray):
        raise NotImplementedError

    def gradient(self, x: np.ndarray):
        raise NotImplementedError

    def Jacobian(self, x: np.ndarray):
        raise NotImplementedError

    def S(self):
        raise NotImplementedError

    def getR(self):
        raise NotImplementedError


class Linear_Model(TestFunction):
    def __init__(self, n, p, dataset:np.ndarray):
        super().__init__(n, p, dataset)

    def function(self, x: np.ndarray, keep_record=True):
        return np.matmul(self.dataset[:, :-1], x[:-1]) + x[-1]

    def gradient(self, x: np.ndarray, keep_record=True):
        grad = np.zeros(self.n)

        return grad

    def abs_diff(self, x: np.ndarray):
        return np.abs(self.function(x) - self.dataset[:-1])

    def S(self):
        sum_ = 0
        for i in range(self.p):
            sum_ += self.R_index[i][0]

        return sum_
