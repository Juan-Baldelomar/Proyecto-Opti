import numpy as np


class TestFunction(object):
    def __init__(self, n: int, p: int, dataset: np.ndarray):
        self.n = n
        self.p = p
        self.r = len(dataset)
        self.f_k = []
        self.norm_g_k = []
        self.dataset = dataset
        self.R_index = np.zeros((self.r, 2))       # stores Ri  (1/2 * (model - yi)^2)  and index of point in the dataset

    # returns vector of model function applied to each point in dataset
    def function(self, x: np.ndarray):
        raise NotImplementedError

    # returns vector of abs difference between model function applied to each point and the real value y_i (last column in dataset)
    def abs_diff(self, x: np.ndarray):
        raise NotImplementedError

    def gradient(self, x: np.ndarray):
        raise NotImplementedError

    def Jacobian(self, x: np.ndarray):
        raise NotImplementedError

    # returns sum of first p values of R_Index
    def S(self, x: np.ndarray, keep_record=True):
        raise NotImplementedError

    # calculates each Ri and sorts it with its index
    def getR(self, x: np.ndarray):
        raise NotImplementedError


class Linear_Model(TestFunction):
    def __init__(self, n, p, dataset: np.ndarray):
        super().__init__(n, p, dataset)

    # model function to adjust
    def function(self, x: np.ndarray, keep_record=True):
        return np.matmul(self.dataset[:, :-1], x[:-1]) + x[-1]

    # gradient of objective function
    def gradient(self, x: np.ndarray, keep_record=True):
        grad = np.zeros(self.n)

        # get indexes of first-p Ri (lower values)
        dataset_index = self.R_index[:self.p, 1].astype(int)
        y_model_diff = self.dataset[:, -1] - self.function(x)

        for i in range(self.n - 1):
            grad[i] = np.sum(np.multiply(-self.dataset[dataset_index, i], y_model_diff[dataset_index]))

        grad[-1] = -np.sum(y_model_diff[dataset_index])

        # record grad norm
        if keep_record:
            norm_gk = np.linalg.norm(grad)
            self.norm_g_k.append(norm_gk)

        return grad

    def Jacobian(self, x: np.ndarray):
        J = np.zeros((self.p, self.n))

        # get indexes of first-p Ri (lower values)
        dataset_index = self.R_index[:self.p, 1].astype(int)

        # Set each Jacobian Entry
        J[:, :-1] = -self.dataset[dataset_index, :-1]
        J[:, -1] = -1

        return J

    def abs_diff(self, x: np.ndarray):
        return np.abs(self.function(x) - self.dataset[:, -1])

    # basically this is the objective function
    def S(self, x: np.ndarray, keep_record=True):

        # sort Ri's
        self.getR(x)

        sum_ = 0
        for i in range(self.p):
            sum_ += self.R_index[i][0]

        if keep_record:
            self.f_k.append(sum_)
        return sum_

    def getR(self, x: np.ndarray):
        R = 1/2 * (self.function(x) - self.dataset[:, -1])**2

        # build vector of R value and its index
        for i in range(len(R)):
            self.R_index[i] = (R[i], i)

        # sort R_index by first attribute, ie, R value
        self.R_index = self.R_index[np.argsort(self.R_index[:, 0])]


