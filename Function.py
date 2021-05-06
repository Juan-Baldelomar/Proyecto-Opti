import numpy as np
import random


class TestFunction(object):
    def __init__(self, n: int):
        self.n = n
        self.f_k = []
        self.norm_g_k = []

    def function(self, x: np.ndarray):
        raise NotImplementedError

    def gradient(self, x: np.ndarray):
        raise NotImplementedError

    def Hessian(self, x: np.ndarray):
        raise NotImplementedError




