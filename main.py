import matplotlib.pyplot as plt
import numpy as np
import Optimizacion
import Function as TestFunction
import Data
import time


print("Hola Como estas ")


dataset = Data.readData("linear_data.cvs")

# dataset = np.array([[0, 1], [1, 2.5], [2, 2.5], [3, 3.8], [4, 5.2], [5, 7.5], [6, 6.7], [7, 20], [8, 7.6], [9, 1],
#                     [10, 11.2], [12, 20], [13, 13.8]])

plt.scatter(dataset[:,0], dataset[:, 1])
plt.show()

linear_model = TestFunction.Linear_Model(2, 1, dataset)
x0 = np.array([1, 1])
xopt = Optimizacion.RAFF(x0, linear_model, 80, 90)

indexes = linear_model.R_index[:91, 1].astype(int)
dataset = linear_model.dataset[indexes, :]
plt.scatter(dataset[:,0], dataset[:, 1])
plt.show()