import matplotlib.pyplot as plt
import numpy as np
import Optimizacion
import Function as TestFunction
import Data
import time


print("Hola Como estas ")


dataset = Data.readData("linear_data.cvs")
plt.scatter(dataset[:,0], dataset[:, 1])
plt.show()

linear_model = TestFunction.Linear_Model(2, 5, dataset)
x0 = np.array([1, 2])
xopt = Optimizacion.RAFF(x0, linear_model, 5, 10)
