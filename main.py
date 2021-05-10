import matplotlib.pyplot as plt
import numpy as np
import Optimizacion as op
import Function as TestFunction
import Data as data
import time


print("Hola Como estas ")


dataset = data.read_data("linear_data.csv")
plt.plot(dataset[:,0], dataset[:, 1])
plt.show()

#linear_model = TestFunction()