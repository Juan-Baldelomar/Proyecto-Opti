import matplotlib.pyplot as plt
import numpy as np
import Optimizacion
import Function as TestFunction
import Data
import time


def generateDatasets():
    cuadratic_dataset = Data.generateRandomData(Data.cuadratic_model2D, 2, 3.5, 5, x=None, n=500,
                                                min_x=-50, max_x=50, std_noise=20, min_out=-50, max_out=1000)
    plt.scatter(cuadratic_dataset[:, 0], cuadratic_dataset[:, 1])
    plt.show()
    Data.saveData("cuadratic_dataset2.csv", cuadratic_dataset)


print("Init")

# read dataset
dataset = Data.readData("linear_data.cvs")

# plot dataset
plt.scatter(dataset[:,0], dataset[:, -1])
plt.show()

# create model function
linear_model = TestFunction.Linear_Model(2, 1, dataset)

# initial point
x0 = np.array([1, 1])

# find optimum and number of trusted points
xopt, n_trusted_points = Optimizacion.RAFF(x0, linear_model, 80, 100, max_iter=100000)

# read first trusted points indexes
indexes = linear_model.R_index[:n_trusted_points, 1].astype(int)

# new dataset without outliers
dataset = linear_model.dataset[indexes, :]

# plot result
plt.scatter(dataset[:, 0], dataset[:, -1])

y = Data.linear_model(dataset[:, 0].reshape((len(indexes), 1)), xopt)

plt.plot(dataset[:, 0], y, color='red')
plt.show()


# ------------------------------------------------ chew data ------------------------------------------------

# # read dataset
# dataset = Data.readData("data_chew.csv")
#
# dataset = dataset/1e+7
# dataset = dataset[:, :-1]   # dump last column
# dataset[:, (0, -1)] = dataset[:, (-1, 0)]
#
# # plot dataset
# plt.scatter(dataset[:,0], dataset[:, -1])
# plt.show()
#
# # create model function
# linear_model = TestFunction.Linear_Model(8, 30, dataset)
#
# # initial point
# x0 = np.array([ 3.20702399e+01, -2.60840463e-02, -9.79738576e-02,  1.27298930e+00,
#        -2.31645646e+00, -5.67326631e+00, -3.33947408e+01,  7.33625290e-02])
#
# # find optimum and number of trusted points
# xopt, n_trusted_points = Optimizacion.RAFF(x0, linear_model, 35, 39, max_iter=10000)
#
# # read first trusted points indexes
# indexes = linear_model.R_index[:n_trusted_points, 1].astype(int)
#
# # new dataset without outliers
# dataset2 = linear_model.dataset[indexes, :]
#
# # plot result
# plt.scatter(dataset2[:, 0], dataset2[:, -1])
#
# y = Data.linear_model(dataset[:, :-1], xopt)
#
# plt.plot(dataset[:, 0], y, color='red')
# plt.show()
#
# linear_model.S(xopt)
