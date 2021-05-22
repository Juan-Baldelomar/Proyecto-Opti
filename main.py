import matplotlib.pyplot as plt
import numpy as np

import Function
import Optimizacion
import Data
import time


def generateDatasets(model, coeffs, csv_filename, n_points, x_dim, out_ratio):
    dataset, out_index = Data.generateRandomData(model, coeffs, x=None, n=n_points, m = x_dim, min_x=1,
                                                 max_x=30, out_ratio=out_ratio, std_noise=200, min_out=1, max_out=2)

    np.savetxt(csv_filename, dataset)
    # Data.saveData(csv_filename, dataset)
    # plot dataset
    plt.scatter(dataset[:, 0], dataset[:, -1])
    plt.title("Original data")
    plt.show()

    return dataset, out_index


print("Init")

# Params:

# model
r = 100  # number of observations
m = 1  # dim of observations
model = Function.Linear_Model
out_ratio = .10  # outliers proportion of the observations
max_iter = 100000

if model == Function.Linear_Model:
    coeffs = np.array([-200., -1000])
    model_name = 'linear'
    gen_model = Data.linear_model
elif model == Function.Polinomial_Model:
    coeffs = np.array([0.5, -20., 300, 1000.])
    model_name = 'cubic'
    gen_model = Data.polinomial_model
elif model == Function.Exponential_Model:
    coeffs = np.array([5000., 4000., 0.2])
    model_name = 'exp'
    gen_model = Data.exponential_model
else:
    coeffs = np.array([6000, -5000., -0.2, -3.7])
    model_name = 'logistic'
    gen_model = Data.logistic_model

# read dataset
csv_filename = model_name + '_' + str(r) + '_data_' + '.csv'
n = len(coeffs)
dataset, real_index = generateDatasets(gen_model, coeffs, csv_filename, r, m, out_ratio=out_ratio)

# create model function
p = len(real_index)
model = model(n, p, dataset)

# initial point
x0 = coeffs + np.random.uniform(-1., 1., coeffs.shape[0])

# find optimum and number of trusted points
p_min = int(.8 * r)
p_max = r
xopt, indexes = Optimizacion.RAFF(x0, model, p_min, p_max, max_iter=max_iter)

# new dataset without outliers
dataset = model.dataset[indexes, :]

# plot result
plt.scatter(dataset[:, 0], dataset[:, -1])

y = gen_model(dataset[:, 0].reshape((len(indexes), 1)), xopt)

plt.plot(dataset[:, 0], y, color='blue')
plt.title("Proposed model based in trusted points")
plt.show()


# # ------------------------------------------------ chew data ------------------------------------------------

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
