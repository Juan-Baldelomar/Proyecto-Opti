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
    coeffs = np.array([0.5, -2., 10., 20])
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
model = model(n, 80, dataset)

# initial point
x0 = coeffs + np.random.normal(0, 1, coeffs.shape)

# find optimum and number of trusted points
p_min = int(.8 * r)
p_max = r
xopt, indexes = Optimizacion.RAFF(x0, model, p_min, p_max, max_iter=max_iter)

# new dataset without outliers
dataset = model.dataset[indexes, :]

# plot result
plt.scatter(dataset[:, 0], dataset[:, -1])

x_model, y_model = dataset[:, 0], gen_model(dataset[:, 0].reshape((len(indexes), 1)), xopt)

# sort model_data by x coordinate
model_data = np.column_stack((x_model, y_model))
model_data = model_data[np.argsort(model_data[:, 0])]


plt.plot(model_data[:, 0], model_data[:, 1], color='blue')
plt.title("Proposed model based in trusted points")
plt.show()



# dataset = Data.readData('linear_data.csv')
# plt.scatter(dataset[:, 0], dataset[:, -1])
# plt.title("Original data")
# plt.show()
# x0 = np.array([1, 1])
