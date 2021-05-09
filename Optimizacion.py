import numpy as np
import Function as TestFunction
from scipy.linalg import solve_triangular


def check_tolerance(value, value_new, epislon=1e-8):
    """
    Evaluación en los criterios de paro por distancias relativas si está bien definido el cociente.
    Evaluación en los criterios de paro por distancias absolutas, en otro caso.

    :param value: valor anterior.
    :param value_new: valor nuevo.
    :param epislon: tolerancia.
    :return: Verdadero si la evalución es menor a la tolerancia. Falso en otro caso.
    """
    # Evaluación para escalares
    if isinstance(value, (float, np.float)) and isinstance(value_new, (float, np.float)):
        return np.abs(value - value_new) / max(1., np.abs(value)) < epislon

    # Evaluación para vectores
    else:
        return np.linalg.norm(value - value_new) / max(1., np.linalg.norm(value)) < epislon


def compute_direction(J: np.ndarray, gamma: float, g: np.ndarray):
    A = np.matmul(J.T, J) + np.diag([gamma]*J.shape[1])
    try:
        L = np.linalg.cholesky(A)
        y = solve_triangular(L, -g, lower=True)
        d = solve_triangular(L.T, y, lower=False)
        return d
    except np.linalg.LinAlgError:
        print('\'A\' cholesky failed')
        return None


def lm_lovo(x: np.ndarray, lmbda_min: float, epsilon: float, lmbda_0: float,
            lmbda_hat: float, test_function: TestFunction, max_iter: int = 400):

    assert (lmbda_0 > 0. and lmbda_hat > 1.)

    # Initializaton
    lmbda = lmbda_0
    g = test_function.gradient(x)

    for k in range(max_iter):

        # Stoppage criteria
        g_norm = test_function.norm_g_k[-1]
        if check_tolerance(g_norm, 0., epsilon):
            break

        # Report
        if k % 1 == 0:
            print(" ")
            print("Iteration n.", k, "results:")
            print("|g(x)| = ", g_norm)
            print("f(x) = ", test_function.function(x))

        while True:
            # Calculate direction
            gamma = lmbda * g_norm * g_norm
            d = compute_direction(test_function.Jacobian(x), gamma, g)

            # Simple decrease test: (trust-region simplification)
            if test_function.function(x+d, False) < test_function.function(x, False):
                break

        # Actualization
        lmbda = max(lmbda_min, lmbda/np.sqrt(lmbda))  # TODO: in [max(lmbda_min, lmbda/np.sqrt(lmbda)), lmbda]
        x = x + d
        g = test_function.function(x)

    return x
