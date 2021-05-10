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
            lmbda = max(lmbda_min, lmbda / np.sqrt(lmbda))  # TODO: in [max(lmbda_min, lmbda/np.sqrt(lmbda)), lmbda]
            x = x + d
            g = test_function.function(x)

        return x

def buildSimilarityMatrix(solutions, rem_indexes):
    n = len(solutions)
    M = np.zeros((n, n))

    # build similarity matrix taking advantage of its symmetry
    for i in range(n-1):
        for j in range(i):
            if (i, j) in rem_indexes:
                # remove element
                M[i][j] = np.inf
                M[j][i] = np.inf

            else:
                x_p, x_q = solutions[i], solutions[j]
                M[i][j] = np.linalg.norm(x_p - x_q)
                M[j][i] = M[i][j]

    if (-1, n-1) in rem_indexes:
        # remove last index
        M[: -1] = np.inf
        M[-1, :] = np.inf

    return M

# S and abs_diff are sorted from min_p to max_p
def preprocess(solutions, S, abs_diff, r):
    indexes = []
    p_min = 0
    S_min = S[0]

    # add indexes that should be eliminated
    for i in range(len(S)):
        for j in range(i):
            if S[j] > S[i]:
                indexes.append((i, j))

        # retrieve model such that we have minimum Sp(xp^*)
        if S[i] < S_min:
            S_min = S[i]
            p_min = i

    # get model with lower Sp and model from p_max
    f_mins = abs_diff[p_min]
    f_maxp = abs_diff[-1]

    # retrieve difference for each point in the dataset between y_i predicted and real y_i for f_mins model and f_maxp model
    diff_mins_maxp = f_mins - f_maxp

    # see how many of previous differences are lower than 0
    k = np.sum(diff_mins_maxp < 0)

    # if k > r/2, then max_p model should be eliminated
    if k > r/2:
        indexes.append((-1, len(solutions)-1))

    return indexes


def RAFF(x0: np.ndarray, f_model: TestFunction, pmin, pmax, epsilon=-1):
    assert (epsilon > 0 and pmin >= 0 and pmax < pmin)

    S = []
    abs_diff = []
    solutions = np.array([])

    # compute set of solutions for p in range(pmin, pmax)
    for p in range(pmin, pmax+1):
        f_model.p = p

        # x_p = optimizar(x0, f_model)
        #solutions.append(x_p)
        S.append(f_model.S())
        # abs_diff.append(f_model.abs_diff(x_p))

    # preprocess solutions
    rem_indexes = preprocess(solutions, S, abs_diff, r=len(f_model.dataset))

    # build similarity matrix
    M = buildSimilarityMatrix(solutions, rem_indexes)
    C = np.zeros(len(M))

    for i in len(M):
        k = 0
        for j in len(M):
            if M[i][j] < epsilon:
                k += 1
        C[i] = k

    # see position of solutions that have more votes and return that x^*
    max_k, max_p = 0, 0
    for p in len(C):
        if max_k <= C[p]:
            max_k = C[p]
            max_p = p

    return solutions[max_p]
