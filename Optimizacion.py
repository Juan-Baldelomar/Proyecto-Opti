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
        print("gradient: ", g)
        L = np.linalg.cholesky(A)
        y = solve_triangular(L, -g, lower=True)
        print("y: ", y)
        d = solve_triangular(L.T, y, lower=False)
        return d
    except np.linalg.LinAlgError:
        print('\'A\' cholesky failed')
        return None


def lm_lovo(x: np.ndarray, lmbda_min: float, epsilon: float, lmbda_0: float,
            lmbda_hat: float, test_function: TestFunction, max_iter: int = 400):

    assert (lmbda_0 > 0. and lmbda_hat > 1.)

    # get I_min (combination where first p values of Ri are the lowest)
    test_function.getR(x)

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
            print("f(x) = ", test_function.S(x))

        while True:
            # Calculate direction
            gamma = lmbda * g_norm * g_norm
            d = compute_direction(test_function.Jacobian(x), gamma, g)

            # Simple decrease test: (trust-region simplification)
            if test_function.S(x+d, False) < test_function.S(x, False):
                break
            else:
                lmbda = lmbda_hat * lmbda

        # Actualization
        lmbda = max(lmbda_min, lmbda / np.sqrt(lmbda))  # TODO: in [max(lmbda_min, lmbda/np.sqrt(lmbda)), lmbda]
        x = x + d
        g = test_function.gradient(x)

    return x, g_norm

def buildSimilarityMatrix(solutions, rem_indexes):
    n = len(solutions)
    M = np.zeros((n, n))

    # build similarity matrix taking advantage of its symmetry
    for i in range(n):

        # remove element
        if i in rem_indexes:
            M[i, :] = np.inf
            M[:, i] = np.inf
            continue

        for j in range(i):
            x_p, x_q = solutions[i][0], solutions[j][0]
            M[i][j] = np.linalg.norm(x_p - x_q)
            M[j][i] = M[i][j]

    return M

# S and abs_diff are sorted from min_p to max_p
def preprocess(solutions, S, abs_diff, r, tau):
    indexes = []
    p_min = 0
    S_min = S[0]

    # add indexes that should be eliminated
    for i in range(len(S)):

        # check if lovo did not converge
        if solutions[i][1] > tau:
            indexes.append(i)
            continue

        for j in range(i):
            if S[j] > S[i]:
                indexes.append(j)

        # retrieve model such that we have minimum Sp(xp^*)
        if S[i] < S_min and i not in indexes:
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
        indexes.append(len(solutions)-1)

    return indexes


def RAFF(x0: np.ndarray, f_model: TestFunction, pmin, pmax, epsilon=-1, tau=1e-4, max_iter=400):
    assert (1 <= pmin < pmax <= len(f_model.dataset))

    S = []              # vector of Sp
    abs_diff = []       # vector of abs diff between yi and model(x, ti)
    solutions = []      # vector of tuples (x^*, ||grad(x^*)||) for each pmin <= p <= pmax

    # compute set of solutions for p in range(pmin, pmax)
    for p in range(pmin, pmax):
        f_model.p = p

        # find optimum
        x_p, g_norm = lm_lovo(x0, 0.01, tau, 1, 2, f_model, max_iter=max_iter)
        solutions.append((x_p, g_norm))
        S.append(f_model.S(x_p))
        abs_diff.append(f_model.abs_diff(x_p))

    solutions = np.array(solutions)

    # preprocess solutions
    rem_indexes = preprocess(solutions, S, abs_diff, r=len(f_model.dataset), tau=tau)

    # build similarity matrix
    M = buildSimilarityMatrix(solutions, rem_indexes)
    C = np.zeros(len(M))

    # calc epsilon if it is not given
    M_aux = M[np.isfinite(M)]

    if epsilon == -1:
        epsilon = np.min(M_aux) + np.mean(M_aux) / (1 + np.sqrt(pmax)) if len(M_aux) > 0 else 0

    for i in range(len(M)):
        k = 0
        for j in range(len(M)):
            if M[i][j] < epsilon:
                k += 1
        C[i] = k

    # see position of solutions that have more votes and return that x^*
    max_k, max_p = 0, 0
    for p in range(len(C)):
        if max_k <= C[p]:
            max_k = C[p]
            max_p = p

    # return x^* and maxp number of trusted points
    return solutions[max_p][0], pmin + max_p
