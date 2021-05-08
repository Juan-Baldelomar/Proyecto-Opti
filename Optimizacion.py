
import numpy as np
import Function as TestFunction

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

        # x_p, f_model = optimizar(x0, f_model)
        # solutions.append(x_p)
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




