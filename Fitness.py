import numpy as np


def Fitness(initP, n, L, ub, lb, fobj):
    fit1 = np.zeros(n)

    for i in range(initP.shape[0]):
        fit1[i] = fobj(initP[i, :])

    # Find min value of all fitness (as minimization functions are considered)
    # maxf - best fitness value
    # pos - position of best fitness
    maxf = np.min(fit1)
    pos = np.argmin(fit1)

    return fit1, maxf, pos
