import numpy as np

def Herding(pop, Vt, fit, n, L, a, t):
    idx = np.argsort(fit)
    pop1 = np.zeros((n, L))
    Vt1 = np.zeros((n, L))
    a1 = np.zeros((n, L))
    t1 = np.zeros(n)

    for i in range(n):
        pop1[i, :] = pop[idx[i], :]
        Vt1[i, :] = Vt[idx[i], :]
        a1[i, :] = a[i, :]
        t1[i] = t[i]

    return pop1, Vt1, fit[idx], a1, t1
