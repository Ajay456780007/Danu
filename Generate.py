import numpy as np


def Generate(n, L, ub, lb):
    ub = [ub] if isinstance(ub, int) else ub
    lb = [lb] if isinstance(lb, int) else lb
    Boundary_no = len(ub)  # number of boundaries
    a = np.zeros((n, L))

    # If the boundaries of all variables are equal and user enters a single
    # number for both ub and lb
    if Boundary_no == 1:
        a = np.random.rand(n, L) * (ub[0] - lb[0]) + lb[0]

    # If each variable has a different lb and ub
    if Boundary_no > 1:
        Positions = np.zeros((n, L))
        for i in range(L):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(n) * (ub_i - lb_i) + lb_i
        a = Positions

    x = np.random.rand(n, L)
    M = a

    return M, x
