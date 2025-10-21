import numpy as np

def check(pop, n, L, ub, lb, acc, Vt, t):
    pop1 = np.copy(pop)
    acc1 = np.copy(acc)
    t1 = np.copy(t)
    Vt1 = np.copy(Vt)

    for i in range(n):
        for j in range(L):
            # Get scalar bounds for the current dimension
            lb_val = lb[j] if isinstance(lb, (np.ndarray, list)) else lb
            ub_val = ub[j] if isinstance(ub, (np.ndarray, list)) else ub

            # Check if out-of-bounds or zero, reinitialize
            if pop1[i, j] >= ub_val or pop1[i, j] <= lb_val or pop1[i, j] == 0 or np.isnan(pop1[i, j]):
                pop1[i, j] = np.random.rand() * (ub_val - lb_val) + lb_val
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()

            # Check acceleration and velocity
            if np.isnan(acc1[i, j]) or acc1[i, j] == 0:
                acc1[i, j] = np.random.rand()
                pop1[i, j] = np.random.rand() * (ub_val - lb_val) + lb_val
                t1[i] = np.random.rand()

            if np.isnan(Vt1[i, j]) or Vt1[i, j] == 0:
                Vt1[i, j] = np.random.rand()
                pop1[i, j] = np.random.rand() * (ub_val - lb_val) + lb_val
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()

        # Check time for each individual
        if np.isnan(t1[i]) or t1[i] == 0:
            t1[i] = np.random.rand()
            for j in range(L):
                lb_val = lb[j] if isinstance(lb, (np.ndarray, list)) else lb
                ub_val = ub[j] if isinstance(ub, (np.ndarray, list)) else ub
                pop1[i, j] = np.random.rand() * (ub_val - lb_val) + lb_val
                acc1[i, j] = np.random.rand()
                Vt1[i, j] = np.random.rand()

    return pop1, acc1, t1, Vt1
