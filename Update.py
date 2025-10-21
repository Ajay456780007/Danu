import numpy as np

def Update(pop, Vt, t, acc, n, L, eye):
    pop1 = np.zeros((n, L))
    for i in range(n):
        for j in range(L):
            if i < 3:  # MATLAB i<=3 means i=1,2,3; Python zero-based so 0,1,2
                pop1[i, j] = Vt[i, j] * t[i] + 0.5 * acc[i, j] * (t[i] ** 2)
            else:
                if eye == 1:
                    pop1[i, j] = Vt[i, j] * t[i] - 0.5 * acc[i, j] * (t[i] ** 2)
                else:
                    pop1[i, j] = Vt[i, j] * t[i] + 0.5 * acc[i, j] * (t[i] ** 2)
    return pop1
