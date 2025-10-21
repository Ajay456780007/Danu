import numpy as np


def UpdateV(Vt, n, L, a, t, pop, fit, eye):
    r1 = np.random.choice([2, 3])
    l1 = 3 if r1 == 2 else 2

    acc = np.ones((n, L))
    t1 = np.ones(n)
    Vt1 = np.copy(Vt)

    r, l = a.shape
    a1 = np.zeros((r, l))
    f1 = fit[0]
    f2 = (fit[1] + fit[2]) / 2
    f = 0
    tempg = np.zeros(n)
    temps = np.zeros(n)

    if eye == 1:
        if fit[r1 - 1] < fit[l1 - 1]:  # Adjust index for zero-based Python
            a1[l1 - 1, :] = -a[l1 - 1, :]
            f = l1 - 1
        else:
            a1[r1 - 1, :] = -a[r1 - 1, :]
            f = r1 - 1

    for i in range(n):
        for j in range(L):
            if i <= 2:  # MATLAB i<=3 means Python i<=2 (zero based)
                Vt1[i, j] = np.sqrt(Vt[i, j] ** 2 + 2 * a[i, j] * pop[i, j]).real
            elif i > 2:
                if eye == 1:
                    Vt1[i, j] = np.sqrt(Vt1[f, j] ** 2 + 2 * a1[f, j] * pop[i, j]).real
                else:
                    if f1 - fit[i] > f2 - fit[i]:
                        Vt1[i, j] = np.sqrt(Vt1[0, j] ** 2 + 2 * a[0, j] * pop[i, j]).real
                        tempg[i] = i
                    else:
                        rand_angle1 = np.tan(np.deg2rad(np.random.randint(1, 90)))  # degrees from 1 to 89
                        rand_angle2 = np.tan(np.deg2rad(np.random.randint(91, 180)))  # degrees from 91 to 179
                        val1 = Vt1[r1 - 1, j] ** 2 * (rand_angle1 ** 2) + 2 * a[r1 - 1, j] * pop[r1 - 1, j]
                        val2 = Vt1[l1 - 1, j] ** 2 * (rand_angle2 ** 2) + 2 * a[l1 - 1, j] * pop[l1 - 1, j]
                        Vt1[i, j] = (np.sqrt(val1) + np.sqrt(val2)) / 2
                        temps[i] = i

    for i in range(n):
        s = 0
        for j in range(L):
            acc[i, j] = abs(Vt1[i, j] - Vt[i, j]) / t[i]
            s += (Vt1[i, j] - Vt[i, j]) / acc[i, j] if acc[i, j] != 0 else 0
        t1[i] = abs(s / L)  # mean over L components

    return Vt1, acc, t1, r1, l1, tempg, temps
