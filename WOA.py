import numpy as np


def WOA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    lowerbound = np.ones(dimension) * lowerbound  # Lower limit for variables
    upperbound = np.ones(dimension) * upperbound  # Upper limit for variables

    # === INITIALIZATION ===
    X = np.zeros((SearchAgents, dimension))
    for i in range(dimension):
        X[:, i] = lowerbound[i] + np.random.rand(SearchAgents) * (upperbound[i] - lowerbound[i])  # Initial population

    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    # === MAIN LOOP ===
    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)

    for t in range(Max_iterations):
        # Update the best candidate solution
        best = np.min(fit)
        location = np.argmin(fit)

        if t == 0:
            Xbest = X[location, :].copy()
            fbest = best
        elif best < fbest:
            fbest = best
            Xbest = X[location, :].copy()

        SW = Xbest.copy()  # Strongest walrus (best solution)

        # === ITERATE OVER SEARCH AGENTS ===
        for i in range(SearchAgents):

            # === PHASE 1: FEEDING STRATEGY (EXPLORATION) ===
            I = np.round(1 + np.random.rand())
            X_P1 = X[i, :] + np.random.rand(dimension) * (SW - I * X[i, :])  # Eq(3)
            X_P1 = np.maximum(X_P1, lowerbound)
            X_P1 = np.minimum(X_P1, upperbound)

            L = X_P1.copy()
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1
            # === END PHASE 1 ===

            # === PHASE 2: MIGRATION ===
            I = np.round(1 + np.random.rand())
            K = np.random.permutation(SearchAgents)
            K = K[K != i]  # Remove the current index
            X_K = X[K[0], :]
            F_RAND = fit[K[0]]

            if fit[i] > F_RAND:
                X_P2 = X[i, :] + np.random.rand() * (X_K - I * X[i, :])  # Eq(5)
            else:
                X_P2 = X[i, :] + np.random.rand() * (X[i, :] - X_K)  # Eq(5 alt)

            X_P2 = np.maximum(X_P2, lowerbound)
            X_P2 = np.minimum(X_P2, upperbound)

            L = X_P2.copy()
            F_P2 = fitness(L)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2
            # === END PHASE 2 ===

            # === PHASE 3: ESCAPING AND FIGHTING AGAINST PREDATORS (EXPLOITATION) ===
            LO_LOCAL = lowerbound / (t + 1)  # Eq(8) (avoid divide-by-zero)
            HI_LOCAL = upperbound / (t + 1)
            I = np.round(1 + np.random.rand())

            X_P3 = X[i, :] + LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL)  # Eq(7)
            X_P3 = np.maximum(X_P3, LO_LOCAL)
            X_P3 = np.minimum(X_P3, HI_LOCAL)
            X_P3 = np.maximum(X_P3, lowerbound)
            X_P3 = np.minimum(X_P3, upperbound)

            L = X_P3.copy()
            F_P3 = fitness(L)
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3
            # === END PHASE 3 ===

        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    Best_score = fbest
    Best_pos = Xbest
    WOA_curve = best_so_far

    return Best_score, Best_pos, WOA_curve
