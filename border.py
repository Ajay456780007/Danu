import numpy as np
from optimizations.BCO.Fitness import Fitness
from optimizations.BCO.Func_details import func_details
from optimizations.BCO.Generate import Generate
from optimizations.BCO.Herding import Herding
from optimizations.BCO.Update import Update
from optimizations.BCO.UpdateV import UpdateV
from optimizations.BCO.check import check

def BCO(epochs, pop_size, to_optimize, fitness_func=None):
    """
    epochs: number of generations
    pop_size: population size
    to_optimize: 1D numpy array (flattened weights to optimize)
    fitness_func: optional, a custom fitness function (overrides func_details)
    """

    n = pop_size
    gen = epochs

    # Problem dimension
    L = to_optimize.size

    lb = np.min(to_optimize)
    ub = np.max(to_optimize)

    if fitness_func is not None:
        fobj = fitness_func  # this will be your Keras evaluation
    else:
        # fallback: use benchmark function (optional)
        from optimizations.BCO.Func_details import func_details
        fname = 'func19'
        lb, ub, _, fobj = func_details(fname)

    # Initialize population within bounds
    initP, acc = Generate(n, L, ub, lb)

    # Velocity of each individual
    Vt = np.zeros((n, L))

    # Time of each individual
    t = np.random.rand(n)

    # Max fitness value
    fopt = -np.inf  # we are maximizing

    # Fitness array
    fit = np.zeros(n)

    # Copy initial population
    pop = initP.copy()

    # Counter for "Eyeing" mechanism
    k = 1
    fopt_history = np.zeros(gen)

    for g in range(gen):
        # --- Compute fitness ---
        if fitness_func is not None:
            # Fitness using custom function for your weight matrix
            fit = np.array([fobj(ind) for ind in pop])
            maxf = np.max(fit)
            pos = np.argmax(fit)
        else:
            # Original BCO fitness
            fit, maxf, pos = Fitness(pop, n, L, ub, lb, fobj)

        eye = 0

        if g == 0:
            fopt = maxf

        if fopt < maxf:
            fopt = maxf

        fopt_history[g] = fopt

        # Eyeing mechanism
        if g > 0:
            if fopt_history[g] > fopt_history[g - 1]:
                k += 1
                if k > 5:
                    eye = 1
                    k = 0

        # Herding update
        pop, Vt, fit, acc, t = Herding(pop, Vt, fit, n, L, acc, t)

        # Update velocities and positions
        Vt, acc, t, r1, l1, tempg, temps = UpdateV(Vt, n, L, acc, t, pop, fit, eye)
        pop = Update(pop, Vt, t, acc, n, L, eye)

        # Check bounds
        pop, acc, t, Vt = check(pop, n, L, ub, lb, acc, Vt, t)

        print(f"Gen {g+1}/{gen} -- fopt {fopt:.6f} -- maxf {maxf:.6f}")

    # Return the best individual (solution) and its fitness
    best_index = np.argmax(fit)  # for maximization
    best_solution = pop[best_index]  # shape: (L,)
    best_fitness = fit[best_index]

    # Ensure output is flattened 1D array (works for Keras weight reshape)
    return best_solution.flatten(), best_fitness
def BCO(epochs, pop_size, to_optimize, fitness_func=None):
    n = pop_size
    gen = epochs
    L = to_optimize.size

    lb = np.full(L, np.min(to_optimize), dtype=np.float32)
    ub = np.full(L, np.max(to_optimize), dtype=np.float32)

    # Use the provided fitness function
    fobj = fitness_func
    if fobj is None:
        raise ValueError("A fitness function must be provided for Keras weight optimization")

    # Initialize population and acceleration
    initP, acc = Generate(n, L, ub, lb)
    Vt = np.zeros((n, L))
    t = np.random.rand(n)
    fopt = -np.inf
    fit = np.zeros(n)
    pop = initP.copy()

    k = 1
    fopt1 = np.zeros(gen)

    for g in range(gen):
        # calculate fitness
        fit, maxf, pos = Fitness(pop, n, L, ub, lb, fobj)
        eye = 0

        if g == 0:
            fopt = maxf
        if fopt < maxf:
            fopt = maxf
        fopt1[g] = fopt

        if g > 0 and fopt1[g] > fopt1[g-1]:
            k += 1
            if k > 5:
                eye = 1
                k = 0

        pop, Vt, fit, acc, t = Herding(pop, Vt, fit, n, L, acc, t)
        Vt, acc, t, _, _, _, _ = UpdateV(Vt, n, L, acc, t, pop, fit, eye)
        pop = Update(pop, Vt, t, acc, n, L, eye)
        pop, acc, t, Vt = check(pop, n, L, ub, lb, acc, Vt, t)

        print(f"\n\nfopt {fopt:.6f}\t maxf {maxf:.6f}\t gen {g + 1}")

    best_index = np.argmax(fit)
    best_solution = pop[best_index]
    best_fitness = fit[best_index]
    return best_solution, best_fitness
