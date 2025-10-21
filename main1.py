# Example usage
from optimizations.Walrus_opt.WOA import WOA
from optimizations.Walrus_opt.fun_info import fun_info

# 1. Specify which benchmark function you want to optimize, e.g., 'F1'
F = 'F1'


def Walrus_opt(epochs, pop_size):
    # 2. Get problem info and fitness function
    lowerbound, upperbound, dimension, fitness = fun_info(F)

    # 3. Set WOA parameters
    SearchAgents = pop_size
    Max_iterations = epochs

    # 4. Run the WOA optimizer
    Best_score, Best_pos, WOA_curve = WOA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

    # 5. Print results (optional)
    print("Best score found:", Best_score)
    print("Best position:", Best_pos)

    # 6. Return results just like BCO
    return Best_pos, Best_score
