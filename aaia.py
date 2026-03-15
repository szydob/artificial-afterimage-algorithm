import numpy as np


def calculate_visual_angle(population, best_solution):
    """Calculate the visual angle based on the population and best solution.
    params:
        population: numpy array of shape (population_size, n_features)
        best_solution: numpy array of shape (n_features,)
    returns: visual angle (numpy array of shape (population_size, n_features))
    """
    return 2 * np.arctan(population / (2 * (population - best_solution) + 1e-10))


def calculate_perceptual_size(visual_angle, population, best_solution):
    """
    Calculate the perceptual size based on visual angle, population, and best solution.
    params:
        visual_angle: numpy array of shape (population_size, n_features)
        population: numpy array of shape (population_size, n_features)
        best_solution: numpy array of shape (n_features,)
    returns: perceptual size (numpy array of shape (population_size, n_features))
    """
    return visual_angle * (population - best_solution)


def calculate_population(perceptual_size, best_solution, worst_solution, population):
    """
    Update the population based on perceptual size, best solution, and worst solution.
    params:
        perceptual_size: numpy array of shape (population_size, n_features)
        best_solution: numpy array of shape (n_features,)
        worst_solution: numpy array of shape (n_features,)
        population: numpy array of shape (population_size, n_features)
    returns: updated population (numpy array of shape (population_size, n_features))"""
    return (
        perceptual_size
        + (
            np.abs(
                perceptual_size - (best_solution - np.abs(worst_solution - population))
            )
        )
        * np.random.rand()
    )


def manhattan_distance(population, best_solution):
    """
    Calculate the Manhattan distance from each candidate in the population to the best solution.
    params:
        population: numpy array of shape (population_size, n_features)
        best_solution: numpy array of shape (n_features,)
    returns: numpy array of shape (population_size,) with Manhattan distances
    """
    return np.sum(np.abs(population - best_solution), axis=1)


def fitness(candidate, data):
    """Sum of manhattan distances of each candidate to every other data point
    params:
        candidate: numpy array of shape (n_features,)
        data: numpy array of shape (n_samples, n_features)
    returns: sum of manhattan distances (float)"""
    return np.sum(np.abs(data - candidate))


def calculate_best_and_worst(population, data):
    """
    Calculate the best and worst candidates in the population based on their fitness (sum of Manhattan distances to all data points).
    params:
        population: numpy array of shape (population_size, n_features)
        data: numpy array of shape (n_samples, n_features)
    returns:
        best candidate (numpy array of shape (n_features,)),
        worst candidate (numpy array of shape (n_features,))
    """
    scores = np.array([fitness(candidate, data) for candidate in population])
    best = population[np.argmin(scores)]
    worst = population[np.argmax(scores)]
    return best, worst


def find_solution(
    data,
    max_iterations=1000,
    population_size=50,
    return_history=False,
    early_stopping_rounds=0,
    tol=0.0,
):
    """
    Find a single solution that minimizes the sum of Manhattan distances to all data points using an AAIA-inspired algorithm.
    params:
        data: numpy array of shape (n_samples, n_features)
        max_iterations: maximum number of iterations to run the algorithm
        population_size: number of candidate solutions to maintain in each iteration
        return_history: if True, also return centroid history across iterations
        early_stopping_rounds: if > 0, stop if no improvement in best solution for this many rounds
        tol: minimum improvement in best solution to reset early stopping counters
    returns: best solution found (numpy array of shape (n_features,))
    """
    count = 0
    # initialize population within the data range, not [0,1]
    low = data.min(axis=0)
    high = data.max(axis=0)
    population = np.random.uniform(low, high, size=(population_size, data.shape[1]))
    best_solution, worst_solution = calculate_best_and_worst(population, data)
    history = [best_solution.copy()]
    best_score = fitness(best_solution, data)
    no_improve = 0

    while count < max_iterations:
        V = calculate_visual_angle(population, best_solution)
        S = calculate_perceptual_size(V, population, best_solution)

        local_best, local_worst = calculate_best_and_worst(population, data)

        local_best_score = fitness(local_best, data)
        if local_best_score + tol < best_score:
            best_solution = local_best
            best_score = local_best_score
            no_improve = 0
        else:
            no_improve += 1
        if fitness(local_worst, data) > fitness(worst_solution, data):
            worst_solution = local_worst

        history.append(best_solution.copy())

        population = calculate_population(S, best_solution, worst_solution, population)
        # clip to data range to prevent overflow
        population = np.clip(population, low, high)
        count += 1

        if early_stopping_rounds and no_improve >= early_stopping_rounds:
            break

    iterations_done = count
    if return_history:
        return best_solution, np.array(history), iterations_done

    return best_solution