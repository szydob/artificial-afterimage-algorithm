import numpy as np

def calculate_visual_angle(population, best_solution):
    return 2 * np.arctan(population/(2 * (population - best_solution) + 1e-10))

def calculate_perceptual_size(visual_angle, population, best_solution):
    return visual_angle * (population - best_solution)

def calculate_population(perceptual_size, best_solution, worst_solution, population):
    return perceptual_size + (np.abs(perceptual_size - (best_solution - np.abs(worst_solution - population)))) * np.random.rand()
    
def manhattan_distance(population, best_solution):
    return np.sum(np.abs(population - best_solution), axis=1)

def fitness(candidate, data):
    """sum of manhattan distances of each candidate to every other data point"""
    return np.sum(np.abs(data - candidate))

def calculate_best_and_worst(population, data):
    scores = np.array([fitness(candidate, data) for candidate in population])
    best = population[np.argmin(scores)]
    worst = population[np.argmax(scores)]
    return best, worst

def find_solution(data, max_iterations=1000, population_size=50):
    count = 0
    # initialize population within the data range, not [0,1]
    low = data.min(axis=0)
    high = data.max(axis=0)
    population = np.random.uniform(low, high, size=(population_size, data.shape[1]))
    best_solution, worst_solution = calculate_best_and_worst(population, data)

    while count < max_iterations:
        V = calculate_visual_angle(population, best_solution)
        S = calculate_perceptual_size(V, population, best_solution)

        local_best, local_worst = calculate_best_and_worst(population, data)

        if fitness(local_best, data) < fitness(best_solution, data):
            best_solution = local_best
        if fitness(local_worst, data) > fitness(worst_solution, data):
            worst_solution = local_worst

        population = calculate_population(S, best_solution, worst_solution, population)
        # clip to data range to prevent overflow
        population = np.clip(population, low, high)
        count += 1

    return best_solution