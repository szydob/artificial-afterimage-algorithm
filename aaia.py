import numpy as np

def calculate_visual_angle(population, best_solution):
    """Calculate the visual angle based on the population and best solution.
    params:
        population: numpy array of shape (population_size, n_features)
        best_solution: numpy array of shape (n_features,)
    returns: visual angle (numpy array of shape (population_size, n_features))
        """
    return 2 * np.arctan(population/(2 * (population - best_solution) + 1e-10))

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
    return perceptual_size + (np.abs(perceptual_size - (best_solution - np.abs(worst_solution - population)))) * np.random.rand()
    
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

def find_solution(data, max_iterations=1000, population_size=50):
    """
    Find a single solution that minimizes the sum of Manhattan distances to all data points using an AAIA-inspired algorithm.
    params:
        data: numpy array of shape (n_samples, n_features)
        max_iterations: maximum number of iterations to run the algorithm
        population_size: number of candidate solutions to maintain in each iteration
    returns: best solution found (numpy array of shape (n_features,))
    """
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


def classify(X_train, y_train, X_test, best_solution):
    """Assign test points to clusters using Manhattan distance ranges from training data.
    params:
        X_train: numpy array of shape (n_train_samples, n_features)
        y_train: numpy array of shape (n_train_samples,) with class labels
        X_test: numpy array of shape (n_test_samples, n_features)
        best_solution: numpy array of shape (n_features,) found by AAIA
    returns:
        labels: numpy array of shape (n_test_samples,) with predicted class labels for test data
        thresholds: list of distance thresholds used for classification
        distances_train: numpy array of shape (n_train_samples,) with distances from training points to
        label_map: dict mapping raw label indices to original class labels
    """
    n_clusters = len(np.unique(y_train))
    distances_train = np.sum(np.abs(X_train - best_solution), axis=1)

    class_ranges = {}
    for cls in range(n_clusters):
        mask = y_train == cls
        class_ranges[cls] = (distances_train[mask].min(), distances_train[mask].max())

    sorted_classes = sorted(class_ranges, key=lambda c: np.mean(distances_train[y_train == c]))
    thresholds = [(class_ranges[sorted_classes[i]][1] + class_ranges[sorted_classes[i+1]][0]) / 2
                  for i in range(n_clusters - 1)]

    distances_test = np.sum(np.abs(X_test - best_solution), axis=1)
    raw_labels = np.digitize(distances_test, thresholds)
    label_map = {i: sorted_classes[i] for i in range(n_clusters)}
    labels = np.array([label_map[l] for l in raw_labels])

    return labels, thresholds, distances_train, label_map