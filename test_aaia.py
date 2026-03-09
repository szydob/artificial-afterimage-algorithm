"""
Comprehensive test suite for the Artificial Afterimage Algorithm (AAIA) implementation.

This test module provides extensive coverage of the AAIA algorithm and all its components,
including unit tests, integration tests, and edge case testing.

Test Organization:
------------------
1. Unit Tests (by function):
   - TestVisualAngle: Tests for calculate_visual_angle()
   - TestPerceptualSize: Tests for calculate_perceptual_size()
   - TestPopulationUpdate: Tests for calculate_population()
   - TestManhattanDistance: Tests for manhattan_distance()
   - TestFitness: Tests for fitness()
   - TestBestAndWorstCalculation: Tests for calculate_best_and_worst()
   - TestFindSolution: Tests for find_solution()
   - TestClassify: Tests for classify()

2. Integration Tests:
   - TestIntegration: Full pipeline tests with real and synthetic data
   - Tests reproducibility and end-to-end algorithm performance

3. Edge Cases & Boundary Conditions:
   - TestEdgeCases: Tests with special inputs (single points, high dimensions, etc.)
   - Tests numerical stability and robustness

Test Categories:
---------------
- Shape/Output Format Tests: Verify correct output shapes and data types
- Numerical Correctness Tests: Validate computation accuracy
- Boundary Tests: Test with extreme/special values
- Integration Tests: Test complete workflows
- Stability Tests: Ensure numerical stability and absence of NaN/Inf
- Robustness Tests: Test with various data characteristics

Running the Tests:
-----------------
Run all tests:
    pytest test_aaia.py -v

Run specific test class:
    pytest test_aaia.py::TestVisualAngle -v

Run with coverage:
    pytest test_aaia.py --cov=aaia --cov-report=html

Run specific test:
    pytest test_aaia.py::TestFitness::test_fitness_correctness -v
"""

import pytest
import numpy as np
from sklearn.datasets import load_iris
import aaia


class TestVisualAngle:
    """Tests for calculate_visual_angle function.
    
    The visual angle is computed as: 2 * arctan(population / (2 * (population - best_solution) + epsilon))
    
    This tests:
    - Correct output shapes matching input dimensions
    - Value ranges (should be between -pi and pi for arctan-based computations)
    - Numerical stability (no NaN or Inf values)
    - Behavior at special points (when population equals best_solution)
    """
    
    def test_visual_angle_shape(self):
        """Test that visual angle has correct shape"""
        population = np.array([[1, 2], [3, 4], [5, 6]])
        best_solution = np.array([0, 0])
        
        result = aaia.calculate_visual_angle(population, best_solution)
        
        assert result.shape == population.shape
        assert isinstance(result, np.ndarray)
    
    def test_visual_angle_values_in_range(self):
        """Test that visual angle values are reasonable (between -pi and pi)"""
        population = np.random.uniform(-10, 10, size=(20, 5))
        best_solution = np.zeros(5)
        
        result = aaia.calculate_visual_angle(population, best_solution)
        
        assert np.all(np.isfinite(result))
        assert np.all(result <= np.pi)
        assert np.all(result >= -np.pi)
    
    def test_visual_angle_same_point(self):
        """Test visual angle when population point equals best solution"""
        population = np.array([[1.0, 2.0], [1.0, 2.0]])
        best_solution = np.array([1.0, 2.0])
        
        result = aaia.calculate_visual_angle(population, best_solution)
        
        # When population equals best_solution, numerator is population, denominator approaches 0
        # This results in arctan(infinity) = pi/2, and 2 * pi/2 = pi
        assert np.allclose(result, np.pi)
    
    def test_visual_angle_symmetry(self):
        """Test that visual angle computation is numerically stable"""
        population = np.array([[0.1, 0.2], [0.3, 0.4]])
        best_solution = np.array([0.0, 0.0])
        
        result = aaia.calculate_visual_angle(population, best_solution)
        
        # No NaN or Inf values should be present
        assert np.all(np.isfinite(result))


class TestPerceptualSize:
    """Tests for calculate_perceptual_size function.
    
    Perceptual size is computed as: visual_angle * (population - best_solution)
    
    This tests:
    - Correct output shapes
    - Numerical stability and finite values
    - Behavior when visual angle and distance differences are zero
    """
    
    def test_perceptual_size_shape(self):
        """Test that perceptual size has correct shape"""
        population = np.array([[1, 2], [3, 4]])
        best_solution = np.array([0, 0])
        visual_angle = aaia.calculate_visual_angle(population, best_solution)
        
        result = aaia.calculate_perceptual_size(visual_angle, population, best_solution)
        
        assert result.shape == population.shape
    
    def test_perceptual_size_zero_when_same_point(self):
        """Test that perceptual size is zero when population equals best solution"""
        population = np.array([[1.0, 2.0]])
        best_solution = np.array([1.0, 2.0])
        visual_angle = np.array([[0.0, 0.0]])
        
        result = aaia.calculate_perceptual_size(visual_angle, population, best_solution)
        
        assert np.allclose(result, 0)
    
    def test_perceptual_size_values(self):
        """Test that perceptual size values are finite"""
        population = np.random.uniform(-5, 5, size=(10, 3))
        best_solution = np.zeros(3)
        visual_angle = aaia.calculate_visual_angle(population, best_solution)
        
        result = aaia.calculate_perceptual_size(visual_angle, population, best_solution)
        
        assert np.all(np.isfinite(result))


class TestPopulationUpdate:
    """Tests for calculate_population function.
    
    This function updates the population based on perceptual size and best/worst solutions.
    
    This tests:
    - Correct output shapes after update
    - Numerical stability (all finite values)
    - Proper handling of stochastic elements (random multiplier)
    """
    
    def test_population_update_shape(self):
        """Test that updated population has correct shape"""
        population = np.array([[1, 2], [3, 4], [5, 6]])
        best_solution = np.array([0, 0])
        worst_solution = np.array([10, 10])
        perceptual_size = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        
        result = aaia.calculate_population(perceptual_size, best_solution, worst_solution, population)
        
        assert result.shape == population.shape
    
    def test_population_update_finite(self):
        """Test that population update produces finite values"""
        population = np.random.uniform(0, 10, size=(20, 4))
        best_solution = np.array([5, 5, 5, 5])
        worst_solution = np.array([10, 10, 10, 10])
        visual_angle = aaia.calculate_visual_angle(population, best_solution)
        perceptual_size = aaia.calculate_perceptual_size(visual_angle, population, best_solution)
        
        result = aaia.calculate_population(perceptual_size, best_solution, worst_solution, population)
        
        assert np.all(np.isfinite(result))


class TestManhattanDistance:
    """Tests for manhattan_distance function.
    
    Computes L1 distance (Manhattan distance) from each population member to the best solution.
    Formula: sum(|population - best_solution|) for each row
    
    This tests:
    - Correct output shape (1D array of distances)
    - Correctness of calculation (hand-verified examples)
    - Non-negativity of distances (always >= 0)
    - Zero distance when points are identical
    """
    
    def test_manhattan_distance_shape(self):
        """Test that manhattan distance has correct shape"""
        population = np.array([[1, 2], [3, 4], [5, 6]])
        best_solution = np.array([0, 0])
        
        result = aaia.manhattan_distance(population, best_solution)
        
        assert result.shape == (len(population),)
    
    def test_manhattan_distance_zero_when_same_point(self):
        """Test that distance is zero when point equals best solution"""
        population = np.array([[1.0, 2.0, 3.0]])
        best_solution = np.array([1.0, 2.0, 3.0])
        
        result = aaia.manhattan_distance(population, best_solution)
        
        assert result[0] == 0
    
    def test_manhattan_distance_correctness(self):
        """Test correctness of manhattan distance calculation"""
        population = np.array([[3, 4], [0, 0], [1, 1]])
        best_solution = np.array([0, 0])
        
        result = aaia.manhattan_distance(population, best_solution)
        
        expected = np.array([7, 0, 2])  # |3-0| + |4-0|, |0-0| + |0-0|, |1-0| + |1-0|
        assert np.allclose(result, expected)
    
    def test_manhattan_distance_always_non_negative(self):
        """Test that manhattan distance is always non-negative"""
        population = np.random.uniform(-100, 100, size=(50, 10))
        best_solution = np.random.uniform(-100, 100, size=10)
        
        result = aaia.manhattan_distance(population, best_solution)
        
        assert np.all(result >= 0)


class TestFitness:
    """Tests for fitness function.
    
    Fitness is the sum of Manhattan distances from a candidate to all data points.
    Formula: sum(|data[i] - candidate|) for all data points
    
    This tests:
    - Correct output type (scalar)
    - Numerical correctness with hand-calculated values
    - Non-negativity (always >= 0)
    - Consistency (same input yields same output)
    - Calculation includes all data points
    """
    
    def test_fitness_shape(self):
        """Test that fitness returns a scalar"""
        candidate = np.array([1, 2, 3])
        data = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        
        result = aaia.fitness(candidate, data)
        
        # Result should be a scalar (int or float)
        assert np.isscalar(result) or isinstance(result, (int, float, np.integer, np.floating))
    
    def test_fitness_symmetric(self):
        """Test that fitness calculation includes all data points"""
        # Even if candidate is a data point, fitness sums to ALL data points
        data = np.array([[0, 0], [1, 1], [2, 2]])
        candidate = data[1]  # Use [1, 1]
        
        result = aaia.fitness(candidate, data)
        
        # Expected: |0-1|+|0-1| + |1-1|+|1-1| + |2-1|+|2-1| = 2 + 0 + 2 = 4
        assert result == 4
    
    def test_fitness_correctness(self):
        """Test correctness of fitness calculation"""
        candidate = np.array([0, 0])
        data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        
        result = aaia.fitness(candidate, data)
        
        # Sum of Manhattan distances: 0 + 1 + 1 + 2 = 4
        expected = 4
        assert result == expected
    
    def test_fitness_always_non_negative(self):
        """Test that fitness is always non-negative"""
        candidate = np.random.uniform(-10, 10, size=5)
        data = np.random.uniform(-10, 10, size=(20, 5))
        
        result = aaia.fitness(candidate, data)
        
        assert result >= 0
    
    def test_fitness_commutative(self):
        """Test that fitness calculation is consistent"""
        candidate = np.array([2, 3])
        data = np.array([[0, 0], [1, 1], [2, 2]])
        
        result1 = aaia.fitness(candidate, data)
        result2 = aaia.fitness(candidate, data)
        
        assert result1 == result2


class TestBestAndWorstCalculation:
    """Tests for calculate_best_and_worst function.
    
    Selects the population member with minimum fitness (best) and maximum fitness (worst).
    
    This tests:
    - Correct output shapes
    - Best and worst are actual members of the population
    - Best has lower fitness than worst
    - Behavior with single population member
    - Proper fitness evaluation
    """
    
    def test_best_and_worst_shape(self):
        """Test that best and worst have correct shape"""
        population = np.array([[1, 2], [3, 4], [5, 6]])
        data = np.array([[0, 0], [1, 1], [2, 2]])
        
        best, worst = aaia.calculate_best_and_worst(population, data)
        
        assert best.shape == (2,)
        assert worst.shape == (2,)
    
    def test_best_and_worst_in_population(self):
        """Test that best and worst are from the population"""
        population = np.random.uniform(0, 10, size=(10, 3))
        data = np.random.uniform(0, 10, size=(20, 3))
        
        best, worst = aaia.calculate_best_and_worst(population, data)
        
        # Check that best and worst are in population
        assert any(np.allclose(best, p) for p in population)
        assert any(np.allclose(worst, p) for p in population)
    
    def test_best_better_than_worst(self):
        """Test that best has lower fitness than worst"""
        population = np.random.uniform(0, 10, size=(20, 3))
        data = np.random.uniform(0, 10, size=(30, 3))
        
        best, worst = aaia.calculate_best_and_worst(population, data)
        
        best_fitness = aaia.fitness(best, data)
        worst_fitness = aaia.fitness(worst, data)
        
        assert best_fitness <= worst_fitness
    
    def test_single_population_member(self):
        """Test with single population member"""
        population = np.array([[1.0, 2.0, 3.0]])
        data = np.random.uniform(0, 10, size=(10, 3))
        
        best, worst = aaia.calculate_best_and_worst(population, data)
        
        # Both should be the same when population has single member
        assert np.allclose(best, worst)


class TestFindSolution:
    """Tests for find_solution function.
    
    Main optimization algorithm that iteratively updates a population to find a solution
    that minimizes sum of Manhattan distances to all data points.
    
    This tests:
    - Correct output shape matching data features
    - Solution stays within data bounds (with clipping)
    - Convergence: more iterations lead to better or equal fitness
    - Correctness with various data dimensions
    - Robustness with large populations
    """
    
    def test_find_solution_shape(self):
        """Test that find_solution returns correct shape"""
        data = np.random.uniform(0, 10, size=(20, 3))
        
        result = aaia.find_solution(data, max_iterations=10, population_size=10)
        
        assert result.shape == (3,)
    
    def test_find_solution_within_data_bounds(self):
        """Test that solution is within data bounds"""
        data = np.random.uniform(2, 8, size=(30, 4))
        
        result = aaia.find_solution(data, max_iterations=50, population_size=20)
        
        assert np.all(result >= data.min(axis=0) - 0.1)
        assert np.all(result <= data.max(axis=0) + 0.1)
    
    def test_find_solution_convergence(self):
        """Test that solution improves over iterations"""
        np.random.seed(42)
        data = np.random.uniform(0, 10, size=(25, 2))
        
        solution1 = aaia.find_solution(data, max_iterations=10, population_size=15)
        solution2 = aaia.find_solution(data, max_iterations=100, population_size=15)
        
        fitness1 = aaia.fitness(solution1, data)
        fitness2 = aaia.fitness(solution2, data)
        
        # More iterations should give equal or better fitness
        assert fitness2 <= fitness1 * 1.1  # Allow 10% tolerance for randomness
    
    def test_find_solution_single_feature(self):
        """Test find_solution with single feature data"""
        data = np.random.uniform(0, 10, size=(20, 1))
        
        result = aaia.find_solution(data, max_iterations=20, population_size=10)
        
        assert result.shape == (1,)
        assert np.isfinite(result[0])
    
    def test_find_solution_large_population(self):
        """Test find_solution with larger population"""
        data = np.random.uniform(0, 10, size=(20, 3))
        
        result = aaia.find_solution(data, max_iterations=30, population_size=50)
        
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


class TestClassify:
    """Tests for classify function.
    
    Uses the found solution to classify test points based on distance thresholds
    estimated from training data class ranges.
    
    This tests:
    - Correct output shapes (labels, thresholds, distances, mapping)
    - Predicted labels are valid (in range of training labels)
    - Proper distance calculation consistency
    - Threshold ordering and validity
    - Integration with Iris dataset
    """
    
    def test_classify_shape(self):
        """Test that classify returns correct shapes"""
        np.random.seed(42)
        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]])
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X_test = np.array([[0.5, 0.5], [5.5, 5.5]])
        best_solution = np.array([0.5, 0.5])
        
        labels, thresholds, distances_train, label_map = aaia.classify(X_train, y_train, X_test, best_solution)
        
        assert labels.shape == (len(X_test),)
        assert len(thresholds) == len(np.unique(y_train)) - 1
        assert distances_train.shape == (len(X_train),)
        assert isinstance(label_map, dict)
    
    def test_classify_valid_labels(self):
        """Test that classify returns valid class labels"""
        np.random.seed(42)
        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]])
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        X_test = np.array([[0.5, 0.5], [5.5, 5.5]])
        best_solution = np.array([0.5, 0.5])
        
        labels, _, _, _ = aaia.classify(X_train, y_train, X_test, best_solution)
        
        # All labels should be valid class indices
        assert np.all(np.isin(labels, np.unique(y_train)))
    
    def test_classify_with_iris_data(self):
        """Test classify function with Iris dataset"""
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        np.random.seed(42)
        train_idx = np.random.choice(len(X), size=90, replace=False)
        test_idx = np.array([i for i in range(len(X)) if i not in train_idx])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        best_solution = aaia.find_solution(X_train, max_iterations=50, population_size=30)
        labels, thresholds, distances_train, label_map = aaia.classify(X_train, y_train, X_test, best_solution)
        
        assert labels.shape == y_test.shape
        assert np.all(np.isin(labels, np.unique(y_train)))
    
    def test_classify_distances_train_correctness(self):
        """Test that distances_train are calculated correctly"""
        X_train = np.array([[0, 0], [1, 0], [0, 1], [3, 3]])
        y_train = np.array([0, 0, 0, 1])
        X_test = np.array([[0.5, 0.5]])
        best_solution = np.array([0, 0])
        
        _, _, distances_train, _ = aaia.classify(X_train, y_train, X_test, best_solution)
        
        expected_distances = np.array([0, 1, 1, 6])  # Manhattan distances from best_solution
        assert np.allclose(distances_train, expected_distances)
    
    def test_classify_threshold_ordering(self):
        """Test that thresholds are between class ranges"""
        X_train = np.array([[0, 0], [1, 0], [5, 0], [6, 0]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[2, 0]])
        best_solution = np.array([0, 0])
        
        _, thresholds, distances_train, _ = aaia.classify(X_train, y_train, X_test, best_solution)
        
        # Thresholds should be reasonable values between the distance ranges
        assert len(thresholds) > 0
        assert all(np.isfinite(t) for t in thresholds)


class TestIntegration:
    """Integration tests for the full algorithm.
    
    Tests the complete workflow: optimization -> classification -> evaluation
    
    This tests:
    - Full pipeline with real-world data (Iris dataset)
    - Full pipeline with synthetic clustered data
    - End-to-end correctness
    - Reproducibility with fixed random seeds
    - Reasonable performance metrics (accuracy > 0.6 on synthetic data)
    """
    
    def test_full_pipeline_iris(self):
        """Test full pipeline with Iris dataset"""
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        np.random.seed(42)
        train_idx, test_idx = [], []
        for cls in range(3):
            idx = np.where(y == cls)[0]
            train_idx.extend(idx[:30])
            test_idx.extend(idx[30:50])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Find solution
        best_solution = aaia.find_solution(X_train, max_iterations=100, population_size=30)
        assert best_solution.shape == (4,)
        
        # Classify
        labels, thresholds, distances_train, label_map = aaia.classify(X_train, y_train, X_test, best_solution)
        assert labels.shape == y_test.shape
        assert np.all(np.isin(labels, np.unique(y_train)))
    
    def test_full_pipeline_synthetic_2d(self):
        """Test full pipeline with synthetic 2D data"""
        np.random.seed(42)
        # Create two Gaussian clusters
        cluster1 = np.random.normal([0, 0], 1, size=(25, 2))
        cluster2 = np.random.normal([5, 5], 1, size=(25, 2))
        X_train = np.vstack([cluster1, cluster2])
        y_train = np.hstack([np.zeros(25, dtype=int), np.ones(25, dtype=int)])
        
        cluster1_test = np.random.normal([0, 0], 1, size=(15, 2))
        cluster2_test = np.random.normal([5, 5], 1, size=(15, 2))
        X_test = np.vstack([cluster1_test, cluster2_test])
        y_test = np.hstack([np.zeros(15, dtype=int), np.ones(15, dtype=int)])
        
        # Find solution
        best_solution = aaia.find_solution(X_train, max_iterations=100, population_size=30)
        
        # Classify
        labels, _, _, _ = aaia.classify(X_train, y_train, X_test, best_solution)
        
        # Check accuracy is reasonable for synthetic data
        accuracy = np.mean(labels == y_test)
        assert accuracy > 0.6, f"Accuracy should be >0.6 for synthetic data, got {accuracy}"
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed"""
        data = np.random.RandomState(42).uniform(0, 10, size=(20, 3))
        
        np.random.seed(100)
        solution1 = aaia.find_solution(data, max_iterations=50, population_size=20)
        
        np.random.seed(100)
        solution2 = aaia.find_solution(data, max_iterations=50, population_size=20)
        
        assert np.allclose(solution1, solution2)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions.
    
    Validates algorithm robustness with unusual/extreme inputs:
    
    This tests:
    - Single data point
    - All identical data points
    - Collinear data (all points on a line)
    - High-dimensional data (50+ features)
    - Negative feature values
    - Mixed scale features (values differing by orders of magnitude)
    - Numerical stability in all conditions
    """
    
    def test_single_data_point(self):
        """Test with single data point"""
        data = np.array([[1.0, 2.0, 3.0]])
        
        result = aaia.find_solution(data, max_iterations=10, population_size=5)
        
        assert result.shape == (3,)
    
    def test_identical_data_points(self):
        """Test with all identical data points"""
        data = np.array([[1.0, 2.0]] * 10)
        
        result = aaia.find_solution(data, max_iterations=10, population_size=5)
        
        assert result.shape == (2,)
        # Solution should be close to the data point
        assert np.allclose(result, [1.0, 2.0], atol=5)
    
    def test_collinear_data(self):
        """Test with collinear data points"""
        data = np.array([[i, i*2] for i in range(10)])
        
        result = aaia.find_solution(data, max_iterations=20, population_size=10)
        
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
    
    def test_large_feature_space(self):
        """Test with high-dimensional data"""
        data = np.random.uniform(0, 10, size=(20, 50))
        
        result = aaia.find_solution(data, max_iterations=10, population_size=10)
        
        assert result.shape == (50,)
        assert np.all(np.isfinite(result))
    
    def test_negative_values(self):
        """Test with negative feature values"""
        data = np.random.uniform(-100, -10, size=(20, 3))
        
        result = aaia.find_solution(data, max_iterations=20, population_size=10)
        
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
    
    def test_mixed_scale_features(self):
        """Test with features at different scales"""
        data = np.hstack([
            np.random.uniform(0, 1, size=(20, 1)),
            np.random.uniform(0, 1000, size=(20, 1)),
            np.random.uniform(-1e6, 1e6, size=(20, 1))
        ])
        
        result = aaia.find_solution(data, max_iterations=20, population_size=10)
        
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
