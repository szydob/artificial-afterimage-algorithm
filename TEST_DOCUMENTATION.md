# AAIA Test Suite Documentation

## Overview

This document provides comprehensive documentation for the test suite of the Artificial Afterimage Algorithm (AAIA) implementation. The test suite contains **41 unit, integration, and edge case tests** organized into **10 test classes**.

## Test Coverage Summary

| Test Class | Tests | Coverage Area |
|-----------|-------|--------------|
| TestVisualAngle | 4 | Visual angle computation |
| TestPerceptualSize | 3 | Perceptual size calculation |
| TestPopulationUpdate | 2 | Population update mechanism |
| TestManhattanDistance | 4 | L1 distance calculation |
| TestFitness | 5 | Fitness evaluation |
| TestBestAndWorstCalculation | 4 | Selection of best/worst candidates |
| TestFindSolution | 5 | Main optimization algorithm |
| TestClassify | 5 | Classification mechanism |
| TestIntegration | 3 | Full pipeline tests |
| TestEdgeCases | 6 | Boundary conditions & robustness |
| **TOTAL** | **41** | **Complete Algorithm** |

## Test Organization

### 1. Unit Tests (Function-Level)

#### TestVisualAngle
Tests the `calculate_visual_angle()` function that computes visual angles using the formula:
```
visual_angle = 2 * arctan(population / (2 * (population - best_solution) + epsilon))
```

**Tests:**
- `test_visual_angle_shape()`: Verifies output shape matches input
- `test_visual_angle_values_in_range()`: Confirms values are within [-π, π]
- `test_visual_angle_same_point()`: Tests behavior when population equals best solution
- `test_visual_angle_symmetry()`: Ensures numerical stability

#### TestPerceptualSize
Tests the `calculate_perceptual_size()` function.

**Formula:** `perceptual_size = visual_angle * (population - best_solution)`

**Tests:**
- `test_perceptual_size_shape()`: Output shape validation
- `test_perceptual_size_zero_when_same_point()`: Zero when distance is zero
- `test_perceptual_size_values()`: Numerical stability check

#### TestPopulationUpdate
Tests the `calculate_population()` function that updates the population.

**Tests:**
- `test_population_update_shape()`: Output shape validation
- `test_population_update_finite()`: Ensures no NaN/Inf values

#### TestManhattanDistance
Tests the `manhattan_distance()` function.

**Formula:** `distance = sum(|population - best_solution|)` for each row

**Tests:**
- `test_manhattan_distance_shape()`: 1D output shape
- `test_manhattan_distance_zero_when_same_point()`: Zero for identical points
- `test_manhattan_distance_correctness()`: Validates calculations with hand-verified examples
- `test_manhattan_distance_always_non_negative()`: Non-negativity constraint

#### TestFitness
Tests the `fitness()` function.

**Formula:** `fitness = sum(|data[i] - candidate|)` for all data points

**Tests:**
- `test_fitness_shape()`: Scalar output validation
- `test_fitness_symmetric()`: Distance calculation includes all points
- `test_fitness_correctness()`: Hand-calculated validation
- `test_fitness_always_non_negative()`: Non-negativity guarantee
- `test_fitness_commutative()`: Consistency test

#### TestBestAndWorstCalculation
Tests the `calculate_best_and_worst()` function.

**Tests:**
- `test_best_and_worst_shape()`: Output shape validation
- `test_best_and_worst_in_population()`: Best/worst are population members
- `test_best_better_than_worst()`: Fitness ordering verification
- `test_single_population_member()`: Edge case with single member

#### TestFindSolution
Tests the main optimization algorithm `find_solution()`.

**Tests:**
- `test_find_solution_shape()`: Output shape matches features
- `test_find_solution_within_data_bounds()`: Solution clipping verification
- `test_find_solution_convergence()`: More iterations improve fitness
- `test_find_solution_single_feature()`: Works with 1D data
- `test_find_solution_large_population()`: Scales to large populations

#### TestClassify
Tests the `classify()` function.

**Tests:**
- `test_classify_shape()`: Output shapes validation
- `test_classify_valid_labels()`: Labels are valid class indices
- `test_classify_with_iris_data()`: Real-world dataset test
- `test_classify_distances_train_correctness()`: Distance calculation accuracy
- `test_classify_threshold_ordering()`: Threshold validity

### 2. Integration Tests

#### TestIntegration
Tests the complete algorithm pipeline.

**Tests:**
- `test_full_pipeline_iris()`: End-to-end with Iris dataset (150 samples, 4 features)
- `test_full_pipeline_synthetic_2d()`: End-to-end with synthetic Gaussian clusters
- `test_reproducibility_with_seed()`: Reproducibility with fixed random seeds

**Performance Expectation:** Accuracy > 60% on synthetic data

### 3. Edge Cases & Robustness

#### TestEdgeCases
Tests algorithm robustness with unusual/extreme inputs.

**Tests:**
- `test_single_data_point()`: Single sample input
- `test_identical_data_points()`: All points are identical
- `test_collinear_data()`: All points on a line
- `test_large_feature_space()`: 50-dimensional data
- `test_negative_values()`: Negative feature values
- `test_mixed_scale_features()`: Features at different scales (1 to 1e6)

## Running the Tests

### Run All Tests
```bash
pytest test_aaia.py -v
```

### Run Specific Test Class
```bash
pytest test_aaia.py::TestFitness -v
```

### Run Single Test
```bash
pytest test_aaia.py::TestFitness::test_fitness_correctness -v
```

### Run with Coverage Report
```bash
pytest test_aaia.py --cov=aaia --cov-report=html
```

### Run with Verbose Output
```bash
pytest test_aaia.py -vv
```

### Run with Output Capture Disabled
```bash
pytest test_aaia.py -v -s
```

### Quick Run (Minimal Output)
```bash
pytest test_aaia.py -q
```

## Generating Test Reports

### HTML Test Report
Generate a detailed HTML test report:
```bash
pytest test_aaia.py -v --html=test_report.html --self-contained-html
```
Output: `test_report.html` (opens in browser for visual test results)

### Code Coverage Report (HTML)
Generate an HTML coverage report:
```bash
pytest test_aaia.py --cov=aaia --cov-report=html
```
Output: `htmlcov/index.html` (opens in browser to view coverage breakdown)

### Combined Report (Tests + Coverage)
```bash
pytest test_aaia.py -v --html=test_report.html --self-contained-html --cov=aaia --cov-report=html
```
This generates both test results and coverage reports.

### JUnit XML Report (for CI/CD)
```bash
pytest test_aaia.py -v --junit-xml=junit.xml
```
Output: `junit.xml` (useful for CI/CD pipelines like Jenkins, GitLab CI, etc.)

### Generate All Reports at Once
```bash
pytest test_aaia.py -v --html=test_report.html --self-contained-html --cov=aaia --cov-report=html --junit-xml=junit.xml
```

### Install Required Packages (if needed)
```bash
pip install pytest-html pytest-cov
```

## Report Files Generated

| Report | Command | Output File | Purpose |
|--------|---------|------------|---------|
| HTML Test Report | `pytest ... --html=test_report.html` | `test_report.html` | Visual test results with pass/fail details |
| Code Coverage | `pytest ... --cov=aaia --cov-report=html` | `htmlcov/index.html` | Coverage percentage by file and function |
| JUnit XML | `pytest ... --junit-xml=junit.xml` | `junit.xml` | Machine-readable format for CI/CD |
| Terminal Output | `pytest ... -v` | Console | Quick overview on command line |

## GitIgnore Entry

Add these to `.gitignore` to prevent uploading report files:
```
# Test reports and coverage
test_report.html
htmlcov/
junit.xml
.coverage
```

## Viewing Reports

### HTML Test Report
```bash
# Windows
start test_report.html

# macOS
open test_report.html

# Linux
xdg-open test_report.html
```

### Coverage Report
```bash
# Windows
start htmlcov/index.html

# macOS
open htmlcov/index.html

# Linux
xdg-open htmlcov/index.html
```

## Test Design Principles

### 1. **Function Isolation**
Each unit test focuses on a single function, with controlled inputs and expected outputs.

### 2. **Numerical Correctness**
Critical calculations are validated against hand-calculated values or mathematical formulas.

### 3. **Numerical Stability**
All tests verify absence of NaN and Inf values, especially in floating-point operations.

### 4. **Boundary Conditions**
Edge cases (single points, identical points, collinear data) are explicitly tested.

### 5. **Shape Consistency**
Output shapes are verified to match expected dimensions.

### 6. **Non-negativity Constraints**
Distance and fitness functions are verified to always return non-negative values.

### 7. **Real-World Validation**
Tests use real datasets (Iris) and synthetic data with known properties.

## Algorithm Component Relationships

```
├── calculate_visual_angle()
│   └── Uses: population, best_solution
│       Output: shapes for perceptual calculation
│
├── calculate_perceptual_size()
│   └── Uses: visual_angle, population, best_solution
│       Output: perceptual sizes for population update
│
├── calculate_population()
│   └── Uses: perceptual_size, best/worst solutions
│       Output: updated population
│
├── manhattan_distance()
│   └── Uses: population, best_solution
│       Output: individual distances
│
├── fitness()
│   └── Uses: candidate, data
│       Output: scalar fitness value
│
├── calculate_best_and_worst()
│   └── Uses: population, fitness function
│       Output: best and worst candidates
│
├── find_solution()
│   └── Orchestrates: visual_angle, perceptual_size,
│       population_update, best/worst calculation
│       Output: optimized solution
│
└── classify()
    └── Uses: distances, training labels
        Output: predicted labels for test data
```

## Expected Test Results

All 41 tests should pass:
```
============================= 41 passed in ~2.5s =============================
```

## Interpreting Test Failures

### Visual Angle Tests Fail
- Check numerical stability in `calculate_visual_angle()`
- Verify epsilon value prevents division by zero

### Population Update Tests Fail
- Check population clipping logic
- Verify random element doesn't cause instability

### Fitness Tests Fail
- Validate Manhattan distance calculation
- Ensure sum includes all data points

### Classification Tests Fail
- Check threshold calculation
- Verify label mapping correctness

### Integration Tests Fail
- Run individual function tests to isolate issue
- Verify convergence is improving (more iterations = lower fitness)

## Test Data Characteristics

### Iris Dataset (Integration Tests)
- **Samples:** 150 (90 train, 60 test)
- **Features:** 4 (sepal length, width, petal length, width)
- **Classes:** 3 (setosa, versicolor, virginica)
- **Expected Accuracy:** > 70% typical

### Synthetic 2D Data (Integration Tests)
- **Samples:** 80 total (50 train, 30 test)
- **Features:** 2 (x, y coordinates)
- **Classes:** 2 (clusters at [0,0] and [5,5])
- **Expected Accuracy:** > 60% (minimum)

### Edge Case Data
- **Dimensions:** 1 to 50 features
- **Value Ranges:** -1e6 to 1e6
- **Samples:** 1 to 30

## Performance Metrics

| Configuration | Iterations | Population | Time (typical) |
|:--------------|:-----------|:-----------|:--------------|
| Small | 10 | 10 | <0.1s |
| Medium | 50 | 20 | 0.1-0.5s |
| Large | 100 | 30 | 0.5-2s |

## Common Test Patterns

### Shape Verification
```python
result = aaia.function(inputs)
assert result.shape == (expected_rows, expected_cols)
```

### Numerical Correctness
```python
result = aaia.function(inputs)
expected = hand_calculated_value
assert np.allclose(result, expected)
```

### Non-negativity
```python
result = aaia.fitness(candidate, data)
assert result >= 0
```

### Finiteness
```python
result = aaia.function(inputs)
assert np.all(np.isfinite(result))
```

## Maintenance Guidelines

### Adding New Tests
1. Identify the function to test
2. Choose appropriate test class (or create new one)
3. Follow naming convention: `test_<function>_<aspect>`
4. Use docstring to explain test purpose
5. Include arrange-act-assert pattern

### Updating Tests
- Maintain independence between tests
- Use fixtures (pytest fixtures) for common setup
- Don't modify test data during execution
- Preserve backward compatibility

### Debugging Tests
1. Run individual test with `-vv` flag
2. Use `-s` to see print statements
3. Check edge cases in isolation
4. Verify test data is correct

## References

- **pytest Documentation:** https://docs.pytest.org/
- **NumPy Testing:** https://numpy.org/doc/stable/reference/testing.html
- **Scikit-learn Datasets:** https://scikit-learn.org/stable/datasets/

## Quick Command Reference

### Install Dependencies
```bash
pip install pytest pytest-html pytest-cov
```

### Essential Commands
```bash
# Run all tests (basic)
pytest test_aaia.py -v

# Run with coverage
pytest test_aaia.py -v --cov=aaia --cov-report=html

# Generate test report
pytest test_aaia.py -v --html=test_report.html --self-contained-html

# All reports in one command
pytest test_aaia.py -v --html=test_report.html --self-contained-html --cov=aaia --cov-report=html --junit-xml=junit.xml

# Run specific test
pytest test_aaia.py::TestFitness::test_fitness_correctness -v
```

### Output Files (to be excluded from repo)
- `test_report.html` - HTML test results
- `htmlcov/` - Coverage report directory
- `junit.xml` - XML test results
- `.coverage` - Coverage data file

These files are listed in `.gitignore` and should not be committed to the repository.

## Test Results History

| Date | Total Tests | Passed | Failed | Success Rate |
|------|------------|--------|--------|--------------|
| 2026-03-09 | 41 | 41 | 0 | 100% |
