#!/usr/bin/env python3
"""
Location optimization script using AAIA algorithm with optional clustering.

Goals:
- cluster existing firms geographically,
- identify clusters with low representation of a given business category (white spots),
- recommend an optimal location within the chosen cluster or entire region for a new business.

Usage examples:
  python optimize_location.py --business-type gastronomia          # optimize across all data
  python optimize_location.py --business-type handel --clusters 5  # cluster data and target sparse cluster
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# clustering
# AAIA-based clustering will be used instead of k-means

# Add path to aaia module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import aaia

def calculate_business_profit(candidate, business_type='gastronomia'):
    """
    Calculate estimated daily profit for business at given location.
    
    Model:
    Profit = Daily Revenue - Daily Costs
    
    Where:
    - Daily Revenue = Customer Flow × Average Spending × (1 - Competition Impact)
    - Daily Costs = Fixed Costs (rent, utilities) + Variable Costs
    """
    lat, lon, pop_density, comp_1km, comp_3km, access, rental, income = candidate

    # Normalize inputs (they are already normalized from StandardScaler)
    # Scale them back to realistic ranges for calculations
    pop_density_scaled = 100 + pop_density * 500  # 100-600 people/km²
    competitors_scaled = max(1, comp_1km * 20)  # Scale up to actual numbers
    accessibility_scaled = max(0.1, access)  # 0.1-1.0 score
    rental_scaled = max(300, rental * 50)  # 300-3000 PLN/month
    income_scaled = 40000 + income * 10000  # 40k-60k PLN/month

    # Business type parameters (monthly averages)
    business_params = {
        'gastronomia': {
            'avg_ticket': 50,  # PLN per customer
            'covers_per_day': 80,  # estimated seats/tables
            'occupancy_rate': 0.7,  # typical occupancy
            'daily_fixed_costs': 800,  # rent, utilities (rough daily estimate)
            'daily_variable_costs_per_customer': 15,  # food, labor per customer
            'foot_traffic_multiplier': 1.2  # restaurants depend on foot traffic
        },
        'handel': {
            'avg_ticket': 200,
            'covers_per_day': 100,  # transactions
            'occupancy_rate': 0.5,
            'daily_fixed_costs': 500,
            'daily_variable_costs_per_customer': 80,  # cost of goods
            'foot_traffic_multiplier': 0.8
        },
        'zdrowie': {
            'avg_ticket': 150,
            'covers_per_day': 40,  # patients
            'occupancy_rate': 0.8,
            'daily_fixed_costs': 600,
            'daily_variable_costs_per_customer': 40,  # supplies
            'foot_traffic_multiplier': 0.5  # less dependent on foot traffic
        }
    }

    params = business_params.get(business_type, business_params['gastronomia'])

    # 1. REVENUE CALCULATION
    # Base foot traffic depends on: population density, accessibility, income level
    base_foot_traffic = 100 * (pop_density_scaled / 300) * accessibility_scaled
    
    # Income level affects willingness to spend
    income_factor = income_scaled / 50000  # Relative to ~50k average
    
    # Revenue with competition impact
    # More competitors reduce revenue (market saturation)
    competition_penalty = 1.0 / (1.0 + 0.05 * competitors_scaled)  # Saturation model
    
    daily_customers = (
        base_foot_traffic * 
        params['foot_traffic_multiplier'] * 
        income_factor * 
        competition_penalty
    )
    
    daily_revenue = daily_customers * params['avg_ticket']

    # 2. COST CALCULATION
    # Fixed costs scale with rental prices (normalization)
    rental_factor = rental_scaled / 1500  # 1500 is assumed average
    daily_fixed_costs = params['daily_fixed_costs'] * rental_factor
    
    # Variable costs
    daily_variable_costs = daily_customers * params['daily_variable_costs_per_customer']
    
    total_daily_costs = daily_fixed_costs + daily_variable_costs

    # 3. PROFIT
    daily_profit = daily_revenue - total_daily_costs
    
    # Monthly profit (rough estimate, 25 working days)
    monthly_profit = daily_profit * 25
    
    # Yearly profit
    yearly_profit = monthly_profit * 12

    return yearly_profit

def objective_function(candidate, data, business_type='gastronomia'):
    """
    Objective function for location optimization based on business profitability.
    Maximizes yearly profit, penalizes unprofitable locations.
    """
    profit = calculate_business_profit(candidate, business_type)
    
    # Return negative profit for minimization (AAIA minimizes objective)
    # Add penalty for very unprofitable locations
    if profit < 0:
        penalty = abs(profit) * 2  # Double penalty for losses
    else:
        penalty = 0
    
    return -(profit - penalty)

def calculate_best_and_worst_location(population, data, objective_func, business_type='gastronomia'):
    scores = np.array([objective_func(candidate, data, business_type) for candidate in population])
    best = population[np.argmin(scores)]  # Minimize objective
    worst = population[np.argmax(scores)]
    return best, worst

def find_optimal_location(data, objective_func, business_type='gastronomia', max_iterations=1000, population_size=50):
    """
    Find optimal location using AAIA-inspired algorithm with custom objective function.
    """
    count = 0
    low = data.min(axis=0)
    high = data.max(axis=0)
    population = np.random.uniform(low, high, size=(population_size, data.shape[1]))
    best_solution, worst_solution = calculate_best_and_worst_location(population, data, objective_func, business_type)

    fitness_history = []

    while count < max_iterations:
        V = aaia.calculate_visual_angle(population, best_solution)
        S = aaia.calculate_perceptual_size(V, population, best_solution)

        local_best, local_worst = calculate_best_and_worst_location(population, data, objective_func, business_type)

        if objective_func(local_best, data, business_type) < objective_func(best_solution, data, business_type):
            best_solution = local_best
        if objective_func(local_worst, data, business_type) > objective_func(worst_solution, data, business_type):
            worst_solution = local_worst

        population = aaia.calculate_population(S, best_solution, worst_solution, population)
        population = np.clip(population, low, high)

        fitness_history.append(objective_func(best_solution, data, business_type))
        count += 1

        if count % 100 == 0:
            profit = -objective_func(best_solution, data, business_type)
            print(f"Iteration {count}: Estimated yearly profit = {profit:,.0f} PLN")

    return best_solution, fitness_history

def plot_optimization_results(firms_df, optimal_location, fitness_history, business_type, cluster_label=None):
    """Create visualization of optimization results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Filter firms by type if specified
    if business_type != 'all':
        plot_df = firms_df[firms_df['business_category'] == business_type]
        title_suffix = f" - {business_type}"
    else:
        plot_df = firms_df
        title_suffix = ""

    if cluster_label is not None and 'cluster' in firms_df.columns:
        # color points by cluster if available
        plot_df = plot_df[plot_df['cluster'] == cluster_label]
        title_suffix += f" (cluster {cluster_label})"

    # 1. Firms and optimal location
    ax = axes[0, 0]
    if 'cluster' in plot_df.columns:
        scatter = ax.scatter(plot_df['lon'], plot_df['lat'],
                            c=plot_df['cluster'],
                            cmap='tab20', alpha=0.6, s=30)
    else:
        scatter = ax.scatter(plot_df['lon'], plot_df['lat'],
                            c=plot_df['competitors_1km'],
                            cmap='Reds', alpha=0.6, s=30)
    ax.scatter(optimal_location[1], optimal_location[0],
              c='blue', marker='*', s=300, label='Optimal Location', edgecolors='black')
    ax.set_title(f'Firm Locations and Optimal New Location{title_suffix}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Competitors within 1km')

    # 2. Population density heatmap
    ax = axes[0, 1]
    scatter = ax.scatter(plot_df['lon'], plot_df['lat'],
                        c=plot_df['population_density'],
                        cmap='viridis', alpha=0.6, s=30)
    ax.scatter(optimal_location[1], optimal_location[0],
              c='red', marker='*', s=300, label='Optimal')
    ax.set_title('Population Density')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax, label='Population Density')

    # 3. Accessibility score
    ax = axes[1, 0]
    scatter = ax.scatter(plot_df['lon'], plot_df['lat'],
                        c=plot_df['accessibility_score'],
                        cmap='Greens', alpha=0.6, s=30)
    ax.scatter(optimal_location[1], optimal_location[0],
              c='red', marker='*', s=300, label='Optimal')
    ax.set_title('Accessibility Score')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax, label='Accessibility Score')

    # 4. Fitness evolution
    ax = axes[1, 1]
    ax.plot(fitness_history)
    ax.set_title('Objective Function Evolution')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value (minimized)')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'optimization_results_{business_type}.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid blocking in headless environment

def aaia_clustering(df, n_clusters=5, max_iterations=500, population_size=50):
    """Cluster data geographically using AAIA find_solution repeatedly.

    Each iteration finds a center that minimizes sum of Manhattan distances to
    all points (using the aaia.find_solution helper).  The algorithm then
    assigns every point to the closest center found so far. This mimics a
    k-means style partitioning while relying on AAIA's global search.
    """
    # only geographic coordinates are used here; extend if needed
    data = df[['lat', 'lon']].values
    n_points = data.shape[0]

    centers = []
    best_dists = np.full(n_points, np.inf)
    assignments = np.zeros(n_points, dtype=int)

    for k in range(n_clusters):
        center = aaia.find_solution(data, max_iterations=max_iterations, population_size=population_size)
        centers.append(center)
        dists = np.sum(np.abs(data - center), axis=1)
        better = dists < best_dists
        assignments[better] = k
        best_dists[better] = dists[better]

    df['cluster'] = assignments
    return df, np.vstack(centers)


def summarize_clusters(df, target_type):
    """Return counts per cluster and identify cluster with fewest target-type firms."""
    summary = df.pivot_table(index='cluster', columns='business_category',
                              aggfunc='size', fill_value=0)
    summary['total'] = summary.sum(axis=1)
    summary['target_count'] = summary.get(target_type, 0)
    # compute proportion
    summary['target_prop'] = summary['target_count'] / summary['total']
    # cluster with smallest proportion of target_type
    best_cluster = summary['target_prop'].idxmin()
    return summary, best_cluster


def main():
    parser = argparse.ArgumentParser(description='Optimize business location using AAIA')
    parser.add_argument('--business-type', default='gastronomia',
                       choices=['gastronomia', 'handel', 'zdrowie', 'all'],
                       help='Type of business to optimize location for')
    parser.add_argument('--clusters', type=int, default=5,
                       help='Number of geographic clusters to form using AAIA (0 to skip)')
    parser.add_argument('--max-iterations', type=int, default=500,
                       help='Maximum number of optimization iterations')
    parser.add_argument('--population-size', type=int, default=30,
                       help='Population size for AAIA')
    parser.add_argument('--weights', nargs=6, type=float,
                       default=[1.0, 0.5, 0.3, 0.8, 0.4, 0.2],
                       help='Weights for: pop_density, comp_1km, comp_3km, access, rental, income')

    args = parser.parse_args()

    # Load optimization dataset
    try:
        df = pd.read_csv('firmy_optimization.csv')
        print(f"Loaded optimization dataset with {len(df)} firms")
    except FileNotFoundError:
        print("Error: firmy_optimization.csv not found. Run build_database.py first.")
        sys.exit(1)

    # perform clustering if requested
    cluster_label = None
    df_cluster = None
    if args.clusters and args.clusters > 0:
        print(f"Clustering firms into {args.clusters} clusters using AAIA...")
        df, centroids = aaia_clustering(df, n_clusters=args.clusters,
                                       max_iterations=args.max_iterations,
                                       population_size=args.population_size)
        summary, best_cluster = summarize_clusters(df, args.business_type)
        print("Cluster summary (counts and proportions):")
        print(summary)
        print(f"Cluster with lowest proportion of '{args.business_type}': {best_cluster}")
        cluster_label = best_cluster
        df_cluster = df[df['cluster'] == cluster_label].copy()
        print(f"Selected cluster {cluster_label} with {len(df_cluster)} total firms")
    else:
        df_cluster = df.copy()

    # Filter for plotting (we don't remove non-target firms from optimization data)
    if args.business_type != 'all':
        plot_df = df_cluster[df_cluster['business_category'] == args.business_type]
        print(f"Plotting {len(plot_df)} firms of type {args.business_type} in chosen dataset")
        if len(plot_df) == 0:
            print(f"No firms found for type: {args.business_type} within chosen data")
        # df remains df_cluster for optimization
    else:
        plot_df = df_cluster

    # Use df_cluster (all firms in cluster or full data) for optimization
    df = df_cluster

    # Prepare data (exclude business_category column)
    features = ['lat', 'lon', 'population_density', 'competitors_1km',
               'competitors_3km', 'accessibility_score', 'rental_cost', 'income_level']
    X = df[features].values

    # Set weights
    weights = {
        'population_density': args.weights[0],
        'competitors_1km': args.weights[1],
        'competitors_3km': args.weights[2],
        'accessibility_score': args.weights[3],
        'rental_cost': args.weights[4],
        'income_level': args.weights[5]
    }

    print(f"Optimization for business type: {args.business_type}")

    # Create objective function with business type
    def obj_func(candidate, data, business_type=args.business_type):
        return objective_function(candidate, data, business_type)

    # Run optimization
    print(f"Starting optimization with {args.max_iterations} iterations and population size {args.population_size}...")
    np.random.seed(42)
    optimal_location, fitness_history = find_optimal_location(
        X, obj_func,
        business_type=args.business_type,
        max_iterations=args.max_iterations,
        population_size=args.population_size
    )

    print("\nOptimization complete!")
    estimated_yearly_profit = -obj_func(optimal_location, X)
    print(f"\n💰 ESTIMATED YEARLY PROFIT: {estimated_yearly_profit:,.0f} PLN")
    print(f"Optimal location: Lat={optimal_location[0]:.4f}, Lon={optimal_location[1]:.4f}")
    print(f"Population density: {optimal_location[2]:.4f}")
    print(f"Competitors 1km: {optimal_location[3]:.4f}")
    print(f"Competitors 3km: {optimal_location[4]:.4f}")
    print(f"Accessibility score: {optimal_location[5]:.4f}")
    print(f"Rental cost: {optimal_location[6]:.4f}")
    print(f"Income level: {optimal_location[7]:.4f}")

    # Create plots using filtered/clustered data
    print("\nGenerating visualization...")
    plot_optimization_results(plot_df if 'plot_df' in locals() else df,
                              optimal_location,
                              fitness_history,
                              args.business_type,
                              cluster_label=cluster_label)

    # Save results
    results = {
        'business_type': args.business_type,
        'estimated_yearly_profit_pln': float(estimated_yearly_profit),
        'optimal_location': {
            'lat': float(optimal_location[0]),
            'lon': float(optimal_location[1]),
            'population_density': float(optimal_location[2]),
            'competitors_1km': float(optimal_location[3]),
            'competitors_3km': float(optimal_location[4]),
            'accessibility_score': float(optimal_location[5]),
            'rental_cost': float(optimal_location[6]),
            'income_level': float(optimal_location[7])
        },
        'cluster_used': int(cluster_label) if cluster_label is not None else None,
        'final_objective_value': float(obj_func(optimal_location, X)),
        'iterations': args.max_iterations,
        'population_size': args.population_size,
        'business_model': {
            'revenue_model': 'Based on foot traffic × average spending × (1 - competition impact)',
            'cost_model': 'Fixed costs (rent) + variable costs (labor, supplies)',
            'profit': 'Revenue - Costs',
            'assumptions': 'Monthly estimates, 25 working days, 12 months per year'
        }
    }

    import json
    with open(f'optimal_location_{args.business_type}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to optimal_location_{args.business_type}.json")
    print(f"Visualization saved to optimization_results_{args.business_type}.png")

if __name__ == "__main__":
    main()