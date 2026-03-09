#!/usr/bin/env python3
"""
Visualization script for firm distribution on map by business type
Usage: python visualize_map.py [--business-type TYPE]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os

def load_data():
    """Load firm data from CSV"""
    try:
        df = pd.read_csv('firmy_unified.csv')
        return df
    except FileNotFoundError:
        print("Error: firmy_unified.csv not found. Run build_database.py first.")
        return None

def create_business_type_map(df, figsize=(16, 12)):
    """Create map showing firm distribution by business type"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Define colors for business types
    colors = {
        'gastronomia': '#FF6B6B',      # Red
        'handel': '#4ECDC4',            # Teal
        'zdrowie': '#45B7D1',           # Blue
        'usługi finansowe': '#FFA07A',  # Light salmon
        'inne': '#95E1D3'               # Mint
    }

    # 1. All businesses by type
    ax = axes[0, 0]
    for business_type in df['business_category'].unique():
        subset = df[df['business_category'] == business_type]
        color = colors.get(business_type, '#CCCCCC')
        ax.scatter(subset['lon'], subset['lat'], 
                  c=color, label=business_type, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_title('Rozkład firm na mapie Krakowa wg typu biznesu', fontsize=14, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 2. Only restaurants
    ax = axes[0, 1]
    restaurants = df[df['business_category'] == 'gastronomia']
    if len(restaurants) > 0:
        scatter = ax.scatter(restaurants['lon'], restaurants['lat'],
                           c=restaurants['competitors_1km'],
                           cmap='YlOrRd', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Konkurenci w 1km')
        ax.set_title(f'Restauracje i kawiarnie ({len(restaurants)})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.grid(True, alpha=0.3)

    # 3. Only shops
    ax = axes[1, 0]
    shops = df[df['business_category'] == 'handel']
    if len(shops) > 0:
        scatter = ax.scatter(shops['lon'], shops['lat'],
                           c=shops['competitors_1km'],
                           cmap='BuGn', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Konkurenci w 1km')
        ax.set_title(f'Sklepy i handel ({len(shops)})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.grid(True, alpha=0.3)

    # 4. Only healthcare
    ax = axes[1, 1]
    healthcare = df[df['business_category'] == 'zdrowie']
    if len(healthcare) > 0:
        scatter = ax.scatter(healthcare['lon'], healthcare['lat'],
                           c=healthcare['competitors_1km'],
                           cmap='Blues', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Konkurenci w 1km')
        ax.set_title(f'Usługi zdrowotne ({len(healthcare)})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mapa_firmy_typy.png', dpi=300, bbox_inches='tight')
    print("Saved: mapa_firmy_typy.png")

def create_single_map_all_types(df, figsize=(14, 10)):
    """Create single map showing all three main business types"""
    fig, ax = plt.subplots(figsize=figsize)

    # Define marker styles for each business type
    markers = {
        'gastronomia': 'o',    # Circle
        'handel': 's',         # Square
        'zdrowie': '^'         # Triangle
    }
    
    colors = {
        'gastronomia': '#FF6B6B',      # Red
        'handel': '#4ECDC4',            # Teal
        'zdrowie': '#45B7D1'            # Blue
    }

    # Plot each business type
    for business_type in ['gastronomia', 'handel', 'zdrowie']:
        subset = df[df['business_category'] == business_type]
        if len(subset) > 0:
            ax.scatter(subset['lon'], subset['lat'],
                      marker=markers.get(business_type, 'o'),
                      c=colors.get(business_type, '#CCCCCC'),
                      s=120, alpha=0.6, edgecolors='black', linewidth=0.5,
                      label=f'{business_type} ({len(subset)})')

    ax.set_title('Rozkład firm na mapie Krakowa\n(Gastronomia = ○, Handel = □, Zdrowie = △)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Długość geograficzna', fontsize=12)
    ax.set_ylabel('Szerokość geograficzna', fontsize=12)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mapa_wszystkie_typy.png', dpi=300, bbox_inches='tight')
    print("Saved: mapa_wszystkie_typy.png")

def create_density_heatmap(df, figsize=(14, 10)):
    """Create heatmap showing firm density"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    business_types = ['gastronomia', 'handel', 'zdrowie']
    colormaps = ['Reds', 'Blues', 'Greens']

    for idx, (business_type, cmap) in enumerate(zip(business_types, colormaps)):
        ax = axes[idx // 2, idx % 2]
        
        subset = df[df['business_category'] == business_type]
        if len(subset) > 0:
            # Create 2D histogram (density)
            h = ax.hist2d(subset['lon'], subset['lat'], bins=20, cmap=cmap, cmin=1)
            plt.colorbar(h[3], ax=ax, label='Liczba firm')
            ax.set_title(f'Gęstość: {business_type}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Długość geograficzna')
        ax.set_ylabel('Szerokość geograficzna')

    # Last subplot - all businesses
    ax = axes[1, 1]
    h = ax.hist2d(df['lon'], df['lat'], bins=20, cmap='hot', cmin=1)
    plt.colorbar(h[3], ax=ax, label='Liczba firm')
    ax.set_title('Gęstość: Wszystkie firmy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')

    plt.tight_layout()
    plt.savefig('mapa_gestość.png', dpi=300, bbox_inches='tight')
    print("Saved: mapa_gestość.png")

def create_competitor_analysis(df, figsize=(14, 10)):
    """Create analysis of competition intensity"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Competition intensity for gastronomia
    ax = axes[0, 0]
    restaurants = df[df['business_category'] == 'gastronomia']
    if len(restaurants) > 0:
        scatter = ax.scatter(restaurants['lon'], restaurants['lat'],
                           c=restaurants['competitors_1km'],
                           s=150, cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Konkurenci w 1km')
        ax.set_title('Konkurencja: Gastronomia', fontsize=12, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.grid(True, alpha=0.3)

    # 2. Population density vs competitors
    ax = axes[0, 1]
    scatter = ax.scatter(df['population_density'], df['competitors_1km'],
                        c=df['accessibility_score'], cmap='viridis',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label='Dostępność')
    ax.set_xlabel('Gęstość zaludnienia')
    ax.set_ylabel('Konkurenci w 1km')
    ax.set_title('Zależność: Gęstość zaludnienia vs Konkurencja', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Accessibility vs Rental cost
    ax = axes[1, 0]
    scatter = ax.scatter(df['accessibility_score'], df['rental_cost'],
                        c=df['business_category'].map({'gastronomia': 0, 'handel': 1, 'zdrowie': 2, 'inne': 3}).fillna(3),
                        cmap='Set1', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Dostępność')
    ax.set_ylabel('Koszt wynajmu')
    ax.set_title('Zależność: Dostępność vs Koszt wynajmu', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Income level distribution
    ax = axes[1, 1]
    for business_type in df['business_category'].unique():
        subset = df[df['business_category'] == business_type]
        ax.scatter(subset['lon'], subset['lat'],
                  s=subset['income_level'] * 5,  # Size proportional to income
                  alpha=0.6, edgecolors='black', linewidth=0.5, label=business_type)
    
    ax.set_title('Poziom dochodów (rozmiar punktu)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mapa_konkurencja.png', dpi=300, bbox_inches='tight')
    print("Saved: mapa_konkurencja.png")

def create_optimal_location_overlay(df, business_type='gastronomia', figsize=(14, 10)):
    """Create map with optimal location overlay"""
    # Try to load optimal location results
    result_file = f'optimal_location_{business_type}.json'
    
    if not os.path.exists(result_file):
        print(f"Info: {result_file} not found. Skipping optimal location overlay.")
        return

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    optimal_lat = results['optimal_location']['lat']
    optimal_lon = results['optimal_location']['lon']
    profit = results['estimated_yearly_profit_pln']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot all firms
    for btype in df['business_category'].unique():
        subset = df[df['business_category'] == btype]
        if btype == business_type:
            ax.scatter(subset['lon'], subset['lat'], c='red', s=80, alpha=0.5,
                      label=f'{business_type} (istniejące)', edgecolors='darkred', linewidth=0.5)
        else:
            ax.scatter(subset['lon'], subset['lat'], c='lightgray', s=30, alpha=0.3,
                      label=f'{btype} (pozostałe)', edgecolors='gray', linewidth=0.3)

    # Plot optimal location
    ax.scatter(optimal_lon, optimal_lat, c='green', s=500, marker='*',
              edgecolors='darkgreen', linewidth=2, label='Optymalna lokalizacja', zorder=5)

    # Add circle showing 1km radius
    from matplotlib.patches import Circle
    circle = Circle((optimal_lon, optimal_lat), 0.009, fill=False, 
                    edgecolor='green', linestyle='--', linewidth=2, label='1km radius (approx)')
    ax.add_patch(circle)

    ax.set_title(f'Optymalna lokalizacja dla {business_type}\n' +
                f'Szacunkowy zysk roczny: {profit:,.0f} PLN',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Długość geograficzna')
    ax.set_ylabel('Szerokość geograficzna')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'mapa_optymalna_{business_type}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: mapa_optymalna_{business_type}.png")

def create_summary_statistics(df):
    """Create summary statistics visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Business type distribution (pie chart)
    ax = fig.add_subplot(gs[0, 0])
    type_counts = df['business_category'].value_counts()
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#95E1D3']
    ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', colors=colors_pie[:len(type_counts)])
    ax.set_title('Udział typu biznesu (%)')

    # 2. Average competitors by type
    ax = fig.add_subplot(gs[0, 1])
    avg_comp = df.groupby('business_category')['competitors_1km'].mean()
    ax.bar(range(len(avg_comp)), avg_comp.values, color=colors_pie[:len(avg_comp)])
    ax.set_xticks(range(len(avg_comp)))
    ax.set_xticklabels(avg_comp.index, rotation=45, ha='right')
    ax.set_ylabel('Średnia liczba konkurentów')
    ax.set_title('Konkurencja wg typu')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Average accessibility
    ax = fig.add_subplot(gs[0, 2])
    avg_access = df.groupby('business_category')['accessibility_score'].mean()
    ax.bar(range(len(avg_access)), avg_access.values, color=colors_pie[:len(avg_access)])
    ax.set_xticks(range(len(avg_access)))
    ax.set_xticklabels(avg_access.index, rotation=45, ha='right')
    ax.set_ylabel('Średnia dostępność')
    ax.set_title('Dostępność wg typu')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Average rental cost
    ax = fig.add_subplot(gs[1, 0])
    avg_rental = df.groupby('business_category')['rental_cost'].mean()
    ax.bar(range(len(avg_rental)), avg_rental.values, color=colors_pie[:len(avg_rental)])
    ax.set_xticks(range(len(avg_rental)))
    ax.set_xticklabels(avg_rental.index, rotation=45, ha='right')
    ax.set_ylabel('Średni koszt wynajmu')
    ax.set_title('Koszt wynajmu wg typu')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Population density distribution
    ax = fig.add_subplot(gs[1, 1])
    for btype in df['business_category'].unique():
        subset = df[df['business_category'] == btype]
        ax.hist(subset['population_density'], bins=15, alpha=0.5, label=btype)
    ax.set_xlabel('Gęstość zaludnienia')
    ax.set_ylabel('Liczba firm')
    ax.set_title('Rozkład gęstości zaludnienia')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Income level distribution
    ax = fig.add_subplot(gs[1, 2])
    for btype in df['business_category'].unique():
        subset = df[df['business_category'] == btype]
        ax.hist(subset['income_level'], bins=15, alpha=0.5, label=btype)
    ax.set_xlabel('Poziom dochodów')
    ax.set_ylabel('Liczba firm')
    ax.set_title('Rozkład poziomu dochodów')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 7. Competitors 1km distribution
    ax = fig.add_subplot(gs[2, 0])
    for btype in df['business_category'].unique():
        subset = df[df['business_category'] == btype]
        ax.hist(subset['competitors_1km'], bins=15, alpha=0.5, label=btype)
    ax.set_xlabel('Liczba konkurentów w 1km')
    ax.set_ylabel('Liczba lokalizacji')
    ax.set_title('Rozkład konkurencji w 1km')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 8. Accessibility score distribution
    ax = fig.add_subplot(gs[2, 1])
    for btype in df['business_category'].unique():
        subset = df[df['business_category'] == btype]
        ax.hist(subset['accessibility_score'], bins=15, alpha=0.5, label=btype)
    ax.set_xlabel('Dostępność')
    ax.set_ylabel('Liczba firm')
    ax.set_title('Rozkład dostępności')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 9. Rental cost distribution
    ax = fig.add_subplot(gs[2, 2])
    for btype in df['business_category'].unique():
        subset = df[df['business_category'] == btype]
        ax.hist(subset['rental_cost'], bins=15, alpha=0.5, label=btype)
    ax.set_xlabel('Koszt wynajmu')
    ax.set_ylabel('Liczba firm')
    ax.set_title('Rozkład kosztów wynajmu')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Statystyka firm wg typu biznesu', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('statystyka_firmy.png', dpi=300, bbox_inches='tight')
    print("Saved: statystyka_firmy.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize firm distribution on map')
    parser.add_argument('--business-type', default='gastronomia',
                       help='Business type for optimal location overlay')
    parser.add_argument('--all-visualizations', action='store_true',
                       help='Create all visualization types')

    args = parser.parse_args()

    # Load data
    df = load_data()
    if df is None:
        return

    print(f"Loaded {len(df)} firms")
    print(f"Business types: {df['business_category'].unique()}")

    # Create visualizations
    print("\nCreating visualizations...")
    
    create_single_map_all_types(df)
    create_business_type_map(df)
    create_density_heatmap(df)
    create_competitor_analysis(df)
    create_summary_statistics(df)
    
    # Create optimal location overlay if available
    create_optimal_location_overlay(df, args.business_type)

    print("\nAll visualizations completed!")
    print("Generated files:")
    print("- mapa_wszystkie_typy.png (mapa na jednym wykresie: gastronomia, handel, zdrowie)")
    print("- mapa_firmy_typy.png (4 mapy osobne)")
    print("- mapa_gestość.png")
    print("- mapa_konkurencja.png")
    print("- statystyka_firmy.png")
    print(f"- mapa_optymalna_{args.business_type}.png")

if __name__ == "__main__":
    main()