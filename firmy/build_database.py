import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
import os
import json

class FirmDatabaseBuilder:
    def __init__(self, db_path='firmy.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def merge_data_sources(self):
        """Merge data from OSM, CEIDG, and demographics"""
        print("Loading data from different sources...")

        # Load OSM data
        try:
            osm_df = pd.read_sql_query("SELECT * FROM firms", self.conn)
            print(f"Loaded {len(osm_df)} firms from OSM")
        except:
            print("No OSM data found")
            osm_df = pd.DataFrame()

        # Load CEIDG data
        try:
            ceidg_df = pd.read_sql_query("SELECT * FROM ceidg_firms", self.conn)
            print(f"Loaded {len(ceidg_df)} firms from CEIDG")
        except:
            print("No CEIDG data found")
            ceidg_df = pd.DataFrame()

        # Load demographics
        try:
            demo_df = pd.read_sql_query("SELECT * FROM regions", self.conn)
            print(f"Loaded {len(demo_df)} regions from demographics")
        except:
            print("No demographics data found")
            demo_df = pd.DataFrame()

        return osm_df, ceidg_df, demo_df

    def standardize_business_types(self, df):
        """Map different business type classifications to standard categories"""
        type_mapping = {
            # OSM to standard
            'restaurant': 'gastronomia',
            'cafe': 'gastronomia',
            'bar': 'gastronomia',
            'pub': 'gastronomia',
            'fast_food': 'gastronomia',
            'bakery': 'handel',
            'supermarket': 'handel',
            'shop': 'handel',
            'butcher': 'handel',
            'clothes': 'handel',
            'electronics': 'handel',
            'hairdresser': 'usługi kosmetyczne',
            'beauty': 'usługi kosmetyczne',
            'bank': 'usługi finansowe',
            'atm': 'usługi finansowe',
            'pharmacy': 'zdrowie',
            'hospital': 'zdrowie',
            'dentist': 'zdrowie',
            'doctor': 'zdrowie',
            'school': 'edukacja',
            'library': 'edukacja',
            'hotel': 'turystyka',
            'cinema': 'rozrywka',
            'theatre': 'rozrywka',
            'fuel': 'usługi transportowe',
            'post_office': 'usługi pocztowe',

            # CEIDG PKD to standard (simplified)
            '56.10.Z': 'gastronomia',
            '47.11.Z': 'handel',
            '64.19.Z': 'usługi finansowe',
            '86.21.Z': 'zdrowie',
            '96.02.Z': 'usługi kosmetyczne',
            '85.10.Z': 'edukacja',
            '55.10.Z': 'turystyka',
            '90.01.Z': 'rozrywka'
        }

        if 'type' in df.columns:
            df['business_category'] = df['type'].map(type_mapping).fillna('inne')
        elif 'pkd' in df.columns:
            df['business_category'] = df['pkd'].map(type_mapping).fillna('inne')

        return df

    def calculate_location_features(self, df):
        """Calculate location-based features for each firm"""
        if len(df) == 0 or 'lat' not in df.columns or 'lon' not in df.columns:
            return df

        # Convert coordinates to radians for distance calculations
        coords = np.radians(df[['lat', 'lon']].values)

        # Calculate competitors within different radii
        distances = haversine_distances(coords) * 6371  # km

        for radius in [1, 3, 5]:
            competitors = []
            for i in range(len(df)):
                # Count competitors of same type within radius
                same_type = df['business_category'] == df.iloc[i]['business_category']
                within_radius = distances[i] <= radius
                count = np.sum(same_type & within_radius & (distances[i] > 0))
                competitors.append(count)

            df[f'competitors_{radius}km'] = competitors

        # Calculate distance to city center (Krakow)
        krakow_center = np.radians([50.0647, 19.9450])
        center_distances = haversine_distances(coords, krakow_center.reshape(1, -1)) * 6371
        df['distance_to_center'] = center_distances.flatten()

        # Estimate population density (higher near center)
        df['population_density'] = 1000 / (1 + df['distance_to_center'])

        # Estimate accessibility score (higher near center and transport hubs)
        df['accessibility_score'] = np.maximum(0.1, 1.0 - df['distance_to_center'] / 20)

        # Estimate rental cost (higher near center)
        df['rental_cost'] = 2000 + df['distance_to_center'] * 100

        # Estimate income level (correlated with distance)
        df['income_level'] = 60000 - df['distance_to_center'] * 500

        return df

    def create_unified_dataset(self):
        """Create a unified dataset from all sources"""
        osm_df, ceidg_df, demo_df = self.merge_data_sources()

        # Standardize columns
        unified_columns = [
            'name', 'business_category', 'lat', 'lon', 'address',
            'pkd', 'regon', 'nip', 'status', 'employees',
            'competitors_1km', 'competitors_3km', 'competitors_5km',
            'population_density', 'income_level', 'accessibility_score', 'rental_cost',
            'source'  # OSM or CEIDG
        ]

        unified_data = []

        # Process OSM data
        if not osm_df.empty:
            osm_df = self.standardize_business_types(osm_df)
            osm_df = self.calculate_location_features(osm_df)
            osm_df['source'] = 'OSM'
            osm_df['nip'] = None
            osm_df['regon'] = None
            osm_df['status'] = 'active'
            osm_df['employees'] = np.random.randint(1, 20, len(osm_df))  # Estimate

            osm_unified = osm_df[[
                'name', 'business_category', 'lat', 'lon', 'address',
                'pkd', 'regon', 'nip', 'status', 'competitors_1km',
                'competitors_3km', 'competitors_5km', 'population_density',
                'income_level', 'accessibility_score', 'rental_cost', 'source'
            ]].copy()
            osm_unified['employees'] = osm_df.get('liczba_zatrudnionych', np.random.randint(1, 20, len(osm_df)))

            unified_data.append(osm_unified)

        # Process CEIDG data
        if not ceidg_df.empty:
            ceidg_df = self.standardize_business_types(ceidg_df)
            ceidg_df = self.calculate_location_features(ceidg_df)
            ceidg_df['source'] = 'CEIDG'
            ceidg_df['address'] = ceidg_df.apply(
                lambda x: f"{x.get('ulica', '')} {x.get('nr_domu', '')} {x.get('nr_lokalu', '')}, {x.get('kod_pocztowy', '')} {x.get('miejscowosc', '')}".strip(),
                axis=1
            )

            ceidg_unified = ceidg_df[[
                'nazwa', 'business_category', 'lat', 'lon', 'address',
                'pkd', 'regon', 'nip', 'status', 'liczba_zatrudnionych',
                'competitors_1km', 'competitors_3km', 'competitors_5km',
                'population_density', 'income_level', 'accessibility_score', 'rental_cost', 'source'
            ]].copy()
            ceidg_unified.columns = unified_columns

            unified_data.append(ceidg_unified)

        if unified_data:
            final_df = pd.concat(unified_data, ignore_index=True)

            # Remove duplicates based on coordinates and name
            final_df = final_df.drop_duplicates(subset=['lat', 'lon', 'name'])

            # Save unified dataset
            final_df.to_sql('unified_firms', self.conn, if_exists='replace', index=False)
            final_df.to_csv('firmy_unified.csv', index=False)

            print(f"Created unified dataset with {len(final_df)} firms")
            print(final_df['business_category'].value_counts())
            print(final_df['source'].value_counts())

            return final_df
        else:
            print("No data to unify")
            return pd.DataFrame()

    def create_location_optimization_dataset(self):
        """Create dataset specifically for location optimization"""
        try:
            df = pd.read_sql_query("SELECT * FROM unified_firms", self.conn)
        except:
            print("No unified firms data found. Run create_unified_dataset first.")
            return

        # Select features for optimization
        features = [
            'lat', 'lon', 'population_density', 'competitors_1km',
            'competitors_3km', 'accessibility_score', 'rental_cost',
            'income_level', 'business_category'
        ]

        opt_df = df[features].copy()

        # Normalize numerical features
        numerical_cols = [
            'population_density', 'competitors_1km', 'competitors_3km',
            'accessibility_score', 'rental_cost', 'income_level'
        ]

        scaler = StandardScaler()
        opt_df[numerical_cols] = scaler.fit_transform(opt_df[numerical_cols])

        # Save normalized dataset
        opt_df.to_sql('optimization_dataset', self.conn, if_exists='replace', index=False)
        opt_df.to_csv('firmy_optimization.csv', index=False)

        print(f"Created optimization dataset with {len(opt_df)} firms")
        print("Features:", features)

        return opt_df

    def generate_summary_report(self):
        """Generate summary statistics of the collected data"""
        try:
            df = pd.read_sql_query("SELECT * FROM unified_firms", self.conn)

            report = {
                'total_firms': len(df),
                'business_categories': df['business_category'].value_counts().to_dict(),
                'sources': df['source'].value_counts().to_dict(),
                'avg_competitors_1km': df['competitors_1km'].mean(),
                'avg_competitors_3km': df['competitors_3km'].mean(),
                'avg_population_density': df['population_density'].mean(),
                'avg_rental_cost': df['rental_cost'].mean(),
                'coordinate_bounds': {
                    'lat_min': df['lat'].min(),
                    'lat_max': df['lat'].max(),
                    'lon_min': df['lon'].min(),
                    'lon_max': df['lon'].max()
                }
            }

            # Save report
            with open('firmy_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print("Summary report generated:")
            print(json.dumps(report, indent=2))

            return report

        except Exception as e:
            print(f"Error generating report: {e}")
            return None

def main():
    builder = FirmDatabaseBuilder('firmy.db')

    print("Step 1: Creating unified dataset...")
    unified_df = builder.create_unified_dataset()

    print("\nStep 2: Creating optimization dataset...")
    opt_df = builder.create_location_optimization_dataset()

    print("\nStep 3: Generating summary report...")
    report = builder.generate_summary_report()

    builder.conn.close()
    print("\nDatabase building complete!")

if __name__ == "__main__":
    main()