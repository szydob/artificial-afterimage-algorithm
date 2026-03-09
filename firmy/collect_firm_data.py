import requests
import sqlite3
import pandas as pd
import numpy as np
from time import sleep
import json
import os
import argparse

class FirmDataCollector:
    def __init__(self, db_path='firmy.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """Create database tables for storing firm data"""
        cursor = self.conn.cursor()

        # Table for firms
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS firms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                type TEXT,
                lat REAL,
                lon REAL,
                address TEXT,
                pkd TEXT,
                regon TEXT,
                competitors_1km INTEGER,
                competitors_3km INTEGER,
                competitors_5km INTEGER,
                population_density REAL,
                income_level REAL,
                unemployment_rate REAL,
                accessibility_score REAL,
                rental_cost REAL
            )
        ''')

        # Table for regions/demographics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_name TEXT,
                population INTEGER,
                area REAL,
                income_per_capita REAL,
                unemployment_rate REAL,
                lat REAL,
                lon REAL
            )
        ''')

        self.conn.commit()

    def fetch_osm_data(self, bbox=None, business_types=None):
        """Fetch business data from OpenStreetMap using Overpass API"""
        if bbox is None:
            # Default to Krakow city center area
            bbox = [49.95, 19.90, 50.05, 20.10]  # [south, west, north, east]

        if business_types is None:
            business_types = ['restaurant', 'cafe', 'bank', 'pharmacy', 'supermarket', 'shop']

        overpass_url = "http://overpass-api.de/api/interpreter"
        firms = []

        for business_type in business_types:
            query = f"""
            [out:json][timeout:25];
            (
                node["amenity"="{business_type}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
                way["amenity"="{business_type}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
                relation["amenity"="{business_type}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
                node["shop"="{business_type}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
                way["shop"="{business_type}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
                relation["shop"="{business_type}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
            );
            out center;
            """

            try:
                response = requests.post(overpass_url, data=query, timeout=30)
                response.raise_for_status()
                data = response.json()

                for element in data['elements']:
                    if 'center' in element:
                        lat, lon = element['center']['lat'], element['center']['lon']
                    elif 'lat' in element and 'lon' in element:
                        lat, lon = element['lat'], element['lon']
                    else:
                        continue

                    name = element.get('tags', {}).get('name', f'{business_type}_{element["id"]}')

                    firms.append({
                        'id': element['id'],
                        'name': name,
                        'type': business_type,
                        'lat': lat,
                        'lon': lon,
                        'address': element.get('tags', {}).get('addr:full', ''),
                        'pkd': business_type,  # Simplified PKD mapping
                        'regon': None  # Will be filled from GUS if available
                    })

                print(f"Fetched {len(data['elements'])} {business_type} locations")
                sleep(1)  # Be nice to the API

            except Exception as e:
                print(f"Error fetching {business_type}: {e}")
                continue

        return firms

    def fetch_gus_demographics(self, region_code=None):
        """Fetch demographic data from GUS BDL API"""
        # GUS BDL API - simplified example
        # In practice, you'd need proper API keys and endpoints

        base_url = "https://bdl.stat.gov.pl/api/v1/data"
        # Example: Population data for Małopolskie voivodeship
        if region_code is None:
            region_code = "12"  # Małopolskie

        try:
            # This is a simplified example - real API requires authentication
            # and specific variable IDs
            params = {
                'format': 'json',
                'id': '1',  # Population variable
                'unit-id': region_code,
                'year': '2023'
            }

            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse and return demographic data
            demographics = {
                'region_name': 'Małopolskie',
                'population': 3412000,  # Approximate
                'area': 15183,  # km²
                'income_per_capita': 45000,  # PLN
                'unemployment_rate': 0.045,
                'lat': 49.95,
                'lon': 20.0
            }

            return demographics

        except Exception as e:
            print(f"Error fetching GUS data: {e}")
            return None

    def calculate_competitors(self, firms_df, radius_km=1):
        """Calculate number of competitors within given radius"""
        from sklearn.metrics.pairwise import haversine_distances
        from math import radians

        # Convert to radians for haversine
        coords = np.radians(firms_df[['lat', 'lon']].values)

        # Calculate distances
        distances = haversine_distances(coords) * 6371  # Earth radius in km

        competitors = []
        for i in range(len(firms_df)):
            # Count firms within radius (excluding self)
            count = np.sum((distances[i] <= radius_km) & (distances[i] > 0))
            competitors.append(count)

        return competitors

    def enrich_firm_data(self, firms):
        """Enrich firm data with calculated features"""
        df = pd.DataFrame(firms)

        if len(df) == 0:
            return df

        # Calculate competitors at different radii
        df['competitors_1km'] = self.calculate_competitors(df, 1)
        df['competitors_3km'] = self.calculate_competitors(df, 3)
        df['competitors_5km'] = self.calculate_competitors(df, 5)

        # Add demographic data (simplified - in reality would interpolate)
        demographics = self.fetch_gus_demographics()
        if demographics:
            df['population_density'] = demographics['population'] / demographics['area']
            df['income_level'] = demographics['income_per_capita']
            df['unemployment_rate'] = demographics['unemployment_rate']
        else:
            # Default values
            df['population_density'] = 200  # people/km²
            df['income_level'] = 40000
            df['unemployment_rate'] = 0.05

        # Calculate accessibility score (simplified)
        # Higher in city centers, lower in outskirts
        center_lat, center_lon = 50.0, 20.0
        distances_from_center = np.sqrt((df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2)
        df['accessibility_score'] = np.maximum(0.1, 1.0 - distances_from_center * 10)

        # Estimate rental cost (higher near center)
        df['rental_cost'] = 1000 + distances_from_center * 5000

        return df

    def save_to_database(self, firms_df):
        """Save enriched firm data to database"""
        firms_df.to_sql('firms', self.conn, if_exists='replace', index=False)
        print(f"Saved {len(firms_df)} firms to database")

    def collect_and_save_data(self, bbox=None, business_types=None):
        """Main method to collect and save all data"""
        print("Fetching firm data from OpenStreetMap...")
        firms = self.fetch_osm_data(bbox, business_types)

        print("Enriching data with calculated features...")
        enriched_df = self.enrich_firm_data(firms)

        print("Saving to database...")
        self.save_to_database(enriched_df)

        print("Fetching demographic data...")
        demographics = self.fetch_gus_demographics()
        if demographics:
            demographics_df = pd.DataFrame([demographics])
            demographics_df.to_sql('regions', self.conn, if_exists='replace', index=False)
            print("Saved demographic data")

        self.conn.close()
        print("Data collection complete!")

        return enriched_df

def main():
    parser = argparse.ArgumentParser(description='Collect firm data from OpenStreetMap')
    parser.add_argument('--bbox', nargs=4, type=float,
                       default=[49.9, 19.8, 50.1, 20.2],
                       help='Bounding box: south west north east')
    parser.add_argument('--business-types', nargs='+',
                       default=['restaurant', 'cafe', 'bank', 'pharmacy', 'supermarket', 'shop'],
                       help='Business types to collect')
    parser.add_argument('--db-path', default='firmy.db', help='Database path')

    args = parser.parse_args()

    # Example usage
    collector = FirmDataCollector(args.db_path)

    df = collector.collect_and_save_data(bbox=args.bbox, business_types=args.business_types)

    print(f"Collected data for {len(df)} firms")
    print(df.head())

if __name__ == "__main__":
    main()