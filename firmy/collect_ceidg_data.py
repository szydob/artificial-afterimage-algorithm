import requests
import pandas as pd
import sqlite3
import time
from datetime import datetime
import argparse
import random
from faker import Faker

class CEIDGDataCollector:
    def __init__(self, api_key=None, db_path='firmy.db'):
        """
        Initialize CEIDG data collector
        Note: CEIDG API requires registration and API key
        """
        self.api_key = api_key
        self.base_url = "https://dane.biznes.gov.pl/api/ceidg/v2"
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def create_ceidg_table(self):
        """Create table for CEIDG data"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ceidg_firms (
                nip TEXT PRIMARY KEY,
                regon TEXT,
                nazwa TEXT,
                wojewodztwo TEXT,
                powiat TEXT,
                gmina TEXT,
                miejscowosc TEXT,
                kod_pocztowy TEXT,
                ulica TEXT,
                nr_domu TEXT,
                nr_lokalu TEXT,
                data_powstania TEXT,
                data_zakonczenia TEXT,
                status TEXT,
                pkd TEXT,
                pkd_nazwa TEXT,
                liczba_zatrudnionych INTEGER,
                forma_prawna TEXT,
                data_wpisu TEXT,
                data_wykreslenia TEXT
            )
        ''')
        self.conn.commit()

    def fetch_ceidg_data(self, wojewodztwo='MAŁOPOLSKIE', limit=1000):
        """
        Fetch data from CEIDG API
        Note: This is a simplified example. Real API has different endpoints and auth.
        """
        if not self.api_key:
            print("Warning: No API key provided. Using mock data.")
            return self.generate_mock_ceidg_data(limit)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        params = {
            'wojewodztwo': wojewodztwo,
            'limit': limit,
            'offset': 0
        }

        firms = []
        try:
            while len(firms) < limit:
                response = requests.get(f"{self.base_url}/firmy",
                                      headers=headers,
                                      params=params,
                                      timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if not data.get('firmy'):
                        break

                    for firma in data['firmy']:
                        firms.append({
                            'nip': firma.get('nip'),
                            'regon': firma.get('regon'),
                            'nazwa': firma.get('nazwa'),
                            'wojewodztwo': firma.get('adres', {}).get('wojewodztwo'),
                            'powiat': firma.get('adres', {}).get('powiat'),
                            'gmina': firma.get('adres', {}).get('gmina'),
                            'miejscowosc': firma.get('adres', {}).get('miejscowosc'),
                            'kod_pocztowy': firma.get('adres', {}).get('kod_pocztowy'),
                            'ulica': firma.get('adres', {}).get('ulica'),
                            'nr_domu': firma.get('adres', {}).get('nr_domu'),
                            'nr_lokalu': firma.get('adres', {}).get('nr_lokalu'),
                            'data_powstania': firma.get('data_powstania'),
                            'data_zakonczenia': firma.get('data_zakonczenia'),
                            'status': firma.get('status'),
                            'pkd': firma.get('pkd', [{}])[0].get('kod'),
                            'pkd_nazwa': firma.get('pkd', [{}])[0].get('nazwa'),
                            'liczba_zatrudnionych': firma.get('liczba_zatrudnionych'),
                            'forma_prawna': firma.get('forma_prawna'),
                            'data_wpisu': firma.get('data_wpisu'),
                            'data_wykreslenia': firma.get('data_wykreslenia')
                        })

                    params['offset'] += len(data['firmy'])
                    time.sleep(1)  # Rate limiting

                else:
                    print(f"API Error: {response.status_code}")
                    break

        except Exception as e:
            print(f"Error fetching CEIDG data: {e}")
            return self.generate_mock_ceidg_data(limit)

        return firms[:limit]

    def generate_mock_ceidg_data(self, limit=100):
        """Generate mock CEIDG data for testing"""
        import random
        from faker import Faker
        fake = Faker('pl_PL')

        pkd_codes = {
            '56.10.Z': 'Restauracje i inne placówki gastronomiczne',
            '47.11.Z': 'Handel detaliczny w niewyspecjalizowanych sklepach',
            '64.19.Z': 'Pozostałe formy udzielania kredytów',
            '86.21.Z': 'Praktyka lekarska ogólna',
            '96.02.Z': 'Fryzjerstwo i pozostałe zabiegi kosmetyczne'
        }

        firms = []
        for i in range(limit):
            pkd_code = random.choice(list(pkd_codes.keys()))
            firms.append({
                'nip': f"{random.randint(1000000000, 9999999999)}",
                'regon': f"{random.randint(100000000, 999999999)}",
                'nazwa': fake.company(),
                'wojewodztwo': 'MAŁOPOLSKIE',
                'powiat': fake.city(),
                'gmina': fake.city(),
                'miejscowosc': fake.city(),
                'kod_pocztowy': fake.postcode(),
                'ulica': fake.street_name(),
                'nr_domu': fake.building_number(),
                'nr_lokalu': random.choice([None, fake.building_number()]),
                'data_powstania': fake.date_between(start_date='-10y', end_date='today').isoformat(),
                'data_zakonczenia': None,
                'status': 'AKTYWNY',
                'pkd': pkd_code,
                'pkd_nazwa': pkd_codes[pkd_code],
                'liczba_zatrudnionych': random.randint(1, 50),
                'forma_prawna': random.choice(['JEDNOOSOBOWA DZIAŁALNOŚĆ GOSPODARCZA', 'SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ']),
                'data_wpisu': fake.date_between(start_date='-10y', end_date='today').isoformat(),
                'data_wykreslenia': None
            })

        return firms

    def geocode_addresses(self, firms_df):
        """Add coordinates to firms using geocoding"""
        # This would use a geocoding service like Nominatim
        # For now, add mock coordinates for Krakow area

        krakow_center_lat, krakow_center_lon = 50.0647, 19.9450

        def mock_geocode(row):
            # Add some random offset
            lat_offset = (random.random() - 0.5) * 0.2
            lon_offset = (random.random() - 0.5) * 0.2
            return pd.Series({
                'lat': krakow_center_lat + lat_offset,
                'lon': krakow_center_lon + lon_offset
            })

        coords = firms_df.apply(mock_geocode, axis=1)
        firms_df = pd.concat([firms_df, coords], axis=1)
        return firms_df

    def save_to_database(self, firms_df):
        """Save CEIDG data to database"""
        firms_df.to_sql('ceidg_firms', self.conn, if_exists='replace', index=False)
        print(f"Saved {len(firms_df)} CEIDG firms to database")

    def collect_and_save_ceidg_data(self, wojewodztwo='MAŁOPOLSKIE', limit=1000):
        """Main method to collect CEIDG data"""
        print("Creating CEIDG table...")
        self.create_ceidg_table()

        print(f"Fetching CEIDG data for {wojewodztwo}...")
        firms = self.fetch_ceidg_data(wojewodztwo, limit)

        print("Converting to DataFrame...")
        df = pd.DataFrame(firms)

        print("Adding coordinates...")
        df = self.geocode_addresses(df)

        print("Saving to database...")
        self.save_to_database(df)

        self.conn.close()
        print("CEIDG data collection complete!")

        return df

def main():
    parser = argparse.ArgumentParser(description='Collect firm data from CEIDG API')
    parser.add_argument('--api-key', help='CEIDG API key')
    parser.add_argument('--wojewodztwo', default='MAŁOPOLSKIE', help='Wojewodztwo to collect data for')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum number of firms to collect')
    parser.add_argument('--db-path', default='firmy.db', help='Database path')

    args = parser.parse_args()

    collector = CEIDGDataCollector(api_key=args.api_key, db_path=args.db_path)

    df = collector.collect_and_save_ceidg_data(wojewodztwo=args.wojewodztwo, limit=args.limit)

    print(f"Collected data for {len(df)} firms")
    print(df.head())

if __name__ == "__main__":
    main()