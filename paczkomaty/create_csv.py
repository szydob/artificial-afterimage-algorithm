import requests
import csv

# URL do Overpass API
overpass_url = "http://overpass-api.de/api/interpreter"

# Zapytanie Overpass do pobrania paczkomatów w Polsce
overpass_query = """
[out:json][timeout:25];
area["name"="Poland"]->.pol;
node(area.pol)["amenity"="vending_machine"]["parcel_pickup"];
out center;
"""

# Wykonanie zapytania do Overpass API
response = requests.get(overpass_url, params={'data': overpass_query})

# Sprawdzanie, czy zapytanie się powiodło
if response.status_code == 200:
    data = response.json()
else:
    print("Błąd podczas pobierania danych")
    exit()

# Tworzenie pliku CSV
with open("paczkomaty_poland.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "Nazwa", "Szerokość", "Długość", "Adres"])  # Nagłówki CSV
    
    # Zapisanie danych paczkomatów do CSV
    for element in data['elements']:
        id_paczkomatu = element['id']
        lat = element['lat']
        lon = element['lon']
        # Zakładając, że adres jest częścią tagu w OSM
        adres = element.get('tags', {}).get('name', 'Brak adresu')
        
        # Zapisanie do CSV
        writer.writerow([id_paczkomatu, adres, lat, lon])

print("Dane paczkomatów zostały zapisane w pliku paczkomaty_poland.csv.")