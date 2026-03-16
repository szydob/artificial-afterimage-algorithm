# Firmy - Optymalizacja Lokalizacji Biznesu z AAIA

Ten folder zawiera implementację systemu do optymalizacji lokalizacji nowej firmy przy użyciu algorytmu Artificial Afterimage Algorithm (AAIA), wykorzystującego realne dane o istniejących firmach.

## Struktura

- `firmy.ipynb` - Główny notebook z implementacją algorytmu optymalizacji
- `collect_firm_data.py` - Skrypt do zbierania danych z OpenStreetMap
- `collect_ceidg_data.py` - Skrypt do zbierania danych z CEIDG API
- `build_database.py` - Skrypt do łączenia danych i tworzenia bazy danych
- `firmy.db` - Baza danych SQLite z zebranymi danymi
- `firmy_unified.csv` - Ujednolicony dataset firm
- `firmy_optimization.csv` - Dataset przygotowany do optymalizacji
- `firmy_report.json` - Raport podsumowujący zebrane dane

## Wymagania

- Python 3.13+
- Środowisko `firmy_env` (stworzone automatycznie)
- Zainstalowane zależności z `pyproject.toml`

## Jak uruchomić

### 1. Aktywacja środowiska
```bash
conda activate firmy_env
```

### 2. Zbieranie danych

#### Dane z OpenStreetMap (bezpłatne)
```bash
python collect_firm_data.py
```

#### Dane z CEIDG (wymaga rejestracji)
```bash
# Uzyskaj klucz API z https://dane.biznes.gov.pl/
python collect_ceidg_data.py
```

### 3. Budowa bazy danych
```bash
python build_database.py
```

### 4. Optymalizacja lokalizacji
Otwórz `firmy.ipynb` w VS Code i uruchom komórki (upewnij się, że kernel to `firmy_env`).

Skrypt `optimize_location.py` umożliwia również uruchomienie całego procesu z wiersza poleceń. Od wersji wykorzystującej AAIA można:

```bash
# optymalizacja na całym obszarze
python optimize_location.py --business-type gastronomia

# najpierw klasteryzacja z użyciem AAIA, a następnie wybór klastra "białej plamy"
python optimize_location.py --business-type gastronomia --clusters 5
```

Parametr `--clusters N` uruchamia algorytm AAIA, który iteracyjnie wyszukuje `N` środków klastrów oraz przydziela firmy do najbliższego centrum. Po wyznaczeniu klasterów algorytm wybiera ten z najmniejszym udziałem wskazanej kategorii (np. gastronomia) i optymalizuje lokalizację tylko w jego obrębie.

## Źródła danych

### 1. OpenStreetMap / Overpass API
- **Dane**: Lokalizacje firm, punktów usługowych
- **API**: http://overpass-api.de/api/interpreter
- **Korzyści**: Darmowe, globalne pokrycie
- **Ograniczenia**: Dane mogą być niekompletne dla małych firm

### 2. CEIDG / Biznes.gov.pl
- **Dane**: Rejestrowane działalności gospodarcze w Polsce
- **API**: https://dane.biznes.gov.pl/api/ceidg/
- **Korzyści**: Oficjalne dane, pełne informacje o firmach
- **Ograniczenia**: Wymaga rejestracji i klucza API

### 3. REGON / GUS
- **Dane**: Centralny Rejestr Podmiotów Gospodarki Narodowej
- **API**: https://api.stat.gov.pl/
- **Korzyści**: Wszystkie firmy, dane statystyczne
- **Ograniczenia**: Złożony dostęp

### 4. GUS BDL (Bank Danych Lokalnych)
- **Dane**: Dane demograficzne, ekonomiczne dla regionów
- **API**: https://bdl.stat.gov.pl/
- **Korzyści**: Szczegółowe dane lokalne
- **Ograniczenia**: Wymaga klucza API dla pełnego dostępu

## Funkcja celu optymalizacji

Funkcja celu maksymalizuje:

```
J(x,y) = w₁ × popyt(x,y) - w₂ × konkurencja(x,y) + w₃ × dostępność(x,y) - w₄ × koszt_wynajmu(x,y)
```

Gdzie:
- `popyt`: gęstość zaludnienia
- `konkurencja`: liczba konkurentów w okolicy
- `dostępność`: ocena dostępności komunikacyjnej
- `koszt_wynajmu`: szacowany koszt wynajmu nieruchomości

## Algorytm AAIA

Artificial Afterimage Algorithm:
1. Inicjalizuje populację kandydackich lokalizacji
2. Oblicza funkcję celu dla każdej lokalizacji
3. Wybiera najlepszą i najgorszą lokalizację
4. Generuje nowe kandydatów na podstawie percepcyjnego modelu widzenia
5. Powtarza aż do osiągnięcia kryterium stopu

## Wyniki

System generuje:
- Optymalną lokalizację dla nowej firmy
- Wizualizacje map z istniejącymi firmami
- Analizę konkurencji w regionie
- Metryki efektywności lokalizacji

## Przykład użycia

```python
from aaia import find_solution
import pandas as pd

# Wczytaj dane
df = pd.read_csv('firmy_optimization.csv')

# Uruchom optymalizację
optimal_location = find_solution(df.values, max_iterations=1000, population_size=50)

print("Optymalna lokalizacja:", optimal_location)
```

## Uwagi

- Dla pełnych danych CEIDG i GUS wymagane są klucze API
- Skrypty zawierają mock data dla celów demonstracyjnych
- Dane geograficzne są dla obszaru Krakowa (można zmienić bbox)
- Wszystkie współrzędne w systemie WGS84 (lat/lon)