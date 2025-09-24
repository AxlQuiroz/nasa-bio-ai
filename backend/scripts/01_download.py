import os
import pandas as pd
import requests
from tqdm import tqdm

# Carpeta donde se guardarán los archivos
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# CSV con los enlaces de descarga
CSV_file = r"C:\Users\Quiroz\Documents\links.csv"

# Leer el CSV
df = pd.read_csv(CSV_file)
urls = df['Link'].tolist()

print(f"Se encontraron {len(urls)} enlaces")
print("Primeros 5 enlaces:", urls[:5])

for i, url in enumerate(tqdm(urls, desc="Descargando documentos")):
    filename = os.path.join(RAW_DIR, f"doc_{i+1}.pdf")
    
    if os.path.exists(filename):
        tqdm.write(f"Ya existe: {filename}, se omite")
        continue

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36"
        }
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Error descargando {url}: {e}")
