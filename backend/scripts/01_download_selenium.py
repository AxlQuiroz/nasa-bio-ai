import os
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import time

# --- Configuración CSV y carpeta de descarga ---
CSV_file = r"C:\Users\Quiroz\Documents\links.csv"
df = pd.read_csv(CSV_file)

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)
print(f"Archivos se guardarán en: {RAW_DIR}")

# --- Configuración de Selenium ---
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ejecuta sin abrir ventana
chrome_options.add_argument("--disable-gpu")

service = Service(r"C:\Users\Quiroz\Documents\chromedriver\chromedriver.exe")  # Ajusta la ruta a tu chromedriver
driver = webdriver.Chrome(service=service, options=chrome_options)

# --- Probar solo los primeros 5 artículos ---
for i, url in enumerate(tqdm(df['Link'][:5], desc="Descargando PDFs con Selenium")):
    try:
        driver.get(url)
        time.sleep(2)  # Espera que cargue la página

        # Busca el botón/link de PDF
        pdf_button = driver.find_element(By.LINK_TEXT, "PDF")
        pdf_url = pdf_button.get_attribute("href")
        print(f"[OK] PDF encontrado: {pdf_url}")

        # Descarga con requests
        filename = os.path.join(RAW_DIR, f"doc_{i+1}.pdf")
        if os.path.exists(filename):
            print(f"Ya existe: {filename}, se omite")
            continue

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36"
        }
        r = requests.get(pdf_url, headers=headers)
        r.raise_for_status()

        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"[DESCARGADO] {filename}")

    except Exception as e:
        print(f"[ERROR] No se pudo descargar {url}: {e}")
        continue

driver.quit()
print("Prueba completada.")
