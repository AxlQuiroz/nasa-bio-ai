import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import time
import re

# --- Configuración CSV ---
CSV_file = r"C:\Users\axelq\Documents\nasa-bio-ai\SB_publication_PMC.csv"
df = pd.read_csv(CSV_file)

# --- Configuración de Selenium ---
chrome_options = Options()
chrome_options.add_argument("--headless=new")  # Ejecuta sin abrir ventana
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

service = Service(r"C:\Users\axelq\Documents\chromedriver\chromedriver.exe")  # Ajusta la ruta a tu chromedriver
driver = webdriver.Chrome(service=service, options=chrome_options)

# --- Archivo de salida ---
OUTPUT_FILE = "pdf_links.txt"

pdf_links = []

# --- Iterar sobre los enlaces ---
for i, url in enumerate(tqdm(df['Link'], desc="Extrayendo enlaces PDF")):
    try:
        driver.get(url)
        time.sleep(10)  # Espera que cargue la página

        # Busca el botón/link de PDF
        pdf_url = None
        pdf_texts = ["PDF", "Download PDF", "Full Text PDF", "PDF Full Text"]
        for text in pdf_texts:
            try:
                pdf_button = driver.find_element(By.LINK_TEXT, text)
                pdf_url = pdf_button.get_attribute("href")
                print(f"[OK] PDF encontrado con LINK_TEXT: {text} - {pdf_url}")
                break
            except:
                pass

        if not pdf_url:
            try:
                pdf_button = driver.find_element(By.XPATH, "//a[contains(@href, '.pdf')]")
                pdf_url = pdf_button.get_attribute("href")
                print(f"[OK] PDF encontrado con XPATH: {pdf_url}")
            except:
                print(f"[ERROR] No se pudo encontrar el enlace al PDF en {url}")

        if pdf_url:
            pdf_links.append(pdf_url)
        else:
            print(f"[ERROR] No se encontró el enlace al PDF para descargar.")

    except Exception as e:
        print(f"[ERROR] No se pudo procesar {url}: {e}")

driver.quit()
print("Extracción completada.")

# --- Guardar los enlaces en el archivo ---
with open(OUTPUT_FILE, "w") as f:
    for link in pdf_links:
        if link:
            f.write(link + "\n")

print(f"Enlaces guardados en {OUTPUT_FILE}")