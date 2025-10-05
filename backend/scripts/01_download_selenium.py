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
CSV_file = r"C:\Users\Quiroz\Documents\links.csv"  # Ajusta ruta
df = pd.read_csv(CSV_file)

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

LOG_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'download_log_test.csv')
ERROR_LOG = os.path.join(os.path.dirname(__file__), '..', 'data', 'download_errors_test.txt')

# --- Inicializar log CSV si no existe ---
if not os.path.exists(LOG_CSV):
    pd.DataFrame(columns=['PMC_ID', 'Title', 'PDF_downloaded', 'File_path', 'Error']).to_csv(LOG_CSV, index=False)

# --- Configuración Selenium ---
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
service = Service(r"C:\Users\Quiroz\Documents\chromedriver\chromedriver.exe")  # Ajusta ruta
driver = webdriver.Chrome(service=service, options=chrome_options)

# --- Función para descargar PDF ---
def download_pdf(url, filename, retries=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            return True, ""
        except Exception as e:
            print(f"[REINTENTO {attempt}] Error descargando {url}: {e}")
            time.sleep(2)
            last_error = str(e)
    return False, last_error

# --- Tomar solo las primeras 5 URLs para prueba ---
df_subset = df.head(5)

# --- Procesar artículos ---
for i, row in enumerate(tqdm(df_subset.itertuples(), total=len(df_subset), desc="Descargando PDFs (prueba)")):
    title = getattr(row, "Title", f"doc_{i+1}")
    pmc_link = getattr(row, "Link", "")
    filename = os.path.join(RAW_DIR, f"doc_test_{i+1}.pdf")

    if os.path.exists(filename):
        print(f"[EXISTE] {filename} ya descargado, se omite")
        continue

    try:
        driver.get(pmc_link)
        time.sleep(3)  # Espera carga completa

        # Buscar enlaces que contengan /pdf/
        pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '/pdf/')]")

        if not pdf_links:
            # Intentar buscar botón con "PDF" u otros
            pdf_btns = driver.find_elements(By.XPATH, "//a[contains(text(),'PDF') or contains(text(),'Full Text PDF')]")
            pdf_links = pdf_btns

        if not pdf_links:
            print(f"[NO PDF] No se encontró PDF en {pmc_link}")
            with open(ERROR_LOG, "a") as f:
                f.write(f"{pmc_link}\n")
            log_entry = pd.DataFrame([{'PMC_ID': '', 'Title': title, 'PDF_downloaded': False, 'File_path': '', 'Error': 'PDF no encontrado'}])
            log_entry.to_csv(LOG_CSV, mode='a', header=False, index=False)
            continue

        pdf_url = pdf_links[0].get_attribute("href")
        success, error_msg = download_pdf(pdf_url, filename)

        log_entry = pd.DataFrame([{
            'PMC_ID': '',
            'Title': title,
            'PDF_downloaded': success,
            'File_path': filename if success else '',
            'Error': error_msg
        }])
        log_entry.to_csv(LOG_CSV, mode='a', header=False, index=False)

        if success:
            print(f"[DESCARGADO] {filename}")
        else:
            with open(ERROR_LOG, "a") as f:
                f.write(f"{pmc_link} - {error_msg}\n")

    except Exception as e:
        print(f"[ERROR] Fallo general en {pmc_link}: {e}")
        with open(ERROR_LOG, "a") as f:
            f.write(f"{pmc_link} - {e}\n")
        log_entry = pd.DataFrame([{'PMC_ID': '', 'Title': title, 'PDF_downloaded': False, 'File_path': '', 'Error': str(e)}])
        log_entry.to_csv(LOG_CSV, mode='a', header=False, index=False)
        continue

driver.quit()
print("Prueba finalizada para las 5 URLs.")
