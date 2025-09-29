import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
import time

INPUT_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\pdf_links.txt"
OUTPUT_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\pdfs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Los PDFs se guardarán en: {OUTPUT_DIR}")

chrome_options = Options()

chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

prefs = {
    "download.default_directory": OUTPUT_DIR,
    "download.prompt_for_download": False, 
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True 
}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--headless=new") 
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

service = Service(r"C:\Users\axelq\Documents\chromedriver\chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)

with open(INPUT_FILE, "r") as f:
    pdf_links = [line.strip() for line in f.readlines()] 

for url in tqdm(pdf_links, desc="Descargando PDFs con Selenium"):
    try:
        driver.get(url)
        print(f"Navegando a {url}. Esperando 15 segundos para la descarga...")
        time.sleep(15)
    except Exception as e:
        print(f"[ERROR] Error al procesar {url}: {e}")

print("Proceso de descarga finalizado. Cerrando el navegador.")
driver.quit()