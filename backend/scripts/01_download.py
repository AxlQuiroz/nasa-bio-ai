import os
import pandas as pd
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

CSV_file = r"C:\Users\axelq\Documents\nasa-bio-ai\SB_publication_PMC.csv"

df = pd.read_csv(CSV_file)
urls = df['Link'].tolist()

print(f"Se encontraron {len(urls)} enlaces")

def extract_pdf_url(url):
    """
    Extrae la URL del PDF de una página web.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, allow_redirects=True)  # Permitir redirecciones
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')

        # Buscar un iframe que contenga el PDF
        iframe = soup.find('iframe', {'id': 'pdfFrame'})
        if iframe:
            pdf_url = iframe['src']
            if not pdf_url.startswith('http'):
                pdf_url = urljoin(url, pdf_url)
            print(f"URL del PDF extraída (desde iframe): {pdf_url}")
            return pdf_url
        
        # Si no se encuentra el iframe, buscar un enlace directo al PDF
        pdf_link = soup.find('a', href=lambda href: href and href.endswith('.pdf'))
        
        if pdf_link:
            pdf_url = pdf_link['href']
            if not pdf_url.startswith('http'):
                pdf_url = urljoin(url, pdf_url)
            print(f"URL del PDF extraída (desde enlace): {pdf_url}")

            # **NUEVO: Analizar el HTML de la página del PDF**
            response = requests.get(pdf_url, headers=headers, allow_redirects=True)
            response.raise_for_status()
            pdf_soup = BeautifulSoup(response.content, 'html.parser')

            # **NUEVO: Buscar la URL del PDF dentro de un script**
            for script in pdf_soup.find_all('script'):
                script_text = script.get_text()
                # Buscar patrones comunes para la URL del PDF
                patterns = [
                    r"var pdf_link = '(.*?)';",
                    r"article_pdf_url = '(.*?)'",
                    r"pdfUrl = '(.*?)'",
                    r"pdf_url = \"(.*?)\"",
                    r"url: '(.*?)'",
                    r"src: '(.*?)'"
                ]
                for pattern in patterns:
                    match = re.search(pattern, script_text)
                    if match:
                        real_pdf_url = match.group(1)
                        if not real_pdf_url.startswith('http'):
                            real_pdf_url = urljoin(pdf_url, real_pdf_url)
                        print(f"URL REAL del PDF extraída (desde script con patrón {pattern}): {real_pdf_url}")
                        return real_pdf_url

            print("No se encontró la URL del PDF dentro de un script")
            return None
        else:
            print(f"No se encontró enlace a PDF en {url}")
            return None
    except Exception as e:
        print(f"Error extrayendo enlace de {url}: {e}")
        return None

def download_pdf(pdf_url, filename):
    """
    Descarga el PDF desde la URL y lo guarda en el archivo.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36"
        }
        r = requests.get(pdf_url, headers=headers, stream=True, allow_redirects=True)  # Permitir redirecciones y usar stream
        r.raise_for_status()

        # Imprimir el Content-Type
        print(f"Content-Type: {r.headers.get('Content-Type')}")

        # Verificar si el Content-Type es PDF
        if 'application/pdf' not in r.headers.get('Content-Type', ''):
            print(f"Error: El Content-Type no es application/pdf. Es: {r.headers.get('Content-Type')}")
            return False

        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):  # Escribir en chunks
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error descargando {pdf_url}: {e}")
        return False

for i, url in enumerate(tqdm(urls, desc="Descargando documentos")):
    filename = os.path.join(RAW_DIR, f"doc_{i+1}.pdf")
    
    if os.path.exists(filename):
        tqdm.write(f"Ya existe: {filename}, se omite")
        continue

    pdf_url = extract_pdf_url(url)
    if not pdf_url:
        continue

    if download_pdf(pdf_url, filename):
        print(f"Descargado correctamente: {filename}")
    else:
        print(f"Error al descargar: {filename}")