import os
import subprocess
from tqdm import tqdm
import requests

INPUT_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\pdf_links.txt"
OUTPUT_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\pdfs"

with open(INPUT_FILE, "r") as f:
    pdf_links = [line.strip() for line in f.readlines()[:5]]  # Leer solo los primeros 5 enlaces
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

for i, url in enumerate(tqdm(pdf_links, desc="Descargando PDFs")):
    filename = os.path.join(OUTPUT_DIR, f"doc_{i + 1}.pdf")

    try:
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()  

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[DESCARGADO] PDF desde {url} guardado como {filename} (con requests)")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Error al descargar el PDF con requests: {e}. Intentando con wget...")

        try:
            command = [
                "wget",
                "-O",
                filename,
                "-L",  
                "--user-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "--no-check-certificate",
                url,
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60) 
            if result.returncode != 0:
                print(f"[ERROR] wget devolvió el código de error {result.returncode}: {result.stderr}")
            else:
                print(f"[DESCARGADO] PDF desde {url} guardado como {filename} (con wget)")

        except subprocess.TimeoutExpired:
            print(f"[ERROR] wget timeout al descargar {url}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error al descargar el PDF con wget: {e}")
        except Exception as e:
            print(f"[ERROR] Error inesperado al usar wget: {e}")
    except Exception as e:
        print(f"[ERROR] Error inesperado: {e}")

print("Descarga completada.")