import os
import requests

url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4136787/pdf/pone.0104830.pdf"
filename = r"C:\Users\Quiroz\Documents\nasa-bio-ai\scripts\data\raw\test.pdf"

# Crear carpeta si no existe
os.makedirs(os.path.dirname(filename), exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://pmc.ncbi.nlm.nih.gov/",
    "Connection": "keep-alive",
}

try:
    session = requests.Session()
    response = session.get(url, headers=headers, allow_redirects=True, timeout=20)
    response.raise_for_status()

    # Validar si realmente es PDF
    if "application/pdf" in response.headers.get("Content-Type", "").lower():
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ PDF descargado correctamente: {filename}")
    else:
        print("⚠️ El servidor no devolvió un PDF. Posible bloqueo o HTML recibido.")
        print(response.text[:500])  # muestra los primeros 500 chars para debug

except Exception as e:
    print(f"❌ Error: {e}")
