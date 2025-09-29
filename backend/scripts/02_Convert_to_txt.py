import os
import fitz  # PyMuPDF
from tqdm import tqdm

# Directorio donde están los PDFs descargados
PDF_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\pdfs"
# Directorio donde guardaremos los archivos de texto
TXT_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\txts"

# Crear el directorio de salida si no existe
os.makedirs(TXT_DIR, exist_ok=True)

# Obtener la lista de todos los archivos PDF en el directorio
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

print(f"Se encontraron {len(pdf_files)} archivos PDF para convertir.")

# Procesar cada archivo PDF
for pdf_file in tqdm(pdf_files, desc="Convirtiendo PDF a TXT"):
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
    txt_path = os.path.join(TXT_DIR, txt_filename)

    # Revisar si el archivo ya fue convertido para no repetir trabajo
    if os.path.exists(txt_path):
        continue

    try:
        # Abrir el documento PDF
        doc = fitz.open(pdf_path)
        full_text = ""
        
        # Iterar sobre cada página y extraer el texto
        for page in doc:
            full_text += page.get_text()
        
        # Guardar el texto extraído en un archivo .txt
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        doc.close()

    except Exception as e:
        print(f"\n[ERROR] No se pudo procesar el archivo {pdf_file}: {e}")

print("\nConversión de todos los archivos completada.")
print(f"Los archivos de texto se han guardado en: {TXT_DIR}")