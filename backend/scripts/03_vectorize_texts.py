import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
TXT_DIR = os.path.join(backend_dir, "data", "Processed")
VECTORS_DIR = os.path.join(backend_dir, "data", "Vectorized")


# Crear el directorio de salida si no existe
os.makedirs(VECTORS_DIR, exist_ok=True)


# --- Carga del Modelo ---
# intfloat/multilingual-e5-large es un modelo de embedding multilingüe de alta calidad.
print("Cargando el modelo de SentenceTransformer (multilingüe-e5-large)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')
print("Modelo cargado.")

# --- Función para dividir el texto en chunks ---
def chunk_text(text, chunk_size=512, overlap=50):
    """Divide el texto en chunks con un ligero solapamiento."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# --- Procesamiento de archivos ---
txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith(".txt")]

print(f"Se encontraron {len(txt_files)} archivos de texto para vectorizar.")

for txt_file in tqdm(txt_files, desc="Vectorizando archivos"):
    txt_path = os.path.join(TXT_DIR, txt_file)
    
    # El archivo de salida para los vectores tendrá extensión .npy
    vector_filename = os.path.splitext(txt_file)[0] + ".npy"
    vector_path = os.path.join(VECTORS_DIR, vector_filename)

    # Revisar si el archivo ya fue vectorizado
    if os.path.exists(vector_path):
        continue

    try:
        # Leer el contenido del archivo de texto
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Dividir el texto en chunks
        text_chunks = chunk_text(content)
        
        if not text_chunks:
            print(f"\n[ADVERTENCIA] El archivo {txt_file} está vacío o no generó chunks.")
            continue

        # 2. Añadir prefijo "passage: " a cada chunk para el modelo E5
        prefixed_chunks = [f"passage: {chunk}" for chunk in text_chunks]

        # 3. Vectorizar los chunks con prefijo
        embeddings = model.encode(prefixed_chunks, show_progress_bar=False)

        # 4. Guardar los vectores en un archivo .npy
        # .npy es un formato binario de NumPy, muy eficiente para guardar arrays numéricos
        np.save(vector_path, embeddings)

    except Exception as e:
        print(f"\n[ERROR] No se pudo procesar el archivo {txt_file}: {e}")

print("\nVectorización de todos los archivos completada.")
print(f"Los vectores se han guardado en: {VECTORS_DIR}")