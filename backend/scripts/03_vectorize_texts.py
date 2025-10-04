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
print("Cargando el modelo de SentenceTransformer (multilingüe-e5-base)...")
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
    
   
    vector_filename = os.path.splitext(txt_file)[0] + ".npy"
    vector_path = os.path.join(VECTORS_DIR, vector_filename)

  
    if os.path.exists(vector_path):
        continue

    try:
      
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Dividir el texto en chunks
        text_chunks = chunk_text(content)
        
        if not text_chunks:
            print(f"\n[ADVERTENCIA] El archivo {txt_file} está vacío o no generó chunks.")
            continue

        # 2. Vectorizar los chunks en lotes para no agotar la memoria
        batch_size = 64
        all_embeddings = []
        
        # Añadimos una barra de progreso para los lotes dentro de un archivo
        for i in tqdm(range(0, len(text_chunks), batch_size), desc=f"  -> Chunks en {txt_file}", leave=False):
            batch_chunks = text_chunks[i:i + batch_size]
            prefixed_batch = [f"passage: {chunk}" for chunk in batch_chunks]
            
            batch_embeddings = model.encode(prefixed_batch, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            continue

        # 3. Unir los embeddings de todos los lotes
        embeddings = np.vstack(all_embeddings)

        # 4. Guardar los vectores
        np.save(vector_path, embeddings)

    except Exception as e:
        print(f"\n[ERROR] No se pudo procesar el archivo {txt_file}: {e}")

print("\nVectorización de todos los archivos completada.")
print(f"Los vectores se han guardado en: {VECTORS_DIR}")