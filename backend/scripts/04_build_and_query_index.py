import os
import numpy as np
import faiss
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- Rutas dinámicas basadas en la ubicación del script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir) # Sube un nivel para llegar a la raíz del backend
VECTORS_DIR = os.path.join(backend_dir, "data", "Vectorized")
TXT_DIR = os.path.join(backend_dir, "data", "Processed")
INDEX_FILE = os.path.join(backend_dir, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(backend_dir, "data", "metadata.json")

# --- Carga del Modelo (debe ser el mismo que se usó para vectorizar) ---
print("Cargando el modelo de SentenceTransformer (multilingual-e5-large)...")
model = SentenceTransformer('intfloat/multilingual-e5-large')
print("Modelo cargado.")

def build_index():
    """
    Lee todos los vectores .npy y sus metadatos temporales, los combina, 
    construye un índice FAISS y crea un archivo de metadatos global.
    """
    print("Construyendo el índice FAISS desde los archivos de vectores...")
    
    vector_files = [f for f in os.listdir(VECTORS_DIR) if f.endswith(".npy")]
    all_embeddings = []
    metadata = {} # Usaremos un diccionario para un acceso más rápido
    
    vector_id_counter = 0
    for vector_file in tqdm(vector_files, desc="Cargando y procesando vectores"):
        base_filename = os.path.splitext(vector_file)[0]
        vector_path = os.path.join(VECTORS_DIR, vector_file)
        meta_path = os.path.join(VECTORS_DIR, base_filename + "_meta.json")

        if not os.path.exists(meta_path):
            print(f"\n[ADVERTENCIA] No se encontró el archivo de metadatos {meta_path} para {vector_file}. Omitiendo.")
            continue

        embeddings = np.load(vector_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            chunk_metadata = json.load(f)

        if embeddings.shape[0] != len(chunk_metadata):
            print(f"\n[ADVERTENCIA] Discrepancia en {vector_file}: {embeddings.shape[0]} vectores vs {len(chunk_metadata)} metadatos. Omitiendo.")
            continue
            
        # Guardar metadatos enriquecidos
        for i in range(embeddings.shape[0]):
            meta_item = chunk_metadata[i]
            metadata[str(vector_id_counter)] = {
                "source_file": base_filename + ".txt",
                "chunk_index": meta_item.get("chunk_index", -1),
                "section": meta_item.get("section", "unknown")
            }
            vector_id_counter += 1
            
        all_embeddings.append(embeddings)
        
        # Eliminar el archivo de metadatos temporal después de procesarlo
        os.remove(meta_path)

    if not all_embeddings:
        print("No se encontraron vectores para procesar. Abortando.")
        return

    # Concatenar todos los embeddings en una sola matriz de NumPy
    final_embeddings = np.vstack(all_embeddings)
    
    # La dimensión de nuestros vectores (multilingual-e5-large produce 1024)
    d = final_embeddings.shape[1]
    
    # Construir el índice FAISS
    index = faiss.IndexFlatL2(d)
    index = faiss.IndexIDMap(index)
    
    # Añadir los vectores al índice con sus IDs
    ids = np.array([int(k) for k in metadata.keys()])
    index.add_with_ids(final_embeddings, ids)
    
    # Guardar el índice en el disco
    faiss.write_index(index, INDEX_FILE)
    
    # Guardar los metadatos
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Índice construido y guardado en {INDEX_FILE}")
    print(f"Metadatos guardados en {METADATA_FILE}")
    print("Archivos de metadatos temporales eliminados.")

def search(query, k=5):
    """
    Toma una pregunta, la vectoriza y busca los k chunks más relevantes en el índice FAISS.
    """
    print(f"\nBuscando los {k} resultados más relevantes para: '{query}'")
    
    # Cargar el índice y los metadatos
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    # Vectorizar la pregunta del usuario con el prefijo correcto
    query_vector = model.encode([f"query: {query}"])
    
    # Realizar la búsqueda en el índice
    distances, indices = index.search(query_vector, k)
    
    print("Resultados encontrados:")
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue # Omitir resultados inválidos
        
        # Usar el ID del vector para encontrar sus metadatos en el diccionario
        meta = metadata.get(str(idx))
        if not meta:
            print(f"  No se encontraron metadatos para el ID {idx}")
            continue

        source_file = meta['source_file']
        section = meta['section']
        
        # Opcional: Cargar y mostrar el chunk de texto original
        try:
            with open(os.path.join(TXT_DIR, source_file), "r", encoding="utf-8") as f:
                content = f.read()
                # Re-creamos los chunks para encontrar el que corresponde
                # NOTA: Esta lógica debe coincidir con la del script 03
                words = content.split()
                chunk_size = 512
                overlap = 50
                start_index = meta['chunk_index'] * (chunk_size - overlap)
                end_index = start_index + chunk_size
                original_chunk = " ".join(words[start_index:end_index])

                print(f"  {i+1}. Distancia: {distances[0][i]:.4f} (Fuente: {source_file}, Chunk: {meta['chunk_index']}, Sección: {section})")
                print(f"     Texto: '{original_chunk[:200]}...'")
                results.append(original_chunk)
        except Exception as e:
            print(f"Error al recuperar el texto del chunk: {e}")
            
    return results

# --- Ejecución ---
if __name__ == "__main__":
    # Paso 1: Construir el índice. Solo necesitas ejecutar esto una vez.
    # Si el índice ya existe, puedes comentar esta línea para solo hacer búsquedas.
    build_index()
    
    # Paso 2: Hacer preguntas al sistema.
    # Puedes ejecutar esta parte cuantas veces quieras.
    retrieved_context = search("What are the effects of microgravity on human bones?")
    
    # El siguiente paso sería tomar 'retrieved_context' y pasarlo a un LLM.
    print("\n--- Contexto recuperado para enviar a un LLM ---")
    print("\n\n".join(retrieved_context))