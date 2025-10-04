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

# --- Carga del Modelo (el mismo que usaste para vectorizar) ---
print("Cargando el modelo de SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Modelo cargado.")

def build_index():
    """
    Lee todos los vectores .npy, los combina y construye un índice FAISS.
    También crea un archivo de metadatos para saber a qué texto corresponde cada vector.
    """
    print("Construyendo el índice FAISS desde los archivos de vectores...")
    
    vector_files = [f for f in os.listdir(VECTORS_DIR) if f.endswith(".npy")]
    all_embeddings = []
    metadata = [] # Lista para guardar la información de origen de cada vector
    
    vector_id_counter = 0
    for vector_file in tqdm(vector_files, desc="Cargando vectores"):
        vector_path = os.path.join(VECTORS_DIR, vector_file)
        embeddings = np.load(vector_path)
        
        # Guardar metadatos: a qué archivo y chunk pertenece cada vector
        base_filename = os.path.splitext(vector_file)[0]
        for i in range(embeddings.shape[0]):
            metadata.append({
                "id": vector_id_counter,
                "source_file": base_filename + ".txt",
                "chunk_index": i
            })
            vector_id_counter += 1
            
        all_embeddings.append(embeddings)

    # Concatenar todos los embeddings en una sola matriz de NumPy
    final_embeddings = np.vstack(all_embeddings)
    
    # La dimensión de nuestros vectores (el modelo all-MiniLM-L6-v2 produce vectores de 384 dimensiones)
    d = final_embeddings.shape[1]
    
    # Construir el índice FAISS
    index = faiss.IndexFlatL2(d)  # Usamos la distancia L2 (euclidiana)
    index = faiss.IndexIDMap(index) # Permite mapear vectores a sus IDs originales
    
    # Añadir los vectores al índice con sus IDs
    index.add_with_ids(final_embeddings, np.array(range(vector_id_counter)))
    
    # Guardar el índice en el disco
    faiss.write_index(index, INDEX_FILE)
    
    # Guardar los metadatos
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)
        
    print(f"Índice construido y guardado en {INDEX_FILE}")
    print(f"Metadatos guardados en {METADATA_FILE}")

def search(query, k=5):
    """
    Toma una pregunta, la vectoriza y busca los k chunks más relevantes en el índice FAISS.
    """
    print(f"\nBuscando los {k} resultados más relevantes para: '{query}'")
    
    # Cargar el índice y los metadatos
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)

    # Vectorizar la pregunta del usuario
    query_vector = model.encode([query])
    
    # Realizar la búsqueda en el índice
    # D son las distancias, I son los IDs de los vectores encontrados
    distances, indices = index.search(query_vector, k)
    
    print("Resultados encontrados:")
    results = []
    for i, idx in enumerate(indices[0]):
        # Usar el ID del vector para encontrar sus metadatos
        meta = metadata[idx]
        source_file = meta['source_file']
        
        # Opcional: Cargar y mostrar el chunk de texto original
        try:
            with open(os.path.join(TXT_DIR, source_file), "r", encoding="utf-8") as f:
                # Esta es una forma simple de recuperar el chunk, se puede optimizar
                content = f.read()
                # Re-creamos los chunks para encontrar el que corresponde
                from_chunking_script = " ".join(content.split()[meta['chunk_index'] * (512-50) : meta['chunk_index'] * (512-50) + 512])

                print(f"  {i+1}. Distancia: {distances[0][i]:.4f} (Fuente: {source_file}, Chunk: {meta['chunk_index']})")
                print(f"     Texto: '{from_chunking_script[:200]}...'")
                results.append(from_chunking_script)
        except Exception as e:
            print(f"Error al recuperar el texto del chunk: {e}")
            
    return results

# --- Ejecución ---
if __name__ == "__main__":
    # Paso 1: Construir el índice. Solo necesitas ejecutar esto una vez.
    # Si el índice ya existe, puedes comentar esta línea.
    if not os.path.exists(INDEX_FILE):
        build_index()
    
    # Paso 2: Hacer preguntas al sistema.
    # Puedes ejecutar esta parte cuantas veces quieras.
    retrieved_context = search("What are the effects of microgravity on human bones?")
    
    # El siguiente paso sería tomar 'retrieved_context' y pasarlo a un LLM.
    print("\n--- Contexto recuperado para enviar a un LLM ---")
    print("\n\n".join(retrieved_context))