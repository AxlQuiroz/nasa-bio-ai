import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

# --- Configuración de la Prueba en Inglés ---
LANGUAGE_TO_TEST = "Español"

# --- Rutas dinámicas ---
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
TXT_DIR = os.path.join(backend_dir, "data", "Processed")
INDEX_FILE = os.path.join(backend_dir, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(backend_dir, "data", "metadata.json")
MODEL_PATH = os.path.join(backend_dir, "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# --- Carga del Modelo (debe ser el mismo que usaste para vectorizar) ---
print("Cargando el modelo de SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Modelo cargado.")

# --- Carga del Índice y Metadatos ---
print(f"Cargando índice desde {INDEX_FILE}")
index = faiss.read_index(INDEX_FILE)
print(f"Cargando metadatos desde {METADATA_FILE}")
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

def get_text_chunk(source_file, chunk_index):
    """Recupera el texto original de un chunk específico."""
    try:
        with open(os.path.join(TXT_DIR, source_file), "r", encoding="utf-8") as f:
            content = f.read()
            # Esta lógica debe coincidir con la del script de vectorización
            words = content.split()
            chunk_size = 512
            overlap = 50
            start_index = chunk_index * (chunk_size - overlap)
            end_index = start_index + chunk_size
            chunk_text = " ".join(words[start_index:end_index])
            return chunk_text
    except Exception as e:
        return f"[ERROR al recuperar el chunk: {e}]"

def search(query, k=3):
    """
    Toma una pregunta, la vectoriza y busca los k chunks más relevantes.
    """
    print("-" * 50)
    print(f"Buscando en {LANGUAGE_TO_TEST} los {k} resultados más relevantes para: '{query}'")
    
    # 1. Vectorizar la pregunta
    query_vector = model.encode([query])
    
    # 2. Realizar la búsqueda en el índice FAISS
    distances, indices = index.search(query_vector, k)
    
    print("\n--- Resultados Encontrados ---")
    results = []
    for i, vector_id in enumerate(indices[0]):
        if vector_id == -1:
            print(f"  {i+1}. No se encontraron más resultados.")
            continue
            
        # 3. Usar el ID para encontrar los metadatos
        meta = metadata[vector_id]
        source_file = meta['source_file']
        chunk_idx = meta['chunk_index']
        
        # 4. Recuperar el texto original del chunk
        retrieved_text = get_text_chunk(source_file, chunk_idx)
        results.append(retrieved_text)
        
        print(f"\n  {i+1}. Relevancia: {distances[0][i]:.4f} (Fuente: {source_file}, Chunk: {chunk_idx})")
        print(f"     Texto: '{retrieved_text[:300].strip()}...'")
        
    return results

def generate_answer(query, context):
    """Genera la respuesta a partir del contexto."""
    prompt = f"""<|system|>
Eres un asistente experto en biología y astronáutica. Responde la pregunta del usuario usando únicamente el contexto proporcionado. La respuesta DEBE SER EN ESPAÑOL. Si la información no está en el contexto, responde exactamente: "La información no se encuentra en mis documentos." No inventes nada.</s>
<|user|>
CONTEXTO:
{context}

PREGUNTA:
{query}</s>
<|assistant|>
Respuesta en español:
"""
    # Aquí iría el código para enviar el 'prompt' a un modelo de lenguaje y obtener la respuesta.
    # Por ahora, solo devolveremos el prompt para propósitos de depuración.
    return prompt

# --- Ejecución de la Prueba ---
if __name__ == "__main__":
    # --- PREGUNTA DE PRUEBA ---
    # Cambiamos la pregunta a español para probar el modelo multilingüe
    query = "¿Qué efectos tiene la microgravedad en los huesos?"

    # Ejecuta la búsqueda con la pregunta
    contexto_recuperado = search(query)

    # Generar la respuesta a partir del contexto recuperado
    respuesta = generate_answer(query, " ".join(contexto_recuperado))

    # Este 'contexto_recuperado' es lo que le pasarías a un modelo como GPT para que genere la respuesta final.
    print("\n" + "-" * 50)
    print("Contexto recuperado listo para ser enviado a un LLM:")
    print(contexto_recuperado)
    print("\nRespuesta generada:")
    print(respuesta)