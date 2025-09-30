import os
import faiss
from sentence_transformers import SentenceTransformer
import json
import torch
from transformers import pipeline

# --- Configuración del Pipeline ---
# Rutas para los datos en Inglés
TXT_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\Processed"
INDEX_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\faiss_index.bin"
METADATA_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\metadata.json"

# --- Carga de Componentes (Búsqueda y Generación) ---
print("Cargando componentes de la IA...")

# 1. Cargando el modelo de búsqueda (Retriever)
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)
print("   Componentes de búsqueda cargados.")

# 2. Cargando el modelo de lenguaje local (Generator)
# La primera vez que se ejecute, descargará el modelo (varios GB).
print("   Cargando el modelo de lenguaje local (puede tardar varios minutos)...")
generator_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16, # Usar menos memoria
    device_map="auto" # Usará CPU si no hay GPU
)
print("Componentes cargados correctamente.")


def get_text_chunk(source_file, chunk_index):
    """Recupera el texto original de un chunk específico."""
    try:
        with open(os.path.join(TXT_DIR, source_file), "r", encoding="utf-8") as f:
            content = f.read()
            words = content.split()
            chunk_size = 512
            overlap = 50
            start_index = chunk_index * (chunk_size - overlap)
            end_index = start_index + chunk_size
            return " ".join(words[start_index:end_index])
    except Exception:
        return ""

def retrieve_context(query, k=5):
    """Paso de Búsqueda (Retrieval): Encuentra los chunks más relevantes."""
    query_vector = retriever_model.encode([query])
    _, indices = index.search(query_vector, k)
    
    retrieved_chunks = [
        get_text_chunk(metadata[vector_id]['source_file'], metadata[vector_id]['chunk_index'])
        for vector_id in indices[0] if vector_id != -1
    ]
    return "\n\n---\n\n".join(retrieved_chunks)

def generate_answer(query, context):
    """Paso de Generación (Generation): Usa el modelo local para responder."""
    
    # Un "prompt template" específico para modelos de chat como TinyLlama
    prompt = f"""
    <|system|>
    You are an expert assistant in biology and astronautics. Answer the question based ONLY on the provided context. If the information is not in the context, say "The information is not in my documents." Do not invent anything.</s>
    <|user|>
    CONTEXT:
    {context}

    QUESTION:
    {query}</s>
    <|assistant|>
    """
    
    print("   Generando respuesta con el modelo local (esto puede ser lento)...")
    
    # Usar el pipeline para generar la respuesta
    sequences = generator_pipeline(
        prompt,
        max_new_tokens=256,  # Limitar la longitud de la respuesta
        do_sample=True,
        temperature=0.1,
        top_p=0.95
    )
    
    # Extraer solo el texto generado por el asistente
    if sequences and len(sequences) > 0:
        full_text = sequences[0]['generated_text']
        # Encontrar el inicio de la respuesta del asistente y devolver solo esa parte
        assistant_response_start = full_text.find("<|assistant|>")
        if assistant_response_start != -1:
            return full_text[assistant_response_start + len("<|assistant|>"):].strip()
    
    return "[ERROR: No se pudo generar una respuesta]"

# --- Ejecución del Pipeline Completo ---
if __name__ == "__main__":
    pregunta_usuario = "What are the effects of microgravity on human bones?"
    
    print(f"\nPregunta del usuario: {pregunta_usuario}")
    print("\n1. Recuperando contexto relevante de los documentos...")
    
    contexto = retrieve_context(pregunta_usuario)
    print("   Contexto recuperado.")
    
    print("\n2. Generando respuesta con el LLM local...")
    
    respuesta_final = generate_answer(pregunta_usuario, contexto)
    
    print("\n--- RESPUESTA DE LA IA ---")
    print(respuesta_final)