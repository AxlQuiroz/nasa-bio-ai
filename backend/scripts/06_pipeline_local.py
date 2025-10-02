import os
import faiss
from sentence_transformers import SentenceTransformer
import json
from llama_cpp import Llama

# --- Configuración del Pipeline ---
# Rutas para los datos en Inglés
TXT_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\Processed"
INDEX_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\faiss_index.bin"
METADATA_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\metadata.json"

# --- Carga de Componentes ---
print("Cargando componentes de la IA...")

# 1. Cargando el modelo de búsqueda (Retriever)
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)
print("   Componentes de búsqueda cargados.")

# 2. Cargando el modelo de lenguaje local (versión CUANTIZADA)
print("   Cargando el modelo de lenguaje local (GGUF)...")
llm = Llama(
  model_path=r"C:\Users\axelq\Documents\nasa-bio-ai\backend\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
  n_ctx=2048,  # Tamaño del contexto
  verbose=False # Añadido para reducir el texto de salida al cargar
)
print("Componentes cargados correctamente.")


def get_text_chunk(source_file, chunk_index):
    """Recupera el texto original de un chunk específico. (VERSIÓN CORREGIDA)"""
    try:
        with open(os.path.join(TXT_DIR, source_file), "r", encoding="utf-8") as f:
            content = f.read()
            
            words = content.split()
            chunk_size = 512
            overlap = 50
            
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = " ".join(words[i:i + chunk_size])
                chunks.append(chunk_text)
            
            if chunk_index < len(chunks):
                return chunks[chunk_index]
            else:
                return "[ERROR: Índice de chunk fuera de rango]"

    except Exception as e:
        return f"[ERROR al recuperar el chunk: {e}]"

def retrieve_context(query, k=5):
    """
    Paso de Búsqueda (Retrieval): Encuentra los chunks más relevantes y se asegura
    de que el contexto combinado no exceda el límite de tokens.
    """
    query_vector = retriever_model.encode([query])
    
    # 1. Realizar la búsqueda para obtener los k chunks más relevantes
    _, indices = index.search(query_vector, k)
    
    # 2. Recuperar el texto de todos los chunks encontrados
    all_chunks = [
        get_text_chunk(metadata[vector_id]['source_file'], metadata[vector_id]['chunk_index'])
        for vector_id in indices[0] if vector_id != -1
    ]
    
    # 3. Construir el contexto de forma segura, respetando el límite de tokens
    final_context = []
    total_tokens = 0
    # Límite de seguridad para dejar espacio para el prompt y la respuesta
    CONTEXT_TOKEN_LIMIT = 1500 

    for chunk in all_chunks:
        # Tokenizar el chunk para saber su tamaño real
        # El método .tokenize() de llama_cpp devuelve una lista de tokens
        chunk_tokens = llm.tokenize(chunk.encode("utf-8"))
        
        if total_tokens + len(chunk_tokens) <= CONTEXT_TOKEN_LIMIT:
            final_context.append(chunk)
            total_tokens += len(chunk_tokens)
        else:
            # Si añadir el siguiente chunk excede el límite, nos detenemos
            break
            
    print(f"   Contexto construido con {len(final_context)} chunks y ~{total_tokens} tokens.")
    return "\n\n---\n\n".join(final_context)

def generate_answer(query, context):
    """Paso de Generación (Generation): Usa el modelo local GGUF para responder."""
    prompt = f"""<|system|>
You are an expert assistant in biology and astronautics. Answer the question based ONLY on the provided context. If the information is not in the context, say "The information is not in my documents." Do not invent anything.</s>
<|user|>
CONTEXT:
{context}

QUESTION:
{query}</s>
<|assistant|>
"""
    
    print("   Generando respuesta con el modelo local (esto puede ser lento)...")
    
    output = llm(
        prompt,
        max_tokens=256,
        stop=["</s>", "<|user|>"],
        echo=False
    )
    
    if output and 'choices' in output and len(output['choices']) > 0:
        return output['choices'][0]['text'].strip()
    
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