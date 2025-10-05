import os
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
from flask import Flask, request, jsonify, Response, send_from_directory
from dotenv import load_dotenv
from groq import Groq

# --- 1. INICIALIZACIÓN Y CONFIGURACIÓN ---
load_dotenv() # Carga las variables de entorno desde el archivo .env

# --- Rutas dinámicas ---
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
frontend_folder = os.path.join(project_root, 'frontend')

# --- INICIO DE DEPURACIÓN ---
print("--- DEBUGGING PATHS ---")
print(f"Backend directory: {backend_dir}")
print(f"Project root: {project_root}")
print(f"Calculated frontend folder: {frontend_folder}")
print(f"Does frontend folder exist? {os.path.exists(frontend_folder)}")
print("-----------------------")
# --- FIN DE DEPURACIÓN ---

# Inicialización de Flask
app = Flask(__name__, static_url_path='', static_folder='static')

# --- 2. CARGA DE COMPONENTES (SE HACE UNA SOLA VEZ) ---
print("Cargando componentes de la IA...")

# Rutas
backend_dir = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.join(backend_dir, "data", "Processed")
INDEX_FILE = os.path.join(backend_dir, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(backend_dir, "data", "metadata.json")

# Modelos de Búsqueda (Retriever y Reranker)
retriever_model = SentenceTransformer('intfloat/multilingual-e5-large')
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

# Cliente de Groq para el LLM
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("La variable de entorno GROQ_API_KEY no está configurada.")
    llm_client = Groq(api_key=groq_api_key)
    LLM_MODEL = "llama-3.1-8b-instant" # Modelo actualizado y recomendado por Groq
    print(f"Cliente de Groq configurado con el modelo: {LLM_MODEL}")
except Exception as e:
    print(f"Error al configurar el cliente de Groq: {e}")
    llm_client = None

print("¡Servidor listo!")

# --- 3. LÓGICA DE LA IA ---

def get_text_chunk(source_file, chunk_index):
    """Obtiene un fragmento de texto de un archivo fuente dado un índice."""
    try:
        with open(os.path.join(TXT_DIR, source_file), "r", encoding="utf-8") as f:
            content = f.read()
            words = content.split()
            chunk_size = 512
            overlap = 50
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
            if chunk_index < len(chunks):
                return chunks[chunk_index]
            else:
                return ""
    except Exception:
        return ""

def retrieve_context(query, k_retriever=20, k_reranker=5, sections=None):
    """Recupera el contexto relevante para una consulta dada."""
    # 1. Búsqueda inicial
    query_vector = retriever_model.encode([f"query: {query}"])
    _, indices = index.search(query_vector, k_retriever)
    
    initial_chunks_with_meta = [
        (
            get_text_chunk(metadata[str(vector_id)]['source_file'], metadata[str(vector_id)]['chunk_index']),
            metadata[str(vector_id)]
        )
        for vector_id in indices[0] if str(vector_id) in metadata
    ]
    
    # 2. Filtrado por sección (si se especifica)
    if sections and isinstance(sections, list) and len(sections) > 0:
        initial_chunks_with_meta = [
            (chunk, meta) for chunk, meta in initial_chunks_with_meta 
            if meta.get('section', 'unknown').lower() in [s.lower() for s in sections]
        ]

    meaningful_chunks_with_meta = [(chunk, meta) for chunk, meta in initial_chunks_with_meta if len(chunk) > 150]

    if not meaningful_chunks_with_meta:
        return "", []

    # 3. Re-clasificación de los chunks
    rerank_pairs = [[query, chunk] for chunk, meta in meaningful_chunks_with_meta]
    scores = cross_encoder_model.predict(rerank_pairs)
    scored_chunks_with_meta = sorted(zip(scores, meaningful_chunks_with_meta), key=lambda x: x[0], reverse=True)

    # 4. Selección final y construcción del contexto
    final_context_chunks = []
    final_sources = set() # Usamos un set para evitar fuentes duplicadas
    CONTEXT_TOKEN_LIMIT = 7500 # Llama3 tiene un contexto más grande

    for score, (chunk, meta) in scored_chunks_with_meta[:k_reranker]:
        # La tokenización ahora es manejada por la API, solo contamos palabras como aproximación
        if len("\n\n---\n\n".join(final_context_chunks).split()) < CONTEXT_TOKEN_LIMIT:
            final_context_chunks.append(chunk)
            final_sources.add(meta['source_file'])
        else:
            break
            
    return "\n\n---\n\n".join(final_context_chunks), list(final_sources)

def parse_llm_output(full_response):
    """Separa el texto de la respuesta y el JSON del grafo."""
    text_part = full_response
    graph_data = None

    # Busca el inicio del JSON del grafo
    json_marker = full_response.find('{')
    if json_marker != -1:
        # Extrae la parte de texto y la parte que podría ser JSON
        text_part = full_response[:json_marker].strip()
        json_part = full_response[json_marker:]
        
        try:
            # Intenta parsear el JSON
            graph_data = json.loads(json_part)
        except json.JSONDecodeError:
            # Si falla, es probable que no fuera un JSON válido, se ignora
            graph_data = None
            
    return text_part, graph_data

def generate_answer_stream(query, context):
    """Genera la respuesta en modo stream usando la API de Groq."""
    if not llm_client:
        yield "token", "[ERROR: El cliente del LLM no está configurado. Revisa la API key de Groq.]"
        return

    # --- INICIO DE LA CORRECCIÓN ---
    # Si el contexto está vacío, no llamar a la API.
    if not context or not context.strip():
        yield "token", "No se encontró información relevante en los documentos para responder a esta pregunta."
        return
    # --- FIN DE LA CORRECIÓN ---

    system_prompt = """Usted es un asistente experto en biología y astronáutica. Responda la pregunta del usuario basándose únicamente en el contexto proporcionado.
Después de su respuesta, genere un objeto JSON con relaciones clave.
EJEMPLO:
Respuesta de texto aquí.
{
  "graph_data": [
    {"source": "Concepto A", "target": "Concepto B", "relationship": "afecta a"}
  ]
}
Si la información no está en el contexto, diga "La información no se encuentra en mis documentos." y no genere JSON."""

    user_prompt = f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{query}"

    # --- INICIO DE DEPURACIÓN ---
    print("\n--- DEBUG: LLAMADA A LA API DE GROQ ---")
    print(f"Modelo: {LLM_MODEL}")
    print("--- System Prompt ---")
    print(system_prompt)
    print("\n--- User Prompt (Contexto y Pregunta) ---")
    print(user_prompt)
    print("--------------------------------------\n")
    # --- FIN DE DEPURACIÓN ---

    try:
        messages_to_send = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        stream = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages_to_send,
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        full_response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                yield "token", token
        
        # Al final, procesar la respuesta completa para extraer el grafo
        _, graph_data = parse_llm_output(full_response)
        if graph_data:
            yield "graph", graph_data

    except Exception as e:
        # --- DEPURACIÓN DE ERRORES ---
        print(f"!!! ERROR DETALLADO DE LA API: {e} !!!")
        yield "token", f"[ERROR: Fallo en la llamada a la API de Groq. Detalles: {e}]"
        # --- FIN DE DEPURACIÓN DE ERRORES ---

# --- 4. RUTAS DE LA API ---

@app.route('/api/ask', methods=['POST'])
def ask():
    """Recibe una pregunta, la procesa con la IA y devuelve la respuesta en stream."""
    data = request.get_json()
    query = data.get('question')
    sections = data.get('sections')

    if not query:
        return jsonify({'error': 'No question provided'}), 400

    print(f"Recibida pregunta: {query} | Secciones: {sections}")

    def stream_response():
        context, sources = retrieve_context(query, sections=sections)
       
        if not context.strip():
            yield f"data: {json.dumps({'token': 'La información no se encuentra en mis documentos.'})}\n\n"
            yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"
            return

        for event_type, event_data in generate_answer_stream(query, context):
            yield f"data: {json.dumps({event_type: event_data})}\n\n"
      
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"

    return Response(stream_response(), mimetype='text/event-stream')

# --- RUTA PARA SERVIR EL FRONTEND ---
@app.route('/')
def root():
    return app.send_static_file('index.html')

# --- 5. PUNTO DE ENTRADA ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)