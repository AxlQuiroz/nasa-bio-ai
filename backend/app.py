import os
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
from llama_cpp import Llama
from flask import Flask, request, jsonify, render_template, Response
import time 

# --- 1. INICIALIZACIÓN DE LA APLICACIÓN FLASK ---
app = Flask(__name__, static_folder='static', static_url_path='')

# --- 2. CARGA DE MODELOS (SE HACE UNA SOLA VEZ AL INICIAR EL SERVIDOR) ---
print("Cargando componentes de la IA... Esto puede tardar varios minutos.")

# Rutas a los datos (usando rutas relativas para portabilidad)
backend_dir = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.join(backend_dir, "data", "Processed")
INDEX_FILE = os.path.join(backend_dir, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(backend_dir, "data", "metadata.json")
MODEL_PATH = os.path.join(backend_dir, "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# Carga de modelos
retriever_model = SentenceTransformer('intfloat/multilingual-e5-large')
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

print("¡Servidor listo! Los modelos de IA se han cargado.")

# --- 3. LÓGICA DE LA IA

def get_text_chunk(source_file, chunk_index):
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
    # 1. Búsqueda inicial
    query_vector = retriever_model.encode([query])
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
        filtered_by_section = []
        for chunk, meta in initial_chunks_with_meta:
            # Asumimos que los metadatos tienen una clave 'section'
            if meta.get('section', 'unknown').lower() in [s.lower() for s in sections]:
                filtered_by_section.append((chunk, meta))
        initial_chunks_with_meta = filtered_by_section

    meaningful_chunks_with_meta = [
        (chunk, meta) for chunk, meta in initial_chunks_with_meta if len(chunk) > 150
    ]

    if not meaningful_chunks_with_meta:
        return "", []

    # 3. Re-clasificación de los chunks
    rerank_pairs = [[query, chunk] for chunk, meta in meaningful_chunks_with_meta]
    scores = cross_encoder_model.predict(rerank_pairs)

    scored_chunks_with_meta = sorted(zip(scores, meaningful_chunks_with_meta), key=lambda x: x[0], reverse=True)

    # 4. Selección final y construcción del contexto
    final_context_chunks = []
    final_sources = set() # Usamos un set para evitar fuentes duplicadas
    total_tokens = 0
    CONTEXT_TOKEN_LIMIT = 1800 

    for score, (chunk, meta) in scored_chunks_with_meta[:k_reranker]:
        chunk_tokens = llm.tokenize(chunk.encode("utf-8", errors="ignore"))
        if total_tokens + len(chunk_tokens) <= CONTEXT_TOKEN_LIMIT:
            final_context_chunks.append(chunk)
            final_sources.add(meta['source_file']) # Añadimos el nombre del archivo
            total_tokens += len(chunk_tokens)
        else:
            break
            
    context_string = "\n\n---\n\n".join(final_context_chunks)
    return context_string, list(final_sources)

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
    """Genera la respuesta en modo stream y extrae el grafo al final."""
    prompt = f"""<|system|>
Usted es un asistente experto en biología y astronáutica. Responda la pregunta del usuario basándose únicamente en el contexto.
Después de su respuesta, genere un objeto JSON con relaciones clave.
EJEMPLO:
Respuesta de texto aquí.
{{
  "graph_data": [
    {{"source": "Concepto A", "target": "Concepto B", "relationship": "afecta a"}}
  ]
}}
Si la información no está en el contexto, diga "La información no se encuentra en mis documentos." y no genere JSON.</s>
<|user|>
CONTEXTO:
{context}

PREGUNTA:
{query}</s>
<assistant|>
"""
   
    stream = llm(prompt, max_tokens=512, stop=["</s>", "<|user|>"], echo=False, temperature=0.1, stream=True)
    
    full_response = ""
    for output in stream:
        if 'choices' in output and len(output['choices']) > 0:
            token = output['choices'][0].get('text', '')
            if token:
                full_response += token
                yield "token", token
    
    # Una vez terminado el stream, procesamos la respuesta completa
    _, graph_data = parse_llm_output(full_response)

    # Enviamos el grafo como un evento separado
    if graph_data:
        yield "graph", graph_data

# --- 4. DEFINICIÓN DE LAS RUTAS DE LA API ---

@app.route('/')
def home():
    """Sirve la página principal HTML desde la carpeta static."""
    return app.send_static_file('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Recibe una pregunta, la procesa con la IA y devuelve la respuesta en stream."""
    data = request.get_json()
    query = data.get('question')
    sections = data.get('sections') # Nuevo: obtener las secciones del request

    if not query:
        return jsonify({'error': 'No question provided'}), 400

    print(f"Recibida pregunta para streaming: {query} | Secciones: {sections}")

    def stream_response():
        context, sources = retrieve_context(query, sections=sections)
       
        if not context.strip():
            # Envía una respuesta de texto y finaliza
            yield f"data: {json.dumps({'token': 'La información no se encuentra en mis documentos.'})}\n\n"
            yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"
            return

        # Stream de los tokens de la respuesta y el grafo al final
        for event_type, event_data in generate_answer_stream(query, context):
            yield f"data: {json.dumps({event_type: event_data})}\n\n"
      
        # Una vez terminada la respuesta, enviar las fuentes
        yield f"data: {json.dumps({'sources': sources})}\n\n"

        # Señal de finalización
        yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"


    return Response(stream_response(), mimetype='text/event-stream')

# --- 5. PUNTO DE ENTRADA PARA EJECUTAR EL SERVIDOR ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)