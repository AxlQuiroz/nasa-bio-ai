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
static_folder = os.path.join(backend_dir, 'static')
static2_folder = os.path.join(backend_dir, 'static2')

app = Flask(__name__)

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

def retrieve_context(query, top_k=20, rerank_top_n=5):
    """
    Recupera el contexto relevante de la base de datos de vectores sin filtrar por sección.
    """
    print(f"Recuperando contexto para la consulta: {query}")
    
    # 1. Búsqueda inicial en FAISS
    query_embedding = retriever_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = []
    for i, doc_id in enumerate(indices[0]):
        if doc_id != -1:
            original_id = id_map.at(doc_id)
            metadata = metadata_dict.get(str(original_id))
            if metadata:
                retrieved_docs.append({
                    "id": original_id,
                    "text": metadata['text'],
                    "score": distances[0][i]
                })

    if not retrieved_docs:
        return "", []

    # 2. Re-ranking con Cross-Encoder
    cross_inp = [[query, doc['text']] for doc in retrieved_docs]
    cross_scores = cross_encoder_model.predict(cross_inp)
    
    for i in range(len(cross_scores)):
        retrieved_docs[i]['cross_score'] = cross_scores[i]

    reranked_docs = sorted(retrieved_docs, key=lambda x: x['cross_score'], reverse=True)
    
    # 3. Construir el contexto final
    final_context = "\n\n---\n\n".join([doc['text'] for doc in reranked_docs[:rerank_top_n]])
    context_metadata = [
        {"id": doc['id'], "text": doc['text']} for doc in reranked_docs[:rerank_top_n]
    ]
    
    print(f"Contexto final construido con {len(context_metadata)} fragmentos.")
    return final_context, context_metadata

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

def generate_answer_stream(query, context, context_metadata, analysis_type='default'):
    """
    Genera una respuesta usando el LLM con un prompt dinámico según el tipo de análisis.
    """
    
    # Plantillas de prompts para cada tipo de análisis
    PROMPT_TEMPLATES = {
        "default": (
            "Eres un asistente de IA experto en biociencia. Tu tarea es responder la pregunta del usuario de forma clara y concisa, basándote únicamente en el contexto proporcionado. "
            "Pregunta del usuario: {query}"
        ),
        "progress_areas": (
            "Eres un analista de investigación experto en biociencia. Tu tarea es analizar el contexto proporcionado para identificar y resumir las 'áreas de progreso' y los 'avances más significativos' relacionados con la pregunta del usuario. "
            "No respondas directamente a la pregunta, en su lugar, extrae y presenta los avances clave. "
            "Pregunta del usuario: {query}"
        ),
        "knowledge_gaps": (
            "Eres un estratega de investigación experto en biociencia. Tu tarea es analizar el contexto proporcionado para identificar y explicar las 'lagunas de conocimiento', las 'preguntas sin resolver' y las 'áreas que requieren más investigación' relacionadas con la pregunta del usuario. "
            "Cita frases que sugieran incertidumbre o necesidad de futuros estudios. "
            "Pregunta del usuario: {query}"
        ),
        "consensus_disagreement": (
            "Eres un revisor científico experto. Tu tarea es analizar el contexto, que puede provenir de múltiples fuentes, para encontrar 'áreas de consenso' (donde los autores coinciden) y 'áreas de desacuerdo' (donde hay controversia o hallazgos contradictorios) sobre la pregunta del usuario. "
            "Estructura tu respuesta separando claramente el consenso del desacuerdo. "
            "Pregunta del usuario: {query}"
        )
    }

    # Seleccionar la plantilla de prompt o usar la por defecto
    system_prompt_template = PROMPT_TEMPLATES.get(analysis_type, PROMPT_TEMPLATES['default'])
    system_prompt = system_prompt_template.format(query=query)

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
    query = request.json.get('query')
    # Obtenemos el nuevo tipo de análisis. 'default' si no se especifica.
    analysis_type = request.json.get('analysis_type', 'default')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Ya no pasamos 'sections' a retrieve_context
        context, context_metadata = retrieve_context(query)
        
        if not context:
            # Si no hay contexto, generamos una respuesta sin él
            return Response(generate_answer_stream(query, "No se encontró contexto relevante.", [], analysis_type), mimetype='application/json')

        # Pasamos el 'analysis_type' a la función de generación
        return Response(generate_answer_stream(query, context, context_metadata, analysis_type), mimetype='application/json')

    except Exception as e:
        print(f"Error en /api/ask: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred"}), 500

# --- RUTAS PARA SERVIR EL FRONTEND ---

# 1. Ruta para la página de bienvenida (landing page)
@app.route('/')
def landing_page():
    # Sirve el index.html desde la carpeta 'static2'
    return send_from_directory(static2_folder, 'index.html')

# 2. Ruta para la página del chat con la IA
@app.route('/chat')
def chat_page():
    # Sirve el chatbot.html directamente desde la carpeta 'static'
    return send_from_directory(static_folder, 'chatbot.html')

# 3. Rutas para servir los archivos de cada página (CSS, JS, etc.)
@app.route('/static/<path:filename>')
def serve_chat_assets(filename):
    # Sirve los archivos necesarios para la página de chat
    return send_from_directory(static_folder, filename)

@app.route('/static2/<path:filename>')
def serve_landing_assets(filename):
    # Sirve los archivos necesarios para la página de bienvenida
    return send_from_directory(static2_folder, filename)

# --- 5. PUNTO DE ENTRADA ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)