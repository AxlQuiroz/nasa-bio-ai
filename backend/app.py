print("--- INICIO DEL SCRIPT APP.PY ---")

import os
import faiss
import json
import traceback
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
load_dotenv()

# Rutas dinámicas
backend_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(backend_dir, 'static')
static2_folder = os.path.join(backend_dir, 'static2')
INDEX_FILE = os.path.join(backend_dir, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(backend_dir, "data", "metadata.json")

# --- 2. INICIALIZACIÓN DE FLASK ---
app = Flask(__name__)
CORS(app)

# --- 3. CARGA DE MODELOS Y DATOS (SE HACE UNA SOLA VEZ AL INICIO) ---
print("Cargando componentes de la IA (esto puede tardar)...")
try:
    # Modelos de Búsqueda (Retriever y Reranker)
    retriever_model = SentenceTransformer('intfloat/multilingual-e5-large')
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Carga de datos y del índice FAISS
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)

    # Cliente de Groq para el LLM
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("La variable de entorno GROQ_API_KEY no está configurada.")
    llm_client = Groq(api_key=groq_api_key)
    LLM_MODEL = "llama-3.1-8b-instant"
    
    print(f"¡Componentes cargados! Servidor listo con el modelo: {LLM_MODEL}")

except Exception as e:
    print("\n\n!!!!!!!!!! ERROR CRÍTICO DURANTE LA INICIALIZACIÓN !!!!!!!!!!")
    print(f"No se pudieron cargar los modelos o archivos necesarios. Error: {e}")
    traceback.print_exc()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    # Salimos si no se pueden cargar los componentes esenciales
    exit()


# --- 4. LÓGICA DE LA IA (FUNCIONES AUXILIARES) ---

def retrieve_context(query, top_k=20, rerank_top_n=5):
    """
    Recupera el contexto relevante de la base de datos de vectores.
    """
    print(f"Recuperando contexto para la consulta: {query}")
    
    query_embedding = retriever_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = []
    for i, doc_id in enumerate(indices[0]):
        if doc_id != -1:
            text_chunk = metadata_dict.get(str(doc_id))
            # --- INICIO DE LA CORRECCIÓN ---
            # Nos aseguramos de que text_chunk no sea None ni una cadena vacía
            if text_chunk and isinstance(text_chunk, str):
            # --- FIN DE LA CORRECCIÓN ---
                retrieved_docs.append({
                    "id": doc_id,
                    "text": text_chunk,
                    "score": distances[0][i]
                })

    if not retrieved_docs:
        return "", []

    cross_inp = [[query, doc['text']] for doc in retrieved_docs]
    cross_scores = cross_encoder_model.predict(cross_inp)
    
    for i in range(len(cross_scores)):
        retrieved_docs[i]['cross_score'] = cross_scores[i]

    reranked_docs = sorted(retrieved_docs, key=lambda x: x['cross_score'], reverse=True)
    
    final_context = "\n\n---\n\n".join([doc['text'] for doc in reranked_docs[:rerank_top_n]])
    context_metadata = [{"id": doc['id'], "text": doc['text']} for doc in reranked_docs[:rerank_top_n]]
    
    print(f"Contexto final construido con {len(context_metadata)} fragmentos.")
    return final_context, context_metadata

def parse_llm_output(full_response):
    """Separa el texto de la respuesta y el JSON del grafo."""
    text_part = full_response
    graph_data = None
    json_marker = full_response.find('{')
    if json_marker != -1:
        text_part = full_response[:json_marker].strip()
        json_part = full_response[json_marker:]
        try:
            graph_data = json.loads(json_part)
        except json.JSONDecodeError:
            graph_data = None
    return text_part, graph_data

def generate_answer_stream(query, context, context_metadata, analysis_type='default'):
    """
    Genera una respuesta usando el LLM con un prompt dinámico según el tipo de análisis.
    """
    PROMPT_TEMPLATES = {
        "default": (
            "Eres un asistente de IA experto en biociencia. Tu tarea es responder la pregunta del usuario de forma clara y concisa, basándote únicamente en el contexto proporcionado. "
            "Pregunta del usuario: {query}"
        ),
        "progress_areas": (
            "Eres un analista de investigación experto. Tu tarea es analizar el contexto para identificar y resumir las 'áreas de progreso' y los 'avances más significativos' relacionados con la pregunta del usuario. "
            "No respondas directamente a la pregunta, en su lugar, extrae y presenta los avances clave. Pregunta del usuario: {query}"
        ),
        "knowledge_gaps": (
            "Eres un estratega de investigación experto. Tu tarea es analizar el contexto para identificar y explicar las 'lagunas de conocimiento' y las 'áreas que requieren más investigación' relacionadas con la pregunta del usuario. "
            "Cita frases que sugieran incertidumbre o necesidad de futuros estudios. Pregunta del usuario: {query}"
        ),
        "consensus_disagreement": (
            "Eres un revisor científico experto. Tu tarea es analizar el contexto para encontrar 'áreas de consenso' (donde los autores coinciden) y 'áreas de desacuerdo' (donde hay controversia) sobre la pregunta del usuario. "
            "Estructura tu respuesta separando claramente el consenso del desacuerdo. Pregunta del usuario: {query}"
        )
    }

    system_prompt_template = PROMPT_TEMPLATES.get(analysis_type, PROMPT_TEMPLATES['default'])
    system_prompt = system_prompt_template.format(query=query)
    user_prompt = f"CONTEXTO:\n{context}"

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
                yield json.dumps({"token": token}) + "\n"
        
        text_part, graph_data = parse_llm_output(full_response)
        if graph_data:
            yield json.dumps({"graph": graph_data}) + "\n"
        
        yield json.dumps({"sources": context_metadata}) + "\n"
        yield json.dumps({"token": "[DONE]"}) + "\n"

    except Exception as e:
        print(f"!!! ERROR DETALLADO DE LA API: {e} !!!")
        yield json.dumps({"token": f"[ERROR: Fallo en la llamada a la API. Detalles: {e}]"}) + "\n"
        yield json.dumps({"token": "[DONE]"}) + "\n"


# --- 5. RUTAS DE LA APLICACIÓN (ENDPOINTS) ---

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')
    analysis_type = data.get('analysis_type', 'default')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        context, context_metadata = retrieve_context(query)
        
        if not context:
            context = "No se encontró contexto relevante."
            context_metadata = []

        return Response(generate_answer_stream(query, context, context_metadata, analysis_type), mimetype='application/x-ndjson')

    except Exception as e:
        print(f"Error en /api/ask: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/')
def landing_page():
    return send_from_directory(static2_folder, 'index.html')

@app.route('/chat')
def chat_page():
    return send_from_directory(static_folder, 'chatbot.html')

@app.route('/static/<path:filename>')
def serve_chat_assets(filename):
    return send_from_directory(static_folder, filename)

@app.route('/static2/<path:filename>')
def serve_static2(filename):
    return send_from_directory(static2_folder, filename)


# --- 6. PUNTO DE ENTRADA ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)