import os
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
from llama_cpp import Llama
from flask import Flask, request, jsonify, render_template, Response
import time 

# --- 1. INICIALIZACIÓN DE LA APLICACIÓN FLASK ---
app = Flask(__name__)

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

def retrieve_context(query, k_retriever=20, k_reranker=4):
    # 1. Búsqueda inicial
    query_vector = retriever_model.encode([query])
    _, indices = index.search(query_vector, k_retriever)
    
    initial_chunks_with_meta = [
        (
            get_text_chunk(metadata[vector_id]['source_file'], metadata[vector_id]['chunk_index']),
            metadata[vector_id]
        )
        for vector_id in indices[0] if vector_id != -1
    ]
    
    meaningful_chunks_with_meta = [
        (chunk, meta) for chunk, meta in initial_chunks_with_meta if len(chunk) > 150
    ]

    if not meaningful_chunks_with_meta:
        return "", []

    # 2. Re-clasificación de los chunks
    rerank_pairs = [[query, chunk] for chunk, meta in meaningful_chunks_with_meta]
    scores = cross_encoder_model.predict(rerank_pairs)

    scored_chunks_with_meta = sorted(zip(scores, meaningful_chunks_with_meta), key=lambda x: x[0], reverse=True)

    # 3. Selección final y construcción del contexto
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

def generate_answer_stream(query, context):
    """Genera la respuesta en modo stream, produciendo cada token."""
    prompt = f"""<|system|>
Usted es un asistente experto en biología y astronáutica. Su tarea es doble:
1. Primero, responda la pregunta del usuario de forma concisa basándose ÚNICAMENTE en el contexto proporcionado.
2. Después de la respuesta, en una nueva línea, genere un objeto JSON que represente un grafo de conocimiento de los conceptos clave. El JSON debe tener una clave "graph_data" que contenga una lista de objetos, donde cada objeto tiene "source", "target" y "relationship".

Si la información no está en el contexto, responda únicamente "La información no se encuentra en mis documentos." y no genere el JSON. No invente nada.
Su respuesta DEBE seguir este formato:
Respuesta en texto...
{{
  "graph_data": [
    {{"source": "concepto1", "target": "concepto2", "relationship": "es parte de"}},
    {{"source": "concepto3", "target": "concepto4", "relationship": "causa"}}
  ]
}}
</s>
<|user|>
CONTEXTO:
{context}

PREGUNTA:
{query}</s>
<|assistant|>
"""
   
    stream = llm(prompt, max_tokens=512, stop=["</s>", "<|user|>"], echo=False, temperature=0.1, stream=True)
    
    for output in stream:
        if 'choices' in output and len(output['choices']) > 0:
            token = output['choices'][0].get('text', '')
            if token:
                yield token

# --- 4. DEFINICIÓN DE LAS RUTAS DE LA API ---

@app.route('/')
def home():
    """Sirve la página principal HTML."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Recibe una pregunta, la procesa con la IA y devuelve la respuesta en stream."""
    data = request.get_json()
    query = data.get('question')

    if not query:
        return jsonify({'error': 'No question provided'}), 400

    print(f"Recibida pregunta para streaming: {query}")

    def stream_response():
        context, sources = retrieve_context(query)
       
        if not context.strip():
            yield 'data: {"token": "The information is not in my documents."}\n\n'
            yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"
            return

        # Stream de los tokens de la respuesta
        for token in generate_answer_stream(query, context):
            yield f"data: {json.dumps({'token': token})}\n\n"
      
        # Una vez terminada la respuesta, enviar las fuentes
        yield f"data: {json.dumps({'sources': sources})}\n\n"

        # Señal de finalización
        yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"


    return Response(stream_response(), mimetype='text/event-stream')

# --- 5. PUNTO DE ENTRADA PARA EJECUTAR EL SERVIDOR ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)