
import os
import faiss
from sentence_transformers import SentenceTransformer
import json
from llama_cpp import Llama
from flask import Flask, request, jsonify, render_template

# --- 1. INICIALIZACIÓN DE LA APLICACIÓN FLASK ---
app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend')

# --- 2. CARGA DE MODELOS (SE HACE UNA SOLA VEZ AL INICIAR EL SERVIDOR) ---
print("Cargando componentes de la IA... Esto puede tardar varios minutos.")

# Rutas a los datos
TXT_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\Processed"
INDEX_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\faiss_index.bin"
METADATA_FILE = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\metadata.json"
MODEL_PATH = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Carga de modelos
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

print("¡Servidor listo! Los modelos de IA se han cargado.")

# --- 3. LÓGICA DE LA IA (LAS MISMAS FUNCIONES DE TU SCRIPT) ---
# (Copiamos las funciones get_text_chunk, retrieve_context, y generate_answer aquí)

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

def retrieve_context(query, k=5):
    query_vector = retriever_model.encode([query])
    _, indices = index.search(query_vector, k)
    all_chunks = [get_text_chunk(metadata[vector_id]['source_file'], metadata[vector_id]['chunk_index']) for vector_id in indices[0] if vector_id != -1]
    
    final_context = []
    total_tokens = 0
    CONTEXT_TOKEN_LIMIT = 1500 

    for chunk in all_chunks:
        chunk_tokens = llm.tokenize(chunk.encode("utf-8", errors="ignore"))
        if total_tokens + len(chunk_tokens) <= CONTEXT_TOKEN_LIMIT:
            final_context.append(chunk)
            total_tokens += len(chunk_tokens)
        else:
            break
    return "\n\n---\n\n".join(final_context)

def generate_answer(query, context):
    prompt = f"""<|system|>
You are an expert assistant in biology and astronautics. Answer the question based ONLY on the provided context. If the information is not in the context, say "The information is not in my documents." Do not invent anything.</s>
<|user|>
CONTEXT:
{context}

QUESTION:
{query}</s>
<|assistant|>
"""
    output = llm(prompt, max_tokens=256, stop=["</s>", "<|user|>"], echo=False)
    if output and 'choices' in output and len(output['choices']) > 0:
        return output['choices'][0]['text'].strip()
    return "Error: Could not generate an answer."

# --- 4. DEFINICIÓN DE LAS RUTAS DE LA API ---

@app.route('/')
def home():
    """Sirve la página principal HTML."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Recibe una pregunta, la procesa con la IA y devuelve una respuesta."""
    data = request.get_json()
    query = data.get('question')

    if not query:
        return jsonify({'error': 'No question provided'}), 400

    print(f"Recibida pregunta: {query}")

    # Ejecutar el pipeline de IA
    context = retrieve_context(query)
    answer = generate_answer(query, context)

    print(f"Respuesta generada: {answer}")
    
    return jsonify({'answer': answer})

# --- 5. PUNTO DE ENTRADA PARA EJECUTAR EL SERVIDOR ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)