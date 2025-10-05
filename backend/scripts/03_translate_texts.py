import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# --- Configuración ---
# Directorio donde están los archivos de texto en inglés
SOURCE_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\Processed"
# Directorio donde guardaremos los textos traducidos
TRANSLATED_DIR = r"C:\Users\axelq\Documents\nasa-bio-ai\backend\data\Translated_es"
# Idioma de destino (para el nombre de la carpeta)
TARGET_LANG = "es" # es: español, fr: francés, de: alemán, etc.

# Crear el directorio de salida si no existe
os.makedirs(TRANSLATED_DIR, exist_ok=True)

# --- Carga del Modelo de Traducción ---
# Elige el modelo según el idioma de destino. Ejemplos:
# Inglés a Español: 'Helsinki-NLP/opus-mt-en-es'
# Inglés a Francés: 'Helsinki-NLP/opus-mt-en-fr'
# Inglés a Alemán: 'Helsinki-NLP/opus-mt-en-de'
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-es'

print(f"Cargando el tokenizador y el modelo de traducción ({MODEL_NAME})...")
# La primera vez se descargará automáticamente.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Modelo cargado.")

# --- NUEVO: Define el tamaño del lote ---
# Este es el número de párrafos que se traducirán a la vez.
# Si tu PC sigue lenta, puedes reducir este número a 16, 8 o incluso 4.
BATCH_SIZE = 32

# --- Procesamiento de archivos ---
txt_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".txt")]

print(f"Se encontraron {len(txt_files)} archivos de texto para traducir a '{TARGET_LANG}'.")

for txt_file in tqdm(txt_files, desc=f"Traduciendo a {TARGET_LANG}"):
    source_path = os.path.join(SOURCE_DIR, txt_file)
    translated_path = os.path.join(TRANSLATED_DIR, txt_file)

    # Revisar si el archivo ya fue traducido
    if os.path.exists(translated_path):
        continue

    try:
        # Leer el contenido del archivo de texto
        with open(source_path, "r", encoding="utf-8") as f:
            # Leemos el texto y lo dividimos en párrafos para no sobrecargar el modelo
            paragraphs = f.read().split('\n')
            paragraphs = [p for p in paragraphs if p.strip()] # Eliminar párrafos vacíos

        if not paragraphs:
            print(f"\n[ADVERTENCIA] El archivo {txt_file} está vacío.")
            continue

        # --- LÓGICA MODIFICADA: Procesar en lotes ---
        all_translated_text = []
        # Itera sobre la lista de párrafos en trozos del tamaño de BATCH_SIZE
        for i in range(0, len(paragraphs), BATCH_SIZE):
            # Obtiene el lote actual de párrafos
            batch = paragraphs[i:i + BATCH_SIZE]
            
            # Tokeniza solo el lote actual
            tokenized_text = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Genera la traducción para el lote
            translation_ids = model.generate(**tokenized_text)
            
            # Decodifica y guarda el resultado del lote
            translated_batch = tokenizer.batch_decode(translation_ids, skip_special_tokens=True)
            all_translated_text.extend(translated_batch)

        # Guardar el texto traducido completo del archivo
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_translated_text))

    except Exception as e:
        print(f"\n[ERROR] No se pudo procesar el archivo {txt_file}: {e}")

print("\nTraducción de todos los archivos completada.")
print(f"Los archivos traducidos se han guardado en: {TRANSLATED_DIR}")