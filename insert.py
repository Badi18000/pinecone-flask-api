import pinecone
import nltk
import re
import logging
import spacy
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import spacy.cli
import pdfplumber
# from pdf2image import convert_from_path  ❌ plus utilisé
import pytesseract
from utils import split_and_embed

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
DIMENSION = 300

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")

# Crée l'index si besoin
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine"
    )

index = pinecone.Index(INDEX_NAME)

spacy.cli.download("fr_core_news_sm")
nlp = spacy.load("fr_core_news_sm")

# ---------- 🧠 Extraction PDF classique ----------
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        logging.warning(f"[extract_text_from_pdf] Failed with error: {e}")
        return ""

# ---------- 🧠 OCR Fallback désactivé (pas supporté Render) ----------
def extract_text_with_ocr(pdf_path):
    logging.warning("[OCR] Skipped — 'pdf2image' non compatible avec Render.")
    return ""

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('french_grammars')
        tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
        return tokenizer
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {e}")
        return None

def create_chunks(text, max_tokens=500, overlap=50):
    text = clean_text(text)
    tokenizer = download_nltk_data()
    try:
        sentences = tokenizer.tokenize(text) if tokenizer else [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    except Exception as e:
        logging.error(f"Tokenization error: {e}")
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            if overlap:
                current_chunk = [current_chunk[-1], sentence]
                current_length = len(current_chunk[-1].split()) + len(sentence.split())
            else:
                current_chunk = [sentence]
                current_length = len(sentence.split())
        else:
            current_chunk.append(sentence)
            current_length += len(words)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def get_embedding(text):
    doc = nlp(text)
    return doc.vector.tolist()

# ---------- 🚀 Fonction principale appelée par app.py ----------
def process_and_upload_pdf(pdf_path):
    try:
        text = extract_text_from_pdf(pdf_path)

        if not text or len(text) < 50:
            logging.info("[INFO] PDF text seems empty or short. OCR skipped (no pdf2image on Render).")

        if not text or len(text.strip()) < 30:
            logging.error("[FAIL] No usable text found in PDF")
            return None

        chunks = create_chunks(text)
        if not chunks:
            logging.error("No valid chunks found.")
            return None

        documents = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            documents.append({
                'id': f'chunk_{i}',
                'text': chunk,
                'metadata': {
                    'chunk_index': i,
                    'language': 'fr',
                    'source': pdf_path,
                    'total_chunks': len(chunks),
                    'chunk_type': 'semantic'
                },
                'embedding': embedding
            })
        return documents
    except Exception as e:
        logging.error(f"[process_and_upload_pdf] Unexpected error: {e}")
        return None

def upsert_to_pinecone(documents, index_name):
    try:
        index = pinecone.Index(index_name)
        vectors = [{
            'id': doc['id'],
            'values': doc['embedding'],
            'metadata': doc['metadata']
        } for doc in documents]

        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i + 100])
        logging.info("Upload to Pinecone successful.")
        return True
    except Exception as e:
        logging.error(f"Upsert failed: {e}")
        return False
