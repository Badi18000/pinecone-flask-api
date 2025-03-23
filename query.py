import pinecone
import spacy
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
DIMENSION = 300

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)


nlp = spacy.load("fr_core_news_md")

def query_pinecone(user_query, top_k=5):
    doc = nlp(user_query)
    query_vector = doc.vector.tolist()
    response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    results = {
        "query": user_query,
        "results": []
    }
    for match in response.get("matches", []):
        results["results"].append({
            "score": match.get("score"),
            "text": match.get("metadata", {}).get("text"),
            "source": match.get("metadata", {}).get("source")
        })
    return results
