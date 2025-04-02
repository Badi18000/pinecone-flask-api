import Pinecone
import spacy
import os
from dotenv import load_dotenv
import spacy.cli

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
DIMENSION = 300

pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")
index = pinecone.Index(INDEX_NAME)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
    )

index = pc.Index(INDEX_NAME)

spacy.cli.download("fr_core_news_sm")


nlp = spacy.load("fr_core_news_sm")


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
