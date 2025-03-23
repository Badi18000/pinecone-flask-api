from flask import Flask, request, jsonify
from insert import process_and_upload_pdf, create_index, upsert_to_pinecone
from query import query_pinecone
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query_route():
    data = request.get_json()
    question = data.get("query")
    if not question:
        return jsonify({"error": "Missing 'query' field"}), 400
    results = query_pinecone(question)
    return jsonify(results), 200

@app.route("/insert", methods=["POST"])
def insert_route():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400

    pdf = request.files['pdf']
    path = f"/tmp/{pdf.filename}"
    pdf.save(path)

    create_index(INDEX_NAME)
    documents = process_and_upload_pdf(path)
    if documents:
        success = upsert_to_pinecone(documents, INDEX_NAME)
        if success:
            return jsonify({"message": "PDF uploaded to Pinecone"}), 200
    return jsonify({"error": "Failed to process PDF"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

#if __name__ == "__main__":
    #app.run(debug=True)
