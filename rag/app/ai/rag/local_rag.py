from pathlib import Path

import faiss
import numpy as np
import ollama


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

# Folder containing your original text documents
DATA_DIR = BASE_DIR / "mimic_textbook" / "textbook" / "en"

LLM_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"  # Or your preferred local model (e.g., embeddinggemma-2:740m)

CHUNK_SIZE = 180
CHUNK_OVERLAP = 40
TOP_K = 3

# Global state variables to store the FAISS index and chunks in memory
rag_index = None
rag_chunks = []


# ==========================================
# 2. LOAD DOCUMENTS
# ==========================================

def load_documents(directory):
    documents = []
    if not directory.exists():
        return documents

    # Find every .txt file, including subdirectories
    for file_path in directory.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        if text.strip():
            documents.append({
                "source": file_path.name,
                "text": text
            })

    return documents


# ==========================================
# 3. SPLIT DOCUMENTS INTO CHUNKS
# ==========================================

def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move forward while keeping some overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def create_chunks(documents):
    chunks = []
    for document in documents:
        document_chunks = chunk_text(document["text"])
        for chunk in document_chunks:
            chunks.append({
                "source": document["source"],
                "text": chunk
            })
    return chunks


# ==========================================
# 4. GENERATE EMBEDDINGS USING OLLAMA
# ==========================================

def generate_embeddings(texts):
    embeddings = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=batch
        )
        embeddings.extend(response.embeddings)

    embedding_matrix = np.array(
        embeddings,
        dtype="float32"
    )

    return embedding_matrix


# ==========================================
# 5. INITIALIZE RAG SYSTEM (Called on startup)
# ==========================================

def initialize_rag():
    """Loads documents, chunks them, computes embeddings, and builds the FAISS index once."""
    global rag_index, rag_chunks

    print("\nLoading documents for RAG...")
    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print(f"Warning: No text documents found in {DATA_DIR}")
        return

    print("Creating document chunks...")
    rag_chunks = create_chunks(documents)
    print(f"Created {len(rag_chunks)} chunks.")

    print("Generating embeddings...")
    texts = [chunk["text"] for chunk in rag_chunks]
    embedding_matrix = generate_embeddings(texts)

    print("Creating FAISS index...")
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embedding_matrix)
    dimension = embedding_matrix.shape[1]

    # Inner product on normalized vectors = cosine similarity
    rag_index = faiss.IndexFlatIP(dimension)
    rag_index.add(embedding_matrix)
    print(f"FAISS index ready! Indexed vectors: {rag_index.ntotal}")


# ==========================================
# 6. QUERY RAG PIPELINE (Called per request)
# ==========================================

def query_rag(question: str):
    global rag_index, rag_chunks

    if rag_index is None or not rag_chunks:
        return {
            "answer": "The RAG system has not been initialized or no documents were loaded.",
            "sources": []
        }

    # Embed the incoming question
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    query_vector = np.array(
        response.embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    scores, indices = rag_index.search(
        query_vector,
        min(TOP_K, rag_index.ntotal)
    )

    retrieved_documents = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        retrieved_documents.append({
            "source": rag_chunks[idx]["source"],
            "text": rag_chunks[idx]["text"],
            "score": float(score)
        })

    # Prepare context from retrieved chunks
    context = "\n\n".join(
        [
            f"Source: {doc['source']}\n{doc['text']}"
            for doc in retrieved_documents
        ]
    )

    system_prompt = """
    You are a helpful AI assistant.

    Answer questions using ONLY the supplied context.

    If the answer is not available in the context,
    say that you cannot find the answer in the documents.

    Do not invent information.

    Mention the source document when possible.
    """

    user_prompt = f"""
    CONTEXT:

    {context}

    QUESTION:

    {question}

    Answer based on the context above.
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return {
        "answer": response.message.content,
        "sources": retrieved_documents
    }
    