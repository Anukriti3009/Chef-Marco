"""
build_index.py — Run this ONCE to chunk + embed knowledge_base.txt
Saves embeddings to vector_store.json

Usage:
    python build_index.py
"""

import json
import re
import google.generativeai as genai


KNOWLEDGE_FILE = "knowledge_base.txt"
OUTPUT_FILE    = "vector_store.json"
EMBED_MODEL    = "models/gemini-embedding-001"  
CHUNK_SIZE     = 15   
CHUNK_OVERLAP  = 3    


api_key = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=api_key)


with open(KNOWLEDGE_FILE, "r") as f:
    raw_text = f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping line-based chunks.
    Skips blank lines and section dividers.
    """
    lines = [
        line.strip() for line in text.splitlines()
        if line.strip() and not re.match(r"^[-=]{4,}$", line.strip())
    ]

    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i : i + chunk_size]
        chunk_text  = "\n".join(chunk_lines)
        chunks.append(chunk_text)
        i += chunk_size - overlap  # slide forward with overlap

    return chunks

chunks = chunk_text(raw_text)
print(f"✅ Created {len(chunks)} chunks from {KNOWLEDGE_FILE}")


def embed_text(text):
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]

vector_store = []

for i, chunk in enumerate(chunks):
    print(f"   Embedding chunk {i+1}/{len(chunks)}...", end="\r")
    embedding = embed_text(chunk)
    vector_store.append({
        "id":        i,
        "text":      chunk,
        "embedding": embedding,
    })

print(f"\n✅ Embedded {len(vector_store)} chunks")


with open(OUTPUT_FILE, "w") as f:
    json.dump(vector_store, f)

print(f"✅ Saved vector store to {OUTPUT_FILE}")
print(f"\nYou're ready to run:  streamlit run app.py")
