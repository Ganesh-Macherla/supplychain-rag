from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

COLLECTION_NAME = "meridian_supply_chain"

EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# -----------------------------
# Load PDFs
# -----------------------------

def load_pdfs():
    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the data/ folder.")

    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)

        print(f"\nReading: {pdf_path.name}")
        print(f"Pages: {len(reader.pages)}")

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            documents.append({
                "text": text.strip(),
                "source": pdf_path.name,
                "page": page_number
            })

    return documents


# -----------------------------
# Chunk documents
# -----------------------------

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for document in documents:
        split_texts = splitter.split_text(document["text"])

        for chunk_number, chunk_text in enumerate(split_texts):
            chunks.append({
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"],
                "chunk": chunk_number
            })

    return chunks


# -----------------------------
# Create embeddings
# -----------------------------

def create_embeddings(chunks):
    all_embeddings = []

    batch_size = 32

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        print(
            f"Embedding chunks {i + 1}–"
            f"{min(i + batch_size, len(chunks))} "
            f"of {len(chunks)}..."
        )

        response = ollama.embed(
            model=EMBED_MODEL,
            input=[chunk["text"] for chunk in batch]
        )

        all_embeddings.extend(response["embeddings"])

    return all_embeddings


# -----------------------------
# Store in ChromaDB
# -----------------------------

def store_in_chroma(chunks, embeddings):
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    ids = []

    for index, chunk in enumerate(chunks):
        ids.append(
            f"{chunk['source']}_page_{chunk['page']}_chunk_{chunk['chunk']}_{index}"
        )

    collection.upsert(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"]
            }
            for chunk in chunks
        ]
    )

    print("\nChromaDB updated successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total chunks stored: {collection.count()}")


# -----------------------------
# Main
# -----------------------------

def main():
    print("=== Meridian Supply Chain RAG Ingestion ===")

    documents = load_pdfs()

    print(f"\nPages loaded: {len(documents)}")

    chunks = create_chunks(documents)

    print(f"Chunks created: {len(chunks)}")
    print(
        f"Chunk size: {CHUNK_SIZE} characters | "
        f"Overlap: {CHUNK_OVERLAP} characters"
    )

    embeddings = create_embeddings(chunks)

    print(f"Embeddings created: {len(embeddings)}")

    store_in_chroma(chunks, embeddings)

    print("\n=== INGESTION COMPLETE ===")


if __name__ == "__main__":
    main()