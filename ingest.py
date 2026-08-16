import os
import sys
from typing import Dict, List

from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    "chroma_db"
)

COLLECTION_NAME = "meridian_supply_chain"

EMBEDDING_MODEL = "nomic-embed-text"

OLLAMA_URL = "http://localhost:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

def get_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


# --------------------------------------------------
# Ollama Embeddings
# --------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:

    embeddings = []

    for index, text in enumerate(texts, start=1):

        print(
            f"Embedding chunk {index}/{len(texts)}..."
        )

        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text,
            },
            timeout=120,
        )

        response.raise_for_status()

        embeddings.append(
            response.json()["embedding"]
        )

    return embeddings


# --------------------------------------------------
# Extract PDF pages
# --------------------------------------------------

def extract_pages(pdf_path: str) -> List[Dict]:

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = text.strip()

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# --------------------------------------------------
# Recursive-style chunking
# --------------------------------------------------

def recursive_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:

    separators = [
        "\n\n",
        "\n",
        ". ",
        " "
    ]

    def split_text(
        current_text: str,
        separators_left: List[str]
    ):

        if len(current_text) <= chunk_size:

            return [
                current_text
            ] if current_text.strip() else []

        if not separators_left:

            return [
                current_text[i:i + chunk_size]
                for i in range(
                    0,
                    len(current_text),
                    chunk_size
                )
            ]

        separator = separators_left[0]

        parts = current_text.split(
            separator
        )

        chunks = []

        current = ""

        for part in parts:

            candidate = (
                current
                + (
                    separator
                    if current
                    else ""
                )
                + part
            )

            if len(candidate) <= chunk_size:

                current = candidate

            else:

                if current:

                    chunks.append(current)

                if len(part) > chunk_size:

                    chunks.extend(
                        split_text(
                            part,
                            separators_left[1:]
                        )
                    )

                    current = ""

                else:

                    current = part

        if current:

            chunks.append(current)

        return chunks

    raw_chunks = split_text(
        text,
        separators
    )

    chunks = []

    for index, chunk in enumerate(
        raw_chunks
    ):

        if index == 0 or overlap == 0:

            chunks.append(chunk)

        else:

            previous_tail = raw_chunks[
                index - 1
            ][-overlap:]

            chunks.append(
                previous_tail + chunk
            )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# --------------------------------------------------
# Ingest PDFs
# --------------------------------------------------

def ingest_files(
    pdf_paths: List[str]
) -> Dict:

    collection = get_collection()

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for pdf_path in pdf_paths:

        filename = os.path.basename(
            pdf_path
        )

        print(
            f"\nProcessing: {filename}"
        )

        pages = extract_pages(
            pdf_path
        )

        print(
            f"Pages extracted: {len(pages)}"
        )

        for page_info in pages:

            page_chunks = recursive_split(
                page_info["text"]
            )

            for chunk_number, chunk_text in enumerate(
                page_chunks
            ):

                chunk_id = (
                    f"{filename}"
                    f"::p{page_info['page']}"
                    f"::c{chunk_number}"
                )

                all_chunks.append(
                    chunk_text
                )

                all_metadatas.append({
                    "source": filename,
                    "page": page_info["page"]
                })

                all_ids.append(
                    chunk_id
                )

    if not all_chunks:

        return {
            "files": len(pdf_paths),
            "chunks": 0
        }

    print(
        f"\nTotal chunks: {len(all_chunks)}"
    )

    print(
        "\nGenerating embeddings..."
    )

    embeddings = embed_texts(
        all_chunks
    )

    print(
        "\nStoring embeddings in ChromaDB..."
    )

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas
    )

    return {
        "files": len(pdf_paths),
        "chunks": len(all_chunks)
    }


# --------------------------------------------------
# Collection statistics
# --------------------------------------------------

def collection_stats() -> Dict:

    collection = get_collection()

    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection.count(),
        "embedding_model": EMBEDDING_MODEL
    }


# --------------------------------------------------
# Standalone execution
# --------------------------------------------------

if __name__ == "__main__":

    paths = sys.argv[1:]

    if not paths:

        paths = [
            "data/Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf",
            "data/Meridian_Procurement_Policy_Handbook_v4.2.pdf"
        ]

    result = ingest_files(
        paths
    )

    print(
        f"\n{result['files']} files processed, "
        f"{result['chunks']} chunks stored."
    )