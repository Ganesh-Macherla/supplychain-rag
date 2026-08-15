import chromadb
import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "meridian_supply_chain"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

TOP_K = 5

OLLAMA_URL = "http://localhost:11434"


# --------------------------------------------------
# Generate query embedding
# --------------------------------------------------

def get_query_embedding(query):

    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": query,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embedding"]


# --------------------------------------------------
# Retrieve relevant chunks
# --------------------------------------------------

def retrieve_context(query):

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    query_embedding = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        retrieved.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
        })

    return retrieved


# --------------------------------------------------
# Build context for LLM
# --------------------------------------------------

def build_context(retrieved):

    context_parts = []

    for index, item in enumerate(
        retrieved,
        start=1,
    ):

        context_parts.append(
            f"""
SOURCE {index}
Document: {item['source']}
Page: {item['page']}

Content:
{item['text']}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Generate answer using Ollama
# --------------------------------------------------

def generate_answer(question, context):

    system_prompt = """
You are a careful document-based question answering
assistant for Meridian Components.

Answer the user's question ONLY using the provided
document context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts, numbers, names, dates,
   suppliers, or policies.

3. NEVER combine information belonging to different
   suppliers.

4. Pay close attention to units:
   - percentages
   - PPM
   - ₹ crore
   - days

5. If a policy requires multiple conditions, verify
   that all conditions are actually supported by the
   documents.

6. Do not assume that a historical value mentioned for
   one supplier belongs to another supplier.

7. If the provided context does not contain enough
   information to answer the question, say:

   "The information is not available in the provided documents."

8. For cross-document questions, synthesize information
   from the relevant documents.

9. Cite the supporting document and page number.

10. Be concise and precise.
"""

    user_prompt = f"""
DOCUMENT CONTEXT
================

{context}

QUESTION
========

{question}

Answer using ONLY the document context.

Do not combine facts from different suppliers.

Include supporting document names and page numbers.
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


# --------------------------------------------------
# Complete RAG pipeline
# --------------------------------------------------

def ask_question(question):

    retrieved = retrieve_context(question)

    context = build_context(retrieved)

    answer = generate_answer(
        question,
        context,
    )

    return answer, retrieved


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

def main():

    print("=" * 70)
    print("MERIDIAN SUPPLY CHAIN RAG")
    print("=" * 70)

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("No question entered.")

        return

    print("\nRetrieving relevant documents...")

    answer, retrieved = ask_question(
        question
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for index, item in enumerate(
        retrieved,
        start=1,
    ):

        print(
            f"{index}. "
            f"{item['source']} "
            f"(Page {item['page']})"
        )


if __name__ == "__main__":
    main()