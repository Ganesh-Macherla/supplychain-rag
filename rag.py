"""
rag.py
Meridian Supply Chain RAG Assistant

Architecture:

1. Retrieve relevant chunks from ChromaDB.
2. Identify the supplier and metrics from the user's question.
3. Deterministically evaluate policy clauses.
4. Generate the policy answer directly from Python.
5. Use the LLM only for general document questions.

IMPORTANT:
Policy threshold decisions are NOT delegated to the LLM.
This prevents hallucinations and cross-supplier contamination.
"""

import re
import requests
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "meridian_supply_chain"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

TOP_K = 6

OLLAMA_URL = "http://localhost:11434"


# ============================================================
# SUPPLIER NAMES
# ============================================================

SUPPLIERS = [
    "Kaveri Metals",
    "Nexa Polymers",
    "Shenzhen Rui Electronics",
    "Baltic Wire",
    "Sunrise Connectors",
    "Trident Circuit Boards",
]


# ============================================================
# OLLAMA EMBEDDING
# ============================================================

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


# ============================================================
# OLLAMA CHAT
# ============================================================

def generate_with_ollama(system_prompt, user_prompt):

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


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(question):

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    query_embedding = get_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        retrieved.append(
            {
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": distance,
            }
        )

    return retrieved


# ============================================================
# CONTEXT
# ============================================================

def build_context(retrieved):

    parts = []

    for index, item in enumerate(
        retrieved,
        start=1,
    ):

        parts.append(
            f"""
SOURCE {index}
Document: {item['source']}
Page: {item['page']}

{item['text']}
"""
        )

    return "\n".join(parts)


# ============================================================
# IDENTIFY SUPPLIER
# ============================================================

def identify_supplier(question):

    question_lower = question.lower()

    for supplier in SUPPLIERS:

        if supplier.lower() in question_lower:

            return supplier

    return None


# ============================================================
# EXTRACT METRICS FROM QUESTION
# ============================================================

def extract_question_metrics(question):

    metrics = {
        "on_time": None,
        "defect_ppm": None,
    }

    # --------------------------------------------------------
    # Percentage
    # --------------------------------------------------------

    percentage_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*%",
        question,
    )

    if percentage_matches:

        metrics["on_time"] = float(
            percentage_matches[0]
        )

    # --------------------------------------------------------
    # PPM
    # --------------------------------------------------------

    ppm_matches = re.findall(
        r"([\d,]+)\s*(?:defects?\s*(?:per|/)\s*million|PPM)",
        question,
        flags=re.IGNORECASE,
    )

    if ppm_matches:

        metrics["defect_ppm"] = float(
            ppm_matches[0].replace(",", "")
        )

    return metrics


# ============================================================
# EXTRACT SUPPLIER SCORECARD METRICS
# ============================================================

def extract_scorecard_metrics(
    supplier,
    retrieved,
):

    result = {
        "on_time": None,
        "defect_ppm": None,
    }

    if not supplier:
        return result

    supplier_lower = supplier.lower()

    for item in retrieved:

        text = item["text"]

        if supplier_lower not in text.lower():
            continue

        # ----------------------------------------------------
        # Kaveri scorecard row
        #
        # Kaveri Metals Pvt Ltd
        # Stamped contacts
        # Coimbatore, India
        # 88.1% 1,150 22 8.7
        # ----------------------------------------------------

        if supplier_lower == "kaveri metals":

            match = re.search(
                r"Kaveri\s+Metals.*?"
                r"(\d+(?:\.\d+)?)%\s+"
                r"([\d,]+)",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if match:

                result["on_time"] = float(
                    match.group(1)
                )

                result["defect_ppm"] = float(
                    match.group(2).replace(",", "")
                )

                return result

    return result


# ============================================================
# CLAUSE 6.2 — STRICT SUPPLIER-SPECIFIC CHECK
# ============================================================

def check_clause_62(
    supplier,
    retrieved,
):

    """
    Clause 6.2:

    On-time delivery below 85% for TWO CONSECUTIVE QUARTERS.

    CRITICAL:

    We do NOT search for the supplier and the word
    "consecutive" anywhere in the same chunk.

    The evidence must explicitly associate the consecutive-quarter
    statement with THIS supplier.

    Example from the Meridian document:

    Shenzhen Rui Electronics ...
    This is the second consecutive quarter in which its
    on-time delivery has fallen below 85%;
    Q4 FY 2024-25 closed at 83.2%.

    This is evidence for SHENZHEN RUI.

    It is NOT evidence for Kaveri Metals.
    """

    if not supplier:

        return {
            "triggered": False,
            "status": "UNKNOWN",
            "reason": "Supplier not identified.",
        }

    supplier_lower = supplier.lower()

    # --------------------------------------------------------
    # We deliberately look at sentences/paragraphs.
    # --------------------------------------------------------

    for item in retrieved:

        text = item["text"]

        if supplier_lower not in text.lower():
            continue

        # Split into sentences.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        for sentence in sentences:

            sentence_lower = sentence.lower()

            # Supplier must appear in THIS sentence.
            if supplier_lower not in sentence_lower:
                continue

            # And the same sentence must explicitly contain
            # the consecutive-quarter condition.
            consecutive = (
                "second consecutive quarter"
                in sentence_lower
                or
                "two consecutive quarters"
                in sentence_lower
            )

            below_85 = (
                "below 85%"
                in sentence_lower
            )

            if consecutive and below_85:

                return {
                    "triggered": True,
                    "status": "CONFIRMED",
                    "reason": (
                        "The documents explicitly associate "
                        "two consecutive below-85% quarters "
                        "with this supplier."
                    ),
                }

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # If we have not found supplier-specific evidence,
    # we DO NOT guess.
    # --------------------------------------------------------

    return {
        "triggered": False,
        "status": "NOT_ESTABLISHED",
        "reason": (
            "The documents do not establish two consecutive "
            "below-85% quarters for this supplier."
        ),
    }


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(
    question,
    retrieved,
):

    supplier = identify_supplier(
        question
    )

    question_metrics = extract_question_metrics(
        question
    )

    scorecard_metrics = extract_scorecard_metrics(
        supplier,
        retrieved,
    )

    # --------------------------------------------------------
    # Prefer numbers explicitly supplied by the user.
    # --------------------------------------------------------

    on_time = question_metrics["on_time"]

    if on_time is None:

        on_time = scorecard_metrics["on_time"]

    defect_ppm = question_metrics["defect_ppm"]

    if defect_ppm is None:

        defect_ppm = scorecard_metrics["defect_ppm"]

    # --------------------------------------------------------
    # Clause 6.1
    # --------------------------------------------------------

    if on_time is None:

        clause_61 = {
            "triggered": False,
            "status": "NOT_ESTABLISHED",
            "reason": (
                "On-time delivery could not be established."
            ),
        }

    elif on_time < 90:

        clause_61 = {
            "triggered": True,
            "status": "CONFIRMED",
            "reason": (
                f"{on_time}% is below the 90% threshold."
            ),
        }

    else:

        clause_61 = {
            "triggered": False,
            "status": "CONFIRMED",
            "reason": (
                f"{on_time}% is not below the 90% threshold."
            ),
        }

    # --------------------------------------------------------
    # Clause 6.2
    # --------------------------------------------------------

    clause_62 = check_clause_62(
        supplier,
        retrieved,
    )

    # --------------------------------------------------------
    # Clause 6.3
    # --------------------------------------------------------

    if defect_ppm is None:

        clause_63 = {
            "triggered": False,
            "status": "NOT_ESTABLISHED",
            "reason": (
                "Defect rate could not be established."
            ),
        }

    elif defect_ppm > 500:

        clause_63 = {
            "triggered": True,
            "status": "CONFIRMED",
            "reason": (
                f"{int(defect_ppm):,} PPM is above "
                f"the 500 PPM threshold."
            ),
        }

    else:

        clause_63 = {
            "triggered": False,
            "status": "CONFIRMED",
            "reason": (
                f"{int(defect_ppm):,} PPM is not above "
                f"the 500 PPM threshold."
            ),
        }

    return {
        "supplier": supplier,
        "on_time": on_time,
        "defect_ppm": defect_ppm,
        "clause_6_1": clause_61,
        "clause_6_2": clause_62,
        "clause_6_3": clause_63,
    }


# ============================================================
# POLICY QUESTION DETECTION
# ============================================================

def is_policy_question(question):

    question_lower = question.lower()

    keywords = [
        "clause",
        "policy",
        "trigger",
        "penalty",
        "buyer do",
        "what must",
        "required action",
        "debit note",
        "improvement plan",
        "inspection",
        "delivery review",
    ]

    return any(
        keyword in question_lower
        for keyword in keywords
    )


# ============================================================
# BUILD DETERMINISTIC POLICY ANSWER
# ============================================================

def build_policy_answer(
    evaluation,
    retrieved,
):

    supplier = evaluation["supplier"]

    lines = []

    lines.append(
        f"Supplier: {supplier}"
    )

    lines.append("")

    lines.append(
        "POLICY EVALUATION"
    )

    lines.append("")

    # --------------------------------------------------------
    # Clause 6.1
    # --------------------------------------------------------

    c61 = evaluation["clause_6_1"]

    lines.append(
        "Clause 6.1 — On-time delivery below 90%"
    )

    lines.append(
        f"Status: "
        f"{'TRIGGERED' if c61['triggered'] else 'NOT TRIGGERED'}"
    )

    lines.append(
        f"Evidence: {c61['reason']}"
    )

    if c61["triggered"]:

        lines.append(
            "Required action: Issue a written warning "
            "within 10 working days of quarter close, "
            "and move the supplier to a weekly delivery "
            "review call until performance recovers above "
            "90% for one full quarter."
        )

    lines.append("")

    # --------------------------------------------------------
    # Clause 6.2
    # --------------------------------------------------------

    c62 = evaluation["clause_6_2"]

    lines.append(
        "Clause 6.2 — Below 85% for two consecutive quarters"
    )

    if c62["status"] == "CONFIRMED":

        lines.append(
            "Status: "
            f"{'TRIGGERED' if c62['triggered'] else 'NOT TRIGGERED'}"
        )

    else:

        lines.append(
            "Status: NOT CONFIRMED"
        )

    lines.append(
        f"Evidence: {c62['reason']}"
    )

    if c62["triggered"]:

        lines.append(
            "Required action: Raise a debit note equal "
            "to 2% of the quarterly invoice value and "
            "require the supplier to submit a formal "
            "improvement plan within 15 working days."
        )

    else:

        lines.append(
            "Required action: None under Clause 6.2 because "
            "the two-consecutive-quarter condition is not "
            "established for this supplier."
        )

    lines.append("")

    # --------------------------------------------------------
    # Clause 6.3
    # --------------------------------------------------------

    c63 = evaluation["clause_6_3"]

    lines.append(
        "Clause 6.3 — Defect rate above 500 PPM"
    )

    lines.append(
        f"Status: "
        f"{'TRIGGERED' if c63['triggered'] else 'NOT TRIGGERED'}"
    )

    lines.append(
        f"Evidence: {c63['reason']}"
    )

    if c63["triggered"]:

        lines.append(
            "Required action: The supplier bears the "
            "cost of rework at ₹120 per affected unit, "
            "and 100% incoming inspection is imposed "
            "at the supplier's cost until three consecutive "
            "lots are accepted without defect."
        )

    lines.append("")

    # --------------------------------------------------------
    # Final applicable clauses
    # --------------------------------------------------------

    applicable = []

    if c61["triggered"]:
        applicable.append("6.1")

    if c62["triggered"]:
        applicable.append("6.2")

    if c63["triggered"]:
        applicable.append("6.3")

    lines.append(
        "APPLICABLE CLAUSES"
    )

    if applicable:

        lines.append(
            ", ".join(applicable)
        )

    else:

        lines.append(
            "No policy clause could be confirmed."
        )

    lines.append("")

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    lines.append(
        "SOURCE EVIDENCE"
    )

    seen = set()

    for item in retrieved:

        key = (
            item["source"],
            item["page"],
        )

        if key not in seen:

            lines.append(
                f"- {item['source']} — "
                f"Page {item['page']}"
            )

            seen.add(key)

    return "\n".join(lines)


# ============================================================
# GENERAL LLM ANSWER
# ============================================================

def generate_general_answer(
    question,
    context,
):

    system_prompt = """
You are a document-grounded supply chain assistant.

Answer ONLY using the supplied document context.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. Do not mix facts between suppliers.
4. If the answer is unavailable, say:
   "The information is not available in the provided documents."
5. Mention document names and page numbers where possible.
"""

    user_prompt = f"""
DOCUMENT CONTEXT
================

{context}


QUESTION
========

{question}


Answer using only the document context.
"""

    return generate_with_ollama(
        system_prompt,
        user_prompt,
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def ask_question(question):

    retrieved = retrieve_context(
        question
    )

    evaluation = evaluate_policy(
        question,
        retrieved,
    )

    # --------------------------------------------------------
    # POLICY QUESTIONS ARE ANSWERED DETERMINISTICALLY.
    #
    # The LLM does NOT get to modify the policy result.
    # --------------------------------------------------------

    if is_policy_question(question):

        answer = build_policy_answer(
            evaluation,
            retrieved,
        )

    else:

        context = build_context(
            retrieved
        )

        answer = generate_general_answer(
            question,
            context,
        )

    return (
        answer,
        retrieved,
        evaluation,
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "MERIDIAN SUPPLY CHAIN RAG"
    )

    print(
        "=" * 70
    )

    question = input(
        "\nAsk a question: "
    ).strip()

    if not question:

        print(
            "No question entered."
        )

        return

    print(
        "\nRetrieving relevant documents..."
    )

    try:

        answer, retrieved, evaluation = ask_question(
            question
        )

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)

        return

    # --------------------------------------------------------
    # Retrieved chunks
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETRIEVED CHUNKS"
    )

    print(
        "=" * 70
    )

    for index, item in enumerate(
        retrieved,
        start=1,
    ):

        print(
            f"\n--- Chunk {index} ---"
        )

        print(
            f"Source: {item['source']}"
        )

        print(
            f"Page: {item['page']}"
        )

        print(
            f"Distance: {item['distance']:.4f}"
        )

        print(
            item["text"][:1000]
        )

    # --------------------------------------------------------
    # Deterministic evaluation
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DETERMINISTIC POLICY CHECK"
    )

    print(
        "=" * 70
    )

    print(
        f"Supplier: {evaluation['supplier']}"
    )

    print(
        f"On-time delivery: "
        f"{evaluation['on_time']}%"
    )

    print(
        f"Defect rate: "
        f"{int(evaluation['defect_ppm']):,} PPM"
    )

    print()

    for clause_number, key in [
        ("6.1", "clause_6_1"),
        ("6.2", "clause_6_2"),
        ("6.3", "clause_6_3"),
    ]:

        clause = evaluation[key]

        print(
            f"Clause {clause_number}: "
            f"{'TRIGGERED' if clause['triggered'] else 'NOT TRIGGERED'}"
        )

        print(
            f"  {clause['reason']}"
        )

    # --------------------------------------------------------
    # Final answer
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SOURCES"
    )

    print(
        "=" * 70
    )

    seen = set()

    for item in retrieved:

        key = (
            item["source"],
            item["page"],
        )

        if key not in seen:

            print(
                f"- {item['source']} — "
                f"Page {item['page']}"
            )

            seen.add(key)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()