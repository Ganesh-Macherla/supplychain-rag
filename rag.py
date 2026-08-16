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

TOP_K = 8

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

def retrieve_context(question, expand_pages=True):
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

    if not expand_pages:
        return retrieved

    # --------------------------------------------------------
    # PAGE EXPANSION
    #
    # Semantic search finds the relevant page.
    # Then retrieve every chunk from that page so that
    # section/list questions receive the complete section.
    # --------------------------------------------------------

    page_keys = []

    for item in retrieved:

        key = (
            item["source"],
            item["page"],
        )

        if key not in page_keys:
            page_keys.append(key)

    expanded = []

    seen = set()

    for source, page in page_keys:

        page_results = collection.get(
            where={
                "$and": [
                    {"source": source},
                    {"page": page},
                ]
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        page_documents = page_results.get(
            "documents",
            [],
        )

        page_metadatas = page_results.get(
            "metadatas",
            [],
        )

        for document, metadata in zip(
            page_documents,
            page_metadatas,
        ):

            key = (
                metadata["source"],
                metadata["page"],
                document,
            )

            if key in seen:
                continue

            seen.add(key)

            expanded.append(
                {
                    "text": document,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "distance": 0.0,
                }
            )

    return expanded


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
# SECTION-AWARE GENERAL CONTEXT
# ============================================================

def select_general_context(question, retrieved):
    """
    Keep general-answer retrieval focused on the relevant
    document section instead of sending unrelated chunks
    to the LLM.

    If a retrieved chunk contains a section heading that
    matches the question, include the other retrieved chunks
    from that same document page.
    """

    question_lower = re.sub(
        r"\s+",
        " ",
        question.lower()
    ).strip()

    # Important section phrases that may appear in questions.
    section_phrases = [
        "risks carried into q2",
        "actions committed for q2",
        "supplier performance scorecard",
        "line-stoppage events",
        "demand and planning accuracy",
        "inventory and safety stock policy",
        "sourcing rules",
        "performance failures and consequences",
        "escalation matrix",
    ]

    # Find the most relevant explicit section phrase.
    matched_phrase = None

    for phrase in section_phrases:
        if phrase in question_lower:
            matched_phrase = phrase
            break

    # If the question clearly refers to a named section,
    # keep chunks from that section's document page.
    if matched_phrase:

        anchor_items = []

        for item in retrieved:

            text_lower = re.sub(
                r"\s+",
                " ",
                item["text"].lower()
            )

            if matched_phrase in text_lower:
                anchor_items.append(item)

        if anchor_items:

            anchor = anchor_items[0]

            same_page = [
                item
                for item in retrieved
                if (
                    item["source"] == anchor["source"]
                    and item["page"] == anchor["page"]
                )
            ]

            if same_page:
                return same_page

    # Otherwise use normal semantic retrieval.
    return retrieved[:6]


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

def extract_scorecard_metrics(supplier, retrieved):
    result = {
        "on_time": None,
        "defect_ppm": None,
    }

    if not supplier:
        return result

    # Normalize whitespace so PDF line breaks do not prevent
    # matching supplier names such as "Shenzhen Rui Electronics".
    normalized_supplier = re.sub(r"\s+", " ", supplier.lower()).strip()

    for item in retrieved:
        text = item["text"]

        # Normalize PDF whitespace while preserving the actual content.
        normalized_text = re.sub(r"\s+", " ", text).strip()
        normalized_lower = normalized_text.lower()

        if normalized_supplier not in normalized_lower:
            continue

        # Scorecard rows follow this structure:
        #
        # Supplier | Category | Location | On-time | Defect PPM |
        # Avg lead time | Q1 spend
        #
        # Example:
        # Shenzhen Rui Electronics Microcontrollers Shenzhen, China
        # 79.5% 210 46 21.9
        #
        # Once the supplier is found, search forward for the first
        # percentage followed by three numeric scorecard values.

        supplier_pos = normalized_lower.find(normalized_supplier)

        nearby = normalized_text[supplier_pos:supplier_pos + 500]

        pattern = re.compile(
            r"(\d+(?:\.\d+)?)%\s+"
            r"([\d,]+)\s+"
            r"\d+\s+"
            r"[\d.]+",
            flags=re.IGNORECASE,
        )

        match = pattern.search(nearby)

        if match:
            result["on_time"] = float(match.group(1))
            result["defect_ppm"] = float(
                match.group(2).replace(",", "")
            )
            return result

    return result


# ============================================================
# CLAUSE 6.2 — STRICT SUPPLIER-SPECIFIC CHECK
# ============================================================

def check_clause_62(supplier, retrieved):
    """
    Clause 6.2:
    On-time delivery below 85% for two consecutive quarters.

    Evidence must explicitly associate the consecutive-quarter
    statement with the requested supplier.
    """

    if not supplier:
        return {
            "triggered": False,
            "status": "UNKNOWN",
            "reason": "Supplier not identified.",
        }

    supplier_lower = supplier.lower()

    for item in retrieved:
        text = item["text"]

        if supplier_lower not in text.lower():
            continue

        # Look at the sentence containing the supplier name.
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if supplier_lower not in sentence_lower:
                continue

            has_consecutive = (
                "second consecutive quarter" in sentence_lower
                or "two consecutive quarters" in sentence_lower
                or "consecutive quarters" in sentence_lower
            )

            has_below_85 = (
                "below 85%" in sentence_lower
                or "below 85 percent" in sentence_lower
            )

            if has_consecutive and has_below_85:
                return {
                    "triggered": True,
                    "status": "CONFIRMED",
                    "reason": (
                        "The documents explicitly associate "
                        "two consecutive below-85% quarters "
                        "with this supplier."
                    ),
                }

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
        "consequence",
        "consequences",
        "policy consequence",
        "policy consequences",
        "what must the buyer do",
        "buyer do",
        "required action",
        "debit note",
        "improvement plan",
        "incoming inspection",
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

1. Never invent facts.
2. Never use outside knowledge.
3. Do not mix facts between suppliers.
4. Do not confuse table columns.
5. Do not infer missing information.
6. If the question asks for multiple items, return ALL
   items explicitly supported by the documents.
7. If the question refers to a named section, preserve
   that section's complete list.
8. Do not stop after the first few items.
9. Do not add information from a different section unless
   the question requires it.
10. If the requested information is not available, respond exactly:

"The information is not available in the provided documents."

11. Keep answers concise but complete.
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

    policy_question = is_policy_question(question)

    retrieved = retrieve_context(
        question,
        expand_pages=not policy_question,
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

        general_retrieved = select_general_context(
            question,
            retrieved,
        )

        context = build_context(
            general_retrieved
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

    if evaluation["on_time"] is not None:
        print(
            f"On-time delivery: "
            f"{evaluation['on_time']}%"
        )
    else:
        print("On-time delivery: Not established from retrieved evidence")

    if evaluation["defect_ppm"] is not None:
        print(
            f"Defect rate: "
            f"{int(evaluation['defect_ppm']):,} PPM"
        )
    else:
        print("Defect rate: Not established from retrieved evidence")

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