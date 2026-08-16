import streamlit as st

from rag import ask_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Meridian Supply Chain RAG",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📦 Meridian Supply Chain RAG Assistant")

st.markdown(
    """
Ask natural-language questions about Meridian Components'
supply-chain and procurement documents.

The system retrieves relevant document sections from ChromaDB
and answers using the retrieved evidence.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
This assistant uses:

- PDF document processing
- Recursive text chunking
- Ollama embeddings
- ChromaDB vector search
- Local LLM answering
- Deterministic policy evaluation
- Document and page-level source evidence
        """
    )

    st.divider()

    st.subheader("Models")

    st.write("Embedding: `nomic-embed-text`")
    st.write("LLM: `llama3.2`")
    st.write("Vector DB: `ChromaDB`")


# ============================================================
# QUESTION
# ============================================================

st.header("Ask a question")

question = st.text_area(
    "Your question",
    placeholder=(
        "Example: Kaveri Metals recorded 88.1% on-time "
        "delivery and 1,150 defects per million in Q1. "
        "Which policy clauses does this trigger, and "
        "what exactly must the buyer do?"
    ),
    height=150,
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🔍 Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question first.")

    else:

        with st.spinner(
            "Retrieving documents and generating answer..."
        ):

            try:

                answer, retrieved, evaluation = ask_question(
                    question.strip()
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )

                st.stop()


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("Answer")

        st.markdown(answer)


        # ====================================================
        # POLICY EVALUATION
        # ====================================================

        if evaluation and evaluation.get("supplier"):

            st.subheader("Policy Evaluation")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Supplier",
                    evaluation["supplier"],
                )

            with col2:

                if evaluation.get("on_time") is not None:

                    st.metric(
                        "On-time Delivery",
                        f"{evaluation['on_time']}%",
                    )

            with col3:

                if evaluation.get("defect_ppm") is not None:

                    st.metric(
                        "Defect Rate",
                        f"{int(evaluation['defect_ppm']):,} PPM",
                    )


            st.write("### Clause Status")


            # ------------------------------------------------
            # CLAUSE 6.1
            # ------------------------------------------------

            clause = evaluation["clause_6_1"]

            if clause["triggered"]:

                st.success(
                    f"**Clause 6.1 — TRIGGERED**\n\n"
                    f"{clause['reason']}"
                )

            else:

                st.info(
                    f"**Clause 6.1 — "
                    f"{'NOT CONFIRMED' if clause['status'] == 'NOT_ESTABLISHED' else 'NOT TRIGGERED'}**\n\n"
                    f"{clause['reason']}"
                )


            # ------------------------------------------------
            # CLAUSE 6.2
            # ------------------------------------------------

            clause = evaluation["clause_6_2"]

            if clause["triggered"]:

                st.success(
                    f"**Clause 6.2 — TRIGGERED**\n\n"
                    f"{clause['reason']}"
                )

            else:

                st.info(
                    f"**Clause 6.2 — "
                    f"{'NOT CONFIRMED' if clause['status'] == 'NOT_ESTABLISHED' else 'NOT TRIGGERED'}**\n\n"
                    f"{clause['reason']}"
                )


            # ------------------------------------------------
            # CLAUSE 6.3
            # ------------------------------------------------

            clause = evaluation["clause_6_3"]

            if clause["triggered"]:

                st.success(
                    f"**Clause 6.3 — TRIGGERED**\n\n"
                    f"{clause['reason']}"
                )

            else:

                st.info(
                    f"**Clause 6.3 — "
                    f"{'NOT CONFIRMED' if clause['status'] == 'NOT_ESTABLISHED' else 'NOT TRIGGERED'}**\n\n"
                    f"{clause['reason']}"
                )


        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader("Retrieved Sources")

        seen = set()

        for item in retrieved:

            key = (
                item["source"],
                item["page"],
            )

            if key in seen:
                continue

            seen.add(key)

            with st.expander(
                f"{item['source']} — Page {item['page']}"
            ):

                st.write(item["text"])

                st.caption(
                    f"ChromaDB distance: "
                    f"{item['distance']:.4f}"
                )