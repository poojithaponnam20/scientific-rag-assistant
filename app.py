import streamlit as st

from rag_backend import (
    load_pdfs,
    build_index,
    retrieve,
    load_llm,
    generate_answer
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Scientific Literature RAG Assistant",
    layout="wide"
)

st.title("Scientific Literature RAG Assistant")

st.caption(
    "Multi-document RAG system for scientific literature using "
    "SentenceTransformers, CrossEncoder reranking, and Qwen3."
)

st.divider()

st.write(
    "Ask questions about scientific papers and receive grounded "
    "answers with document and page citations."
)


# ============================================================
# LOAD RAG SYSTEM ONCE
# ============================================================

@st.cache_resource
def initialize_rag():

    chunks = load_pdfs("data")

    embedding_model, reranker, normalized_embeddings = build_index(
        chunks
    )

    tokenizer, llm = load_llm()

    return (
        chunks,
        embedding_model,
        reranker,
        normalized_embeddings,
        tokenizer,
        llm
    )


with st.spinner("Loading RAG models and scientific papers..."):

    (
        chunks,
        embedding_model,
        reranker,
        normalized_embeddings,
        tokenizer,
        llm
    ) = initialize_rag()


document_names = sorted(
    set(chunk["source"] for chunk in chunks)
)

st.success(
    f"Ready — {len(document_names)} papers and "
    f"{len(chunks)} chunks loaded."
)

with st.sidebar:

    st.header("📄 Loaded Papers")

    for document in document_names:
        st.write(f"• {document}")

    st.divider()

    st.write("**Retrieval:** MiniLM embeddings")
    st.write("**Reranking:** CrossEncoder")
    st.write("**Generator:** Qwen3-1.7B")


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is Self-RAG?"
)


if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching papers and generating answer..."):

            results = retrieve(
                question,
                chunks,
                embedding_model,
                reranker,
                normalized_embeddings,
                initial_k=20,
                final_k=3
            )

            answer = generate_answer(
                question,
                results,
                tokenizer,
                llm
            )

        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("Answer")
        st.write(answer)

        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader("Sources")

        seen_sources = set()

        for result in results:

            source_key = (
                result["source"],
                result["page"]
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            st.write(
                f"📄 **{result['source']}** — "
                f"Page {result['page']}"
            )

            with st.expander(
                f"View retrieved passage — "
                f"{result['source']}, Page {result['page']}"
            ):

                st.write(result["text"])

                st.caption(
                    f"Reranker score: {result['score']:.4f}"
                )