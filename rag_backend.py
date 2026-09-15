# rag_backend.py

import os
import re
import numpy as np
import pymupdf
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# MODEL SETTINGS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen3-1.7B"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Fix PDF line-break words such as:
    # "knowl- edge" -> "knowledge"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SENTENCE-AWARE CHUNKING
# ============================================================

def sentence_chunk_text(
    text,
    max_words=180,
    overlap_sentences=2
):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []

    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = len(
            sentence.split()
        )

        if (
            current_sentences
            and current_word_count + sentence_words > max_words
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = (
                current_sentences[-overlap_sentences:]
            )

            current_word_count = sum(
                len(s.split())
                for s in current_sentences
            )

        current_sentences.append(sentence)

        current_word_count += sentence_words

    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return chunks

# ============================================================
# LOAD PDFs
# ============================================================

def load_pdfs(data_folder="data"):

    chunks = []

    pdf_files = [
        file
        for file in os.listdir(data_folder)
        if file.lower().endswith(".pdf")
    ]

    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            data_folder,
            pdf_file
        )

        document = pymupdf.open(pdf_path)

        for page_number, page in enumerate(document):

            text = page.get_text()
            text = clean_text(text)

            page_chunks = sentence_chunk_text(text)

            for chunk_number, chunk in enumerate(page_chunks):

                chunks.append({
                    "source": pdf_file,
                    "page": page_number + 1,
                    "chunk": chunk_number + 1,
                    "text": chunk
                })

        document.close()

    return chunks

# ============================================================
# BUILD EMBEDDING INDEX
# ============================================================

def build_index(chunks):

    # Load embedding model
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    # Load reranker
    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu"
    )

    # Extract chunk text
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Create embeddings
    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=False
    )

    # Normalize embeddings once
    normalized_embeddings = (
        chunk_embeddings
        / np.linalg.norm(
            chunk_embeddings,
            axis=1,
            keepdims=True
        )
    )

    return (
        embedding_model,
        reranker,
        normalized_embeddings
    )

# ============================================================
# RETRIEVE + RERANK
# ============================================================

def retrieve(
    question,
    chunks,
    embedding_model,
    reranker,
    normalized_embeddings,
    initial_k=20,
    final_k=3
):

    # Create question embedding
    question_embedding = embedding_model.encode(question)

    question_embedding = (
        question_embedding
        / np.linalg.norm(question_embedding)
    )

    # Dense similarity search
    scores = normalized_embeddings @ question_embedding

    candidate_indices = np.argsort(
        scores
    )[-initial_k:][::-1]

    # Prepare question-chunk pairs for CrossEncoder
    pairs = [
        (question, chunks[idx]["text"])
        for idx in candidate_indices
    ]

    reranker_scores = reranker.predict(pairs)

    # Rerank candidates
    reranked = sorted(
        zip(candidate_indices, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return best chunks
    results = []

    for idx, score in reranked[:final_k]:

        results.append({
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "chunk": chunks[idx]["chunk"],
            "text": chunks[idx]["text"],
            "score": float(score)
        })

    return results

# ============================================================
# LOAD LLM
# ============================================================

def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dtype = torch.float16 if device == "mps" else torch.float32

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        dtype=dtype
    )

    llm = llm.to(device)

    return tokenizer, llm

# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_results,
    tokenizer,
    llm
):

    context_parts = []

    for result in retrieved_results:

        context_parts.append(
            f"[Source: {result['source']}, Page {result['page']}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant answering questions "
                "from scientific documents. "
                "Use only the provided document context. "
                "Carefully read all context before answering. "
                "Give a concise and factual answer. "
                "If the answer is truly not present in the context, "
                "say that there is not enough information."
            )
        },
        {
            "role": "user",
            "content": f"""
DOCUMENT CONTEXT:

{context}

QUESTION:
{question}

Answer using the document context above.
"""
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    model_inputs = {
        key: value.to(llm.device)
        for key, value in model_inputs.items()
    }

    with torch.no_grad():

        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=180,
            do_sample=False
        )

    new_tokens = generated_ids[
        :,
        model_inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True
    )

    return answer