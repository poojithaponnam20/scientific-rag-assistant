import numpy as np
# STEP 1: Setup

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Model loaded!")

# STEP 2: Understanding Embeddings

sentences = [
    "Early life stress affects decision making.",
    "Stress can influence choices and behavior.",
    "The weather is sunny today."
]

embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print(embeddings[0][:10])

# STEP 3: Cosine Similarity

from sentence_transformers.util import cos_sim

similarity = cos_sim(embeddings, embeddings)

print("\nCosine similarity matrix:")
print(similarity)


# STEP 33: Load Multiple PDFs

import os
import pymupdf

data_folder = "data"

pdf_files = [
    file
    for file in os.listdir(data_folder)
    if file.lower().endswith(".pdf")
]

print("\nPDF files found:")
print(pdf_files)

# STEP 4: Load and Extract Text from PDF
import pymupdf

pdf_path = "data/PDF_1.pdf"

pdf = pymupdf.open(pdf_path)

print("\nNumber of pages:", len(pdf))

pages = []

for page_number, page in enumerate(pdf):
    text = page.get_text()

    pages.append({
        "page": page_number + 1,
        "text": text
    })

print("Extracted pages:", len(pages))

print("\nFirst page preview:")
print(pages[0]["text"][:1000])

# STEP 5: Split PDF Text into Chunks

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = []

for page in pages:
    page_chunks = chunk_text(page["text"])

    for chunk_number, chunk in enumerate(page_chunks):
        chunks.append({
            "page": page["page"],
            "chunk": chunk_number + 1,
            "text": chunk
        })

print("\nTotal chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])

# STEP 6: Create Embeddings for PDF Chunks

chunk_texts = [chunk["text"] for chunk in chunks]

chunk_embeddings = model.encode(
    chunk_texts,
    show_progress_bar=True
)

print("\nNumber of chunk texts:", len(chunk_texts))
print("Chunk embedding shape:", chunk_embeddings.shape)

# STEP 7: Retrieve Relevant Chunks

question = "What is Retrieval-Augmented Generation?"

question_embedding = model.encode(question)

# Normalize question embedding
question_embedding = question_embedding / np.linalg.norm(question_embedding)

# Normalize chunk embeddings
normalized_chunks = chunk_embeddings / np.linalg.norm(
    chunk_embeddings,
    axis=1,
    keepdims=True
)

# Cosine similarity
scores = normalized_chunks @ question_embedding

# Get top 5 most similar chunks
top_k = 5
top_indices = np.argsort(scores)[-top_k:][::-1]

print("\nTop relevant chunks:\n")

for idx in top_indices:
    print("Similarity:", round(float(scores[idx]), 4))
    print("Page:", chunks[idx]["page"])
    print("Chunk:", chunks[idx]["chunk"])
    print(chunks[idx]["text"])
    print("-" * 80)

# STEP 8: FAISS Vector Search

import faiss

# Make a float32 copy because FAISS expects float32 vectors
faiss_embeddings = chunk_embeddings.astype("float32").copy()

# Normalize chunk embeddings
faiss.normalize_L2(faiss_embeddings)

# Get embedding dimension
dimension = faiss_embeddings.shape[1]

# Create FAISS index using inner product
index = faiss.IndexFlatIP(dimension)

# Add all chunk embeddings
index.add(faiss_embeddings)

print("\nVectors stored in FAISS:", index.ntotal)

question = "What is Retrieval-Augmented Generation?"

question_embedding = model.encode([question]).astype("float32")

# Normalize question embedding
faiss.normalize_L2(question_embedding)

top_k = 5

scores, indices = index.search(
    question_embedding,
    top_k
)

print("\nTop FAISS results:\n")

for score, idx in zip(scores[0], indices[0]):
    print("Similarity:", round(float(score), 4))
    print("Page:", chunks[idx]["page"])
    print("Chunk:", chunks[idx]["chunk"])
    print(chunks[idx]["text"])
    print("-" * 80)

# STEP 9: Clean Extracted PDF Text

import re

def clean_text(text):

    # Join words broken across PDF lines
    # "mem- ory" -> "memory"
    # "down- stream" -> "downstream"
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    # Replace line breaks with spaces
    text = text.replace("\n", " ")

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


cleaned_pages = []

for page in pages:
    cleaned_pages.append({
        "page": page["page"],
        "text": clean_text(page["text"])
    })

print("\nCleaned pages:", len(cleaned_pages))

print("\nRAW TEXT:\n")
print(pages[1]["text"][:500])

print("\n" + "-" * 80)

print("\nCLEANED TEXT:\n")
print(cleaned_pages[1]["text"][:500])

# STEP 10: Create Cleaned Chunks

cleaned_chunks = []

for page in cleaned_pages:
    page_chunks = chunk_text(page["text"])

    for chunk_number, chunk in enumerate(page_chunks):
        cleaned_chunks.append({
            "page": page["page"],
            "chunk": chunk_number + 1,
            "text": chunk
        })

print("\nOriginal chunks:", len(chunks))
print("Cleaned chunks:", len(cleaned_chunks))

print("\nExample cleaned chunk:\n")

print("Page:", cleaned_chunks[0]["page"])
print("Chunk:", cleaned_chunks[0]["chunk"])
print(cleaned_chunks[0]["text"])

# STEP 11: Create Embeddings for Cleaned Chunks

cleaned_chunk_texts = [
    chunk["text"] for chunk in cleaned_chunks
]

cleaned_chunk_embeddings = model.encode(
    cleaned_chunk_texts,
    show_progress_bar=True
)

print("\nCleaned chunk embeddings shape:")
print(cleaned_chunk_embeddings.shape)

# STEP 12: Compare Retrieval Before and After Cleaning

question = "What is Retrieval-Augmented Generation?"

# Create question embedding
question_embedding = model.encode(question)

# Normalize question embedding
question_embedding = question_embedding / np.linalg.norm(question_embedding)

# Normalize cleaned chunk embeddings
normalized_cleaned_chunks = cleaned_chunk_embeddings / np.linalg.norm(
    cleaned_chunk_embeddings,
    axis=1,
    keepdims=True
)

# Calculate cosine similarity
cleaned_scores = normalized_cleaned_chunks @ question_embedding

# Get top 5 results
top_k = 5

cleaned_top_indices = np.argsort(cleaned_scores)[-top_k:][::-1]

print("\nTOP RESULTS AFTER CLEANING:\n")

for idx in cleaned_top_indices:
    print("Similarity:", round(float(cleaned_scores[idx]), 4))
    print("Page:", cleaned_chunks[idx]["page"])
    print("Chunk:", cleaned_chunks[idx]["chunk"])
    print(cleaned_chunks[idx]["text"])
    print("-" * 80)

# STEP 13: Sentence-Aware Chunking

import re

def sentence_chunk_text(text, max_words=180, overlap_sentences=2):

    # Split text at sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = len(sentence.split())

        # If adding this sentence makes the chunk too large,
        # save the current chunk
        if current_sentences and current_word_count + sentence_words > max_words:

            chunks.append(" ".join(current_sentences))

            # Keep the last few sentences as overlap
            current_sentences = current_sentences[-overlap_sentences:]

            current_word_count = sum(
                len(s.split()) for s in current_sentences
            )

        current_sentences.append(sentence)
        current_word_count += sentence_words

    # Add final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks

smart_chunks = []

for page in cleaned_pages:

    page_chunks = sentence_chunk_text(page["text"])

    for chunk_number, chunk in enumerate(page_chunks):

        smart_chunks.append({
            "page": page["page"],
            "chunk": chunk_number + 1,
            "text": chunk
        })


print("\nOld cleaned chunks:", len(cleaned_chunks))
print("Sentence-aware chunks:", len(smart_chunks))

print("\nSENTENCE-AWARE CHUNKS:\n")

for chunk in smart_chunks[:3]:

    print("Page:", chunk["page"])
    print("Chunk:", chunk["chunk"])
    print(chunk["text"])
    print("-" * 80)

# STEP 14: Create Embeddings for Sentence-Aware Chunks

smart_chunk_texts = [
    chunk["text"] for chunk in smart_chunks
]

smart_chunk_embeddings = model.encode(
    smart_chunk_texts,
    show_progress_bar=True
)

print("\nSentence-aware embedding shape:")
print(smart_chunk_embeddings.shape)

# Retrieve using sentence-aware chunks

question = "What is Retrieval-Augmented Generation?"

question_embedding = model.encode(question)

# Normalize question
question_embedding = (
    question_embedding /
    np.linalg.norm(question_embedding)
)

# Normalize sentence-aware chunk embeddings
normalized_smart_chunks = (
    smart_chunk_embeddings /
    np.linalg.norm(
        smart_chunk_embeddings,
        axis=1,
        keepdims=True
    )
)

# Cosine similarity
smart_scores = (
    normalized_smart_chunks @ question_embedding
)

# Top 5
top_k = 5

smart_top_indices = np.argsort(
    smart_scores
)[-top_k:][::-1]

print("\nTOP RESULTS WITH SENTENCE-AWARE CHUNKING:\n")

for idx in smart_top_indices:

    print(
        "Similarity:",
        round(float(smart_scores[idx]), 4)
    )

    print("Page:", smart_chunks[idx]["page"])
    print("Chunk:", smart_chunks[idx]["chunk"])

    print(smart_chunks[idx]["text"])

    print("-" * 80)

# STEP 15: Better PDF Text Cleaning

def clean_text(text):

    # Join words broken by PDF line wrapping
    # Example: "knowl-\nedge" -> "knowledge"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Replace remaining line breaks with spaces
    text = text.replace("\n", " ")

    # Collapse multiple spaces/tabs
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# STEP 16: Rerank Retrieved Chunks

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cpu"
)

print("\nReranker loaded!")

top_k = 10

candidate_indices = np.argsort(
    smart_scores
)[-top_k:][::-1]

pairs = []

for idx in candidate_indices:
    pairs.append(
        (
            question,
            smart_chunks[idx]["text"]
        )
    )

reranker_scores = reranker.predict(pairs)

reranked = sorted(
    zip(
        candidate_indices,
        reranker_scores
    ),
    key=lambda x: x[1],
    reverse=True
)

print("\nTOP RESULTS AFTER RERANKING:\n")

for idx, score in reranked[:5]:

    print(
        "Reranker score:",
        round(float(score), 4)
    )

    print(
        "Page:",
        smart_chunks[idx]["page"]
    )

    print(
        "Chunk:",
        smart_chunks[idx]["chunk"]
    )

    print(smart_chunks[idx]["text"])

    print("-" * 80)

# STEP 17: Prepare Context for the LLM

top_chunks = reranked[:3]

context_parts = []

for idx, score in top_chunks:
    chunk = smart_chunks[idx]

    context_parts.append(
        f"[Page {chunk['page']}]\n{chunk['text']}"
    )

context = "\n\n".join(context_parts)

print("\nCONTEXT FOR LLM:\n")
print(context)

# STEP 18: Load Hugging Face LLM

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

llm_model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

print("\nTokenizer loaded!")

test_text = "What is Retrieval-Augmented Generation?"

tokens = tokenizer(test_text)

print("\nToken IDs:")
print(tokens["input_ids"])

print("\nNumber of tokens:")
print(len(tokens["input_ids"]))

# STEP 18B: Load the Qwen LLM

from transformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    dtype=torch.float16
)

llm = llm.to("mps")

print("\nQwen model loaded!")
print("Model device:", next(llm.parameters()).device)

print("\nQwen model loaded!")
print("Model device:", next(llm.parameters()).device)

# STEP 18C: Build the RAG Prompt

messages = [
    {
    "role": "system",
    "content": (
        "You are a research assistant answering questions from scientific documents. "
        "Use only the provided document context. "
        "Carefully read all provided context before answering. "
        "If the context contains information that answers the question, use it to give "
        "a concise and factual answer. "
        "Do not say that information is unavailable when relevant information is present "
        "anywhere in the provided context. "
        "Only say that there is not enough information if the context truly does not "
        "contain an answer."
    )
    },
    {
    "role": "user",
    "content": f"""
DOCUMENT CONTEXT:

{context}

QUESTION:
{question}

Answer the question using the document context above.
"""

    }
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

print("\nFORMATTED PROMPT:\n")
print(formatted_prompt[:2000])

enable_thinking=False

model_inputs = tokenizer(
    formatted_prompt,
    return_tensors="pt"
)

model_inputs = {
    key: value.to(llm.device)
    for key, value in model_inputs.items()
}

print("\nRAG prompt token count:")
print(model_inputs["input_ids"].shape[1])

# STEP 18D: Generate the Answer

with torch.no_grad():
    generated_ids = llm.generate(
    **model_inputs,
    max_new_tokens=150,
    do_sample=False
)

# Keep only newly generated tokens

new_tokens = generated_ids[
    :,
    model_inputs["input_ids"].shape[1]:
]
# STEP 18E: Decode the Answer

answer = tokenizer.decode(
    new_tokens[0],
    skip_special_tokens=True
)

print("\n" + "=" * 80)
print("FINAL RAG ANSWER")
print("=" * 80)

print(answer)

# STEP 18F: Add Source Citations

source_pages = []

for idx, score in top_chunks:
    page = smart_chunks[idx]["page"]

    if page not in source_pages:
        source_pages.append(page)

sources = ", ".join(
    f"Page {page}" for page in source_pages
)

print("\n" + "=" * 80)
print("FINAL RAG ANSWER")
print("=" * 80)

print(answer)

print("\nSources:", sources)

# STEP 19: Reusable Retrieval + Reranking Function

def retrieve_and_rerank(question, initial_k=10, final_k=5):

    # 1. Create question embedding
    question_embedding = model.encode(question)

    question_embedding = (
        question_embedding /
        np.linalg.norm(question_embedding)
    )

    # 2. Normalize smart chunk embeddings
    normalized_embeddings = (
        smart_chunk_embeddings /
        np.linalg.norm(
            smart_chunk_embeddings,
            axis=1,
            keepdims=True
        )
    )

    # 3. Semantic similarity
    scores = normalized_embeddings @ question_embedding

    # 4. Get Top 10 candidates
    candidate_indices = np.argsort(
        scores
    )[-initial_k:][::-1]

    # 5. Create question-chunk pairs
    pairs = [
        (question, smart_chunks[idx]["text"])
        for idx in candidate_indices
    ]

    # 6. CrossEncoder reranking
    reranker_scores = reranker.predict(pairs)

    # 7. Sort by reranker score
    reranked_results = sorted(
        zip(candidate_indices, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 8. Return best results
    return reranked_results[:final_k]

test_question = "What is Retrieval-Augmented Generation?"

results = retrieve_and_rerank(test_question)

print("\nEVALUATION TEST:\n")

for idx, score in results:

    print("Reranker score:", round(float(score), 4))
    print("Page:", smart_chunks[idx]["page"])
    print("Chunk:", smart_chunks[idx]["chunk"])
    print("-" * 50)

# STEP 20: Evaluate Retrieval Performance

evaluation_questions = [
    {
       "question": "What is Retrieval-Augmented Generation?",
        "expected_pages": [1, 2]
    },
    {
        "question": "What are the parametric and non-parametric components of RAG?",
        "expected_pages": [1, 2]
    },
    {
        "question": "What are RAG-Sequence and RAG-Token?",
        "expected_pages": [3]
    },
    {
        "question": "How does the RAG-Sequence model generate an output sequence?",
        "expected_pages": [3]
    },
    {
        "question": "How does the RAG-Token model differ from RAG-Sequence?",
        "expected_pages": [3]
    },
    {
        "question": "What retriever and generator architectures are used in RAG?",
        "expected_pages": [2]
    },
    {
        "question": "How does RAG perform compared with BART on MS-MARCO?",
        "expected_pages": [6]
    },
    {
        "question": "Can RAG update its knowledge by changing the external document index?",
        "expected_pages": [8]
    },
    {
        "question": "What societal risks or disadvantages of RAG are discussed?",
        "expected_pages": [10]
    },
    {
        "question": "What is retrieval collapse?",
        "expected_pages": [19]
    }
]

results_summary = []

for item in evaluation_questions:

    question = item["question"]
    expected_pages = item["expected_pages"]

    results = retrieve_and_rerank(
        question,
        initial_k=20,
        final_k=5
    )

    retrieved_pages = [
        smart_chunks[idx]["page"]
        for idx, score in results
    ]

    recall_at_1 = any(
        page in expected_pages
        for page in retrieved_pages[:1]
    )

    recall_at_3 = any(
        page in expected_pages
        for page in retrieved_pages[:3]
    )

    recall_at_5 = any(
        page in expected_pages
        for page in retrieved_pages[:5]
    )

    results_summary.append({
        "question": question,
        "expected": expected_pages,
        "retrieved": retrieved_pages,
        "R@1": recall_at_1,
        "R@3": recall_at_3,
        "R@5": recall_at_5
    })

    print("\n" + "=" * 80)
print("RETRIEVAL EVALUATION")
print("=" * 80)

for result in results_summary:

    print("\nQuestion:", result["question"])
    print("Expected pages:", result["expected"])
    print("Retrieved pages:", result["retrieved"])

    print("Recall@1:", result["R@1"])
    print("Recall@3:", result["R@3"])
    print("Recall@5:", result["R@5"])

failed_question = "What are the main components of the RAG model?"

failed_results = retrieve_and_rerank(
    failed_question,
    initial_k=10,
    final_k=5
)

print("\nFAILED QUESTION ANALYSIS:\n")

for idx, score in failed_results:

    print("Score:", round(float(score), 4))
    print("Page:", smart_chunks[idx]["page"])
    print("Chunk:", smart_chunks[idx]["chunk"])
    print("Text:")
    print(smart_chunks[idx]["text"][:700])
    print("\n" + "-" * 80)

diagnostic_question = (
    "What are the parametric and non-parametric components "
    "of Retrieval-Augmented Generation?"
)

diagnostic_results = retrieve_and_rerank(
    diagnostic_question,
    initial_k=10,
    final_k=5
)

print("\nDIAGNOSTIC TEST:\n")

for idx, score in diagnostic_results:
    print("Score:", round(float(score), 4))
    print("Page:", smart_chunks[idx]["page"])
    print("Chunk:", smart_chunks[idx]["chunk"])
    print("-" * 50)

# STEP 22: Overall Retrieval Metrics

total_questions = len(results_summary)

recall_1 = sum(result["R@1"] for result in results_summary) / total_questions
recall_3 = sum(result["R@3"] for result in results_summary) / total_questions
recall_5 = sum(result["R@5"] for result in results_summary) / total_questions

print("\n" + "=" * 80)
print("OVERALL RETRIEVAL METRICS")
print("=" * 80)

print(f"Recall@1: {recall_1:.2%}")
print(f"Recall@3: {recall_3:.2%}")
print(f"Recall@5: {recall_5:.2%}")

# STEP 24: Inspect Dense Retrieval Before Reranking

test_questions = [
    "What are the parametric and non-parametric components of RAG?",
    "What retriever and generator architectures are used in RAG?"
]

for question in test_questions:

    question_embedding = model.encode(question)
    question_embedding = (
        question_embedding /
        np.linalg.norm(question_embedding)
    )

    normalized_embeddings = (
        smart_chunk_embeddings /
        np.linalg.norm(
            smart_chunk_embeddings,
            axis=1,
            keepdims=True
        )
    )

    scores = normalized_embeddings @ question_embedding

    top_indices = np.argsort(scores)[-10:][::-1]

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    for rank, idx in enumerate(top_indices, start=1):

        print(
            f"{rank}. Page {smart_chunks[idx]['page']} | "
            f"Chunk {smart_chunks[idx]['chunk']} | "
            f"Similarity {scores[idx]:.4f}"
        )

# STEP 26: Compare Dense Retrieval vs Reranking

dense_hits_1 = 0
dense_hits_3 = 0
dense_hits_5 = 0

rerank_hits_1 = 0
rerank_hits_3 = 0
rerank_hits_5 = 0

normalized_embeddings = (
    smart_chunk_embeddings /
    np.linalg.norm(
        smart_chunk_embeddings,
        axis=1,
        keepdims=True
    )
)

for item in evaluation_questions:

    question = item["question"]
    expected_pages = item["expected_pages"]

    # -------------------------
    # Dense retrieval only
    # -------------------------

    question_embedding = model.encode(question)
    question_embedding /= np.linalg.norm(question_embedding)

    scores = normalized_embeddings @ question_embedding

    dense_indices = np.argsort(scores)[-5:][::-1]

    dense_pages = [
        smart_chunks[idx]["page"]
        for idx in dense_indices
    ]

    dense_hits_1 += any(
        p in expected_pages for p in dense_pages[:1]
    )

    dense_hits_3 += any(
        p in expected_pages for p in dense_pages[:3]
    )

    dense_hits_5 += any(
        p in expected_pages for p in dense_pages[:5]
    )

    # -------------------------
    # Dense + reranking
    # -------------------------

    reranked = retrieve_and_rerank(
        question,
        initial_k=20,
        final_k=5
    )

    reranked_pages = [
        smart_chunks[idx]["page"]
        for idx, score in reranked
    ]

    rerank_hits_1 += any(
        p in expected_pages for p in reranked_pages[:1]
    )

    rerank_hits_3 += any(
        p in expected_pages for p in reranked_pages[:3]
    )

    rerank_hits_5 += any(
        p in expected_pages for p in reranked_pages[:5]
    )

n = len(evaluation_questions)

print("\n" + "=" * 80)
print("DENSE RETRIEVAL VS RERANKING")
print("=" * 80)

print("\nDense retrieval only:")
print(f"Recall@1: {dense_hits_1 / n:.2%}")
print(f"Recall@3: {dense_hits_3 / n:.2%}")
print(f"Recall@5: {dense_hits_5 / n:.2%}")

print("\nDense retrieval + CrossEncoder:")
print(f"Recall@1: {rerank_hits_1 / n:.2%}")
print(f"Recall@3: {rerank_hits_3 / n:.2%}")
print(f"Recall@5: {rerank_hits_5 / n:.2%}")

# STEP 27: Reusable RAG Answer Function

def generate_rag_answer(question, initial_k=20, context_k=3):

    # 1. Retrieve + rerank
    results = retrieve_and_rerank(
        question,
        initial_k=initial_k,
        final_k=context_k
    )

    # 2. Build context
    context_parts = []
    source_pages = []

    for idx, score in results:

        chunk = smart_chunks[idx]

        context_parts.append(
            f"[Page {chunk['page']}]\n{chunk['text']}"
        )

        if chunk["page"] not in source_pages:
            source_pages.append(chunk["page"])

    context = "\n\n".join(context_parts)

    # 3. Build messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. "
                "Answer using only the provided document context. "
                "If the answer is not present, say that there is "
                "not enough information in the document. "
                "Give a concise factual answer."
            )
        },
        {
            "role": "user",
            "content": f"""
QUESTION:
{question}

DOCUMENT CONTEXT:
{context}
"""
        }
    ]

    # 4. Apply Qwen chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # 5. Tokenize
    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    model_inputs = {
        key: value.to(llm.device)
        for key, value in model_inputs.items()
    }

    # 6. Generate
    with torch.no_grad():

        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )

    # 7. Keep only new tokens
    new_tokens = generated_ids[
        :,
        model_inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        new_tokens[0],
        skip_special_tokens=True
    )

    return answer, source_pages

test_questions = [
    "What is Retrieval-Augmented Generation?",
    "What is retrieval collapse?",
    "What societal risks of RAG are discussed?"
]

for question in test_questions:

    answer, sources = generate_rag_answer(question)

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:", sources)

generation_test_questions = [
    item["question"]
    for item in evaluation_questions
]

for question in generation_test_questions:

    answer, sources = generate_rag_answer(question)

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    print("ANSWER:")
    print(answer)

    print("\nSOURCES:", sources)

# STEP 30: Inspect context for generation failure

question = "What societal risks or disadvantages of RAG are discussed?"

results = retrieve_and_rerank(
    question,
    initial_k=20,
    final_k=3
)

print("\nCONTEXT INSPECTION:\n")

for rank, (idx, score) in enumerate(results, start=1):

    chunk = smart_chunks[idx]

    print(f"RESULT {rank}")
    print("Page:", chunk["page"])
    print("Reranker score:", round(float(score), 4))
    print("Text:")
    print(chunk["text"])
    print("\n" + "-" * 80)

new_question = "What societal risks or disadvantages of RAG are discussed?"

answer, sources = generate_rag_answer(new_question)

print("\nANSWER:")
print(answer)

print("\nSOURCES:", sources)

print("\n" + "=" * 80)
print("STEP 33 CHECK")
print("=" * 80)
print("PDF files found:", pdf_files)

# STEP 34: Extract pages from multiple PDFs

multi_pages = []

for pdf_file in pdf_files:

    pdf_path_multi = os.path.join(data_folder, pdf_file)
    document = pymupdf.open(pdf_path_multi)

    for page_number, page in enumerate(document):

        multi_pages.append({
            "source": pdf_file,
            "page": page_number + 1,
            "text": page.get_text()
        })

    document.close()


print("\n" + "=" * 80)
print("STEP 34 CHECK")
print("=" * 80)

print("Total pages extracted:", len(multi_pages))
print("First source:", multi_pages[0]["source"])
print("First page:", multi_pages[0]["page"])

# STEP 35: Clean and chunk multiple PDFs

multi_cleaned_pages = []

for page in multi_pages:
    multi_cleaned_pages.append({
        "source": page["source"],
        "page": page["page"],
        "text": clean_text(page["text"])
    })


multi_chunks = []

for page in multi_cleaned_pages:

    page_chunks = sentence_chunk_text(page["text"])

    for chunk_number, chunk in enumerate(page_chunks):

        multi_chunks.append({
            "source": page["source"],
            "page": page["page"],
            "chunk": chunk_number + 1,
            "text": chunk
        })


print("\n" + "=" * 80)
print("STEP 35 CHECK")
print("=" * 80)

print("Total chunks:", len(multi_chunks))
print("First chunk source:", multi_chunks[0]["source"])
print("First chunk page:", multi_chunks[0]["page"])
print("First chunk number:", multi_chunks[0]["chunk"])

# STEP 36: Create embeddings for multi-PDF chunks

multi_chunk_texts = [
    chunk["text"]
    for chunk in multi_chunks
]

multi_chunk_embeddings = model.encode(
    multi_chunk_texts,
    show_progress_bar=True
)

print("\n" + "=" * 80)
print("STEP 36 CHECK")
print("=" * 80)

print("Embedding shape:", multi_chunk_embeddings.shape)
print("Number of chunks:", len(multi_chunks))

# STEP 37: Multi-PDF retrieval + reranking

def retrieve_and_rerank_multi(question, initial_k=20, final_k=5):

    question_embedding = model.encode(question)
    question_embedding = (
        question_embedding /
        np.linalg.norm(question_embedding)
    )

    normalized_embeddings = (
        multi_chunk_embeddings /
        np.linalg.norm(
            multi_chunk_embeddings,
            axis=1,
            keepdims=True
        )
    )

    scores = normalized_embeddings @ question_embedding

    candidate_indices = np.argsort(scores)[-initial_k:][::-1]

    pairs = [
        (question, multi_chunks[idx]["text"])
        for idx in candidate_indices
    ]

    reranker_scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(candidate_indices, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return reranked_results[:final_k]

test_question = "What is Retrieval-Augmented Generation?"

results = retrieve_and_rerank_multi(test_question)

print("\n" + "=" * 80)
print("STEP 37 CHECK")
print("=" * 80)

for idx, score in results[:3]:

    chunk = multi_chunks[idx]

    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Chunk:", chunk["chunk"])
    print("Score:", round(float(score), 4))
    print("-" * 50)

# STEP 38: Multi-PDF source test

multi_test_questions = [
    "What is Self-RAG?",
    "What are the parametric and non-parametric components of the original RAG model?"
]

for question in multi_test_questions:

    results = retrieve_and_rerank_multi(
        question,
        initial_k=20,
        final_k=5
    )

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    for idx, score in results[:5]:

        chunk = multi_chunks[idx]

        print(
            f"Source: {chunk['source']} | "
            f"Page: {chunk['page']} | "
            f"Chunk: {chunk['chunk']} | "
            f"Score: {float(score):.4f}"
        )

# STEP 39: Multi-PDF RAG generation

def generate_multi_rag_answer(question, initial_k=20, context_k=3):

    results = retrieve_and_rerank_multi(
        question,
        initial_k=initial_k,
        final_k=context_k
    )

    context_parts = []
    sources = []

    for idx, score in results:

        chunk = multi_chunks[idx]

        context_parts.append(
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

        source_info = (chunk["source"], chunk["page"])

        if source_info not in sources:
            sources.append(source_info)

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant answering questions from scientific documents. "
                "Use only the provided document context. "
                "Carefully read all provided context before answering. "
                "Give a concise and factual answer."
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
            max_new_tokens=150,
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

    return answer, sources

question = "What is Self-RAG?"

answer, sources = generate_multi_rag_answer(question)

print("\nANSWER:")
print(answer)

print("\nSOURCES:")
for source, page in sources:
    print(f"{source} — Page {page}")