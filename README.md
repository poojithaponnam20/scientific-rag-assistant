# Scientific Literature RAG Assistant

A multi-document Retrieval-Augmented Generation (RAG) system for querying scientific literature and generating grounded answers with document and page-level citations.

## Overview

This project implements an end-to-end RAG pipeline for scientific PDF documents. Users can ask natural-language questions through a Streamlit interface, retrieve relevant passages from multiple papers, rerank the retrieved passages, and generate answers using a local Hugging Face language model.

The system was developed incrementally with a focus on understanding the full RAG architecture rather than relying on high-level RAG frameworks.

## Features

* Multi-PDF scientific document ingestion
* PDF text extraction using PyMuPDF
* Text cleaning and sentence-aware chunking
* Semantic embeddings using SentenceTransformers
* Dense similarity retrieval
* CrossEncoder reranking
* Local answer generation using Qwen3-1.7B
* Document and page-level source attribution
* Expandable retrieved passages in the Streamlit interface
* Retrieval and generation evaluation

## Architecture

```text
Scientific PDFs
      ↓
Text Extraction
      ↓
Text Cleaning
      ↓
Sentence-Aware Chunking
      ↓
SentenceTransformer Embeddings
      ↓
Dense Similarity Retrieval
      ↓
Top-20 Candidate Passages
      ↓
CrossEncoder Reranking
      ↓
Top-3 Context Passages
      ↓
Qwen3-1.7B
      ↓
Grounded Answer + Source Pages
```

## Retrieval Evaluation

Retrieval was evaluated on a 10-question benchmark based on the original RAG paper.

| Method                                   | Hit@1 | Hit@3 | Hit@5 |
| ---------------------------------------- | ----: | ----: | ----: |
| Dense retrieval                          |   70% |   80% |   80% |
| Dense retrieval + CrossEncoder reranking |   80% |  100% |  100% |

Increasing the initial candidate pool and applying CrossEncoder reranking substantially improved retrieval quality, particularly for Top-3 and Top-5 results.

## Generation Evaluation

Two local Qwen models were compared using deterministic decoding.

| Model      | Result                                               |
| ---------- | ---------------------------------------------------- |
| Qwen3-0.6B | 8/10 fully correct, 1 partially correct, 1 incorrect |
| Qwen3-1.7B | 10/10 substantively correct                          |

The comparison also revealed cases where the correct document context was successfully retrieved but the smaller language model failed to use it correctly, demonstrating that retrieval quality and generation quality represent separate failure points in a RAG system.

## Example

**Question**

> What is Self-RAG?

**Answer**

The system explains Self-RAG using retrieved passages from the Self-RAG paper and provides document-level and page-level source attribution.

**Sources**

```text
Asai_2023_Self_RAG.pdf — Page 3
Asai_2023_Self_RAG.pdf — Page 10
```

## Technology Stack

* Python
* PyMuPDF
* NumPy
* SentenceTransformers
* CrossEncoder
* Hugging Face Transformers
* Qwen3-1.7B
* PyTorch
* Streamlit

## Project Structure

```text
scientific-rag-assistant/
│
├── app.py
├── rag_backend.py
├── experiments.py
├── requirements.txt
├── README.md
│
└── data/
    ├── .gitkeep # Add your own PDFs here
```

`rag_backend.py` contains the cleaned production pipeline, while `experiments.py` contains the experiments and evaluation work used during development.

## Run Locally

### Add Documents

The application automatically discovers all PDF files in the data/ folder and builds the retrieval index when the app starts.

Place one or more scientific PDF files inside the `data/` folder:

```text
data/
├── paper_1.pdf
├── paper_2.pdf
└── ...
```


Create or activate a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Future Improvements

Potential extensions include:

* larger scientific document collections
* persistent vector indexes for larger datasets
* automatic RAG evaluation datasets
* more precise citation attribution at the individual passage level
* hybrid lexical and semantic retrieval
* comparison of additional embedding and generation models
* deployment of the Streamlit application

## Key Findings

The development process highlighted several practical characteristics of RAG systems:

* sentence-aware chunking produces cleaner retrieval units than fixed-character chunking
* semantic similarity alone may retrieve broadly relevant but not question-specific passages
* CrossEncoder reranking improves question-specific retrieval
* increasing the candidate pool gives the reranker more opportunities to recover relevant passages
* query wording can significantly affect retrieval performance
* correct retrieval does not guarantee correct generation
* increasing LLM capacity improved generation quality when the relevant context was already available
