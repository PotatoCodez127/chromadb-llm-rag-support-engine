# E-Commerce Customer Support Resolution Engine (RAG)

[![CI Pipeline](https://github.com/potatocodez127/chromadb-llm-rag-support-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/potatocodez127/chromadb-llm-rag-support-engine/actions/workflows/ci.yml)

## Executive Summary
`chromadb-llm-rag-support-engine` is an institutional-grade, locally persistent Retrieval-Augmented Generation (RAG) system designed to act as a deterministic customer support agent. By grounding generative LLM processing loops completely within a localized vector space, the system guarantees zero-shot hallucination defense, enforcing absolute compliance with proprietary enterprise policy matrices.

## Architectural Ingenuity
* **Deterministic Information Retrieval**: Leverages a persistent storage layer powered by ChromaDB paired with `all-MiniLM-L6-v2` embeddings to perform low-latency semantic chunk searching across corporate policy documents.
* **Mock-Isolated Test Grid**: Implements comprehensive test suites utilizing execution intercept techniques (`unittest.mock.patch`) to validate extraction, prompt-splicing, and routing logic without external network dependencies, high-compute downloads, or live API usage.
* **Strict Production Parity**: Uses a layer-cached Docker blueprint anchored to `python:3.11-slim` with `build-essential` compilation optimization layers and immutable UTC timezone enforcement to guarantee exact state execution profiles from development to distributed cloud environments.

## System Topology & Pipeline Layout
1. **Ingestion Pipeline (`ingest.py`)**: Consumes text payloads, segments data into semantic chunks bounded by rule sections, generates numerical vectors via local embedding layers, and writes downstream records into persistent local storage blocks (`chroma_db/`).
2. **Retrieval-Generation Loop (`chat.py`)**: Intercepts real-time queries, matches vector parameters to locate the top 2 highest-ranking context segments, maps extracted components into an absolute "open-book" prompt constraint layer, and streams the validated answer back to the operator interface via Ollama Cloud API execution.

## The Problem Solved
Large Language Models (LLMs) suffer from a "Knowledge Cutoff" and tend to hallucinate when asked about proprietary or highly specific company data. If an ungrounded bot gives a customer the wrong refund policy, the company is liable. This architecture grounds the AI's generation process entirely in a private Vector Database.

## Tech Stack
* **Python 3.10+**
* **ChromaDB:** A lightweight, local vector database to store document embeddings.
* **Sentence-Transformers:** To convert our text into mathematical vectors (embeddings).
* **Ollama Cloud:** For the LLM generation phase (`gemma4:31b-cloud` or similar).

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your API key:
   `OLLAMA_API_KEY=your_api_key_here`

## Usage
1. Run the ingestion script to populate the vector database with company policies:
   `python ingest.py`
2. Run the main chat engine to test queries against the knowledge base:
   `python chat.py`