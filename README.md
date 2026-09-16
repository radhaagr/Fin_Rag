# FinRAG / FinDER Retrieval Benchmark

This repository is a finance-focused retrieval benchmarking workspace for evaluating chunking and retrieval strategies on local FinanceRAG / FinDER-style datasets. It is designed for research and experimentation rather than a packaged production application.

The main goal is to compare how different document chunking strategies and retriever backends affect retrieval quality for financial question-answering tasks.

## Project overview

The project includes:

- local dataset files under `finrag_data/` for multiple finance retrieval benchmarks
- notebook-based experiments for splitter and retriever sweeps
- dense, sparse, and hybrid retrieval baselines
- evaluation metrics such as Recall@k and NDCG@k
- CSV outputs saved in the `results/` directory

The repository is especially focused on:

- chunking strategies (`char`, `recursive_char`, `token`)
- embedding-based retrievers (`cosine_numpy`, `faiss_flat`, `faiss_hnsw`, `chroma`)
- sparse retrieval (`bm25`)
- hybrid retrieval using reciprocal rank fusion (`hybrid_rrf`)

## Repository structure

```text
Fin_Rag/
├── finrag_data/                     # local corpora, queries, and qrels for finance datasets
│   ├── ConvFinQA_qrels.tsv
│   ├── FinanceBench_qrels.tsv
│   ├── FinDER_qrels.tsv
│   ├── FinQA_qrels.tsv
│   ├── FinQABench_qrels.tsv
│   ├── MultiHeirtt_qrels.tsv
│   ├── TATQA_qrels.tsv
│   ├── ..._corpus.jsonl/
│   └── ..._queries.jsonl/
├── notebooks/
│   ├── Fin_rag_Finder.ipynb
│   ├── Fin_rag_Finder2.ipynb
│   ├── Finrag_Finder_Faiss.ipynb
│   ├── Finrag_Finder_HybridRetriever.ipynb
│   ├── Retrievers.py
│   ├── Splitters.py
│   └── ...
├── results/
│   ├── finder_hybridRetriever_results.csv
│   ├── finder_splitter_results.csv
│   └── finder_sweep_results.csv
├── Retriever.py
├── requirements.txt
├── LICENSE (if added later)
└── README.md
```

## Dataset and evaluation setup

The project loads BEIR-style retrieval assets:

- corpus files in JSONL format
- query files in JSONL format
- relevance judgments in TSV or CSV-based qrels

The benchmarking flow typically does the following:

1. loads a corpus and query set
2. chunks text using selected splitter settings
3. embeds chunks and queries with a SentenceTransformer model
4. retrieves top-k documents with a retriever
5. evaluates retrieval outcomes using qrels
6. saves comparison results to CSV

## Technologies used

- Python
- SentenceTransformers
- NumPy
- Pandas
- scikit-learn
- FAISS
- ChromaDB
- Rank-BM25
- LangChain text splitters
- Jupyter notebooks

## Requirements

Python 3.10+ is recommended.

Install the core dependencies:

```bash
pip install -r requirements.txt
```

For the full retrieval benchmark, including FAISS, Chroma, BM25, and advanced splitting:

```bash
pip install langchain-text-splitters faiss-cpu chromadb rank_bm25 sentence-transformers
```

## Quick start

### Option 1: Open the notebook workflow

Open the notebook in the `notebooks/` folder, especially:

- `notebooks/Finrag_Finder_HybridRetriever.ipynb`
- `notebooks/Finrag_Finder_Faiss.ipynb`
- `notebooks/Fin_rag_Finder.ipynb`

Run the cells in order to:

- load the dataset
- create chunks
- build embeddings
- run retrievers
- evaluate results
- export CSV benchmark summaries

### Option 2: Run the retrieval logic in Python

You can also use the retriever and splitter implementations from the notebook/helpers:

```python
from notebooks.Retrievers import get_retriever
from notebooks.Splitters import get_splitter

retriever = get_retriever("hybrid_rrf")
splitter = get_splitter("token", chunk_size=256, overlap=40)
```

## Example experiment patterns

The project compares combinations such as:

- `char` vs `token` chunkers
- chunk sizes like 128, 256, 512
- overlap settings like 20, 40, 64, 100
- retrievers like `faiss_flat`, `faiss_hnsw`, `bm25`, `hybrid_rrf`
- evaluation at multiple `k` values such as 1, 3, 5, 10, 20, 50

This makes it useful for testing trade-offs between:

- retrieval recall
- computational speed
- chunk granularity
- dense vs sparse performance

## Metrics

The notebooks evaluate retrieval using standard IR metrics, including:

- Recall@k
- NDCG@k

These scores are computed against the local qrels files and summarized in output CSV files.

## Output files

Benchmark results are saved under `results/` as CSV files, including:

- `finder_hybridRetriever_results.csv`
- `finder_splitter_results.csv`
- `finder_sweep_results.csv`

These outputs can be used for downstream analysis or comparison against different retrieval settings.



This repository is a practical benchmarking setup for evaluating retrieval quality on financial corpora. It emphasizes modularity, experimentation, and comparison across chunking and retrieval strategies, making it a strong foundation for finance-domain RAG and retrieval research.
