# 📚 NextRead — Semantic Book Recommendation Engine

> Describe a world, a plot, or a feeling. NextRead finds your next book.

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/atp1111/nextread)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-IndexIVFPQ-green)](https://github.com/facebookresearch/faiss)

---

## What It Does

NextRead is a production-grade semantic book recommendation system that searches a corpus of **901,182 books** using natural language. Unlike keyword-based search, it understands that *"a gritty murder investigation"* and *"detective crime noir"* mean the same thing.

Type a description of what you're looking for. The engine finds it.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 1: Short Query Expansion         │
│  Expands vague queries (<6 words) with  │
│  genre-aware context terms              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Phase 2: Bi-Encoder Retrieval          │
│  all-MiniLM-L6-v2 encodes query into    │
│  384-dim semantic vector                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Phase 3: FAISS IVFPQ Search            │
│  Searches 901k vectors in <100ms        │
│  Returns top 5,000 semantic candidates  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Phase 4: DuckDB Metadata Filtering     │
│  Hard filters: pages, era, rating       │
│  Narrows to ~300 quality candidates     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Phase 5: Cross-Encoder Reranking       │
│  ms-marco-MiniLM-L-6-v2 scores each     │
│  candidate pair with full attention     │
│  85% semantic score + 15% rating boost  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Phase 6: Author Deduplication          │
│  Ensures no author appears twice        │
│  Prevents series flooding               │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Phase 7: MMR Diversity Reranking       │
│  Maximal Marginal Relevance balances    │
│  relevance vs result diversity          │
│  User-tunable λ parameter (0.5–1.0)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
            Top 5 Results
```

---

## Technical Highlights

### Semantic Search at Scale
- **901,182 books** indexed using `all-MiniLM-L6-v2` sentence embeddings (384 dimensions)
- **Metadata stitching** — title, author, rating sentiment, and description are combined before encoding, giving the model richer signal per book
- **FAISS IndexIVFPQ** with Product Quantization reduces the index from **1.3GB → 91MB (93% reduction)** with under 5% recall loss

### Two-Stage Retrieval Pipeline
A bi-encoder + cross-encoder architecture commonly used in production search systems:
- **Bi-encoder** (FAISS): Fast approximate search across all 901k books in milliseconds. Trades some precision for speed.
- **Cross-encoder** (ms-marco-MiniLM): Precise full-attention reranking on the top 300 candidates. Slow but accurate — only runs on a small subset.

### Maximal Marginal Relevance (MMR)
Standard recommendation systems return the top-N most similar results, which often means 5 near-identical books. MMR solves this with:

```
MMR(dᵢ) = λ · Relevance(dᵢ, q) − (1−λ) · max Similarity(dᵢ, dⱼ)
                                              dⱼ ∈ Selected
```

Where λ is user-controllable via a sidebar slider, allowing real-time tuning between focused relevance and diverse discovery.

### Data Pipeline
- Raw CSVs processed with **DuckDB** (chosen over pandas for memory efficiency at 900k rows)
- HTML tag stripping, outlier removal, and deduplication via `QUALIFY ROW_NUMBER()`
- Output stored as **Parquet with ZSTD compression** for fast columnar reads

---

## Project Structure

```
nextread/
├── app.py                 # Streamlit application — full 7-phase pipeline
├── build_vector_db.py     # One-time: encodes 901k books → FAISS index
├── optimize_index.py      # One-time: FlatIP → IVFPQ compression
├── requirements.txt
└── .gitignore
```

---

## Run Locally

**1. Clone and install dependencies**
```bash
git clone https://github.com/atp1111/nextread
cd nextread
pip install -r requirements.txt

# Install PyTorch with CUDA (RTX 3070 or similar)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. Download large files from HuggingFace Hub**
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download data files
python -c "
from huggingface_hub import hf_hub_download
files = ['massive_clean_books.parquet', 'books_optimized.faiss', 'vector_id_mapping.parquet']
for f in files:
    hf_hub_download(repo_id='atp1111/nextread-data', filename=f, local_dir='.')
"
```

**3. Launch**
```bash
streamlit run app.py
```

---

## Known Limitations & Future Work

| Limitation | Planned Fix |
|---|---|
| 85/15 score weighting is hand-tuned | A/B test with user feedback signals |
| Short query expansion uses fixed keyword map | Replace with LLM query rewriting |
| No user history | Collect implicit feedback for hybrid model |
| Cross-encoder runs on CPU in deployment | Upgrade to GPU-enabled HuggingFace Space |

---

## Stack

| Component | Technology |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Reranking Model | `ms-marco-MiniLM-L-6-v2` (Cross-Encoder) |
| Vector Search | FAISS IndexIVFPQ |
| Metadata Filtering | DuckDB |
| Data Storage | Apache Parquet (ZSTD) |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces |

---

## Dataset

Based on a Kaggle books dataset — 901,182 books with titles, authors, descriptions, ratings, page counts, and publication years. Cleaned and processed using DuckDB.
