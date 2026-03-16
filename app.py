import streamlit as st
import duckdb
import requests
import urllib.parse
import re
import os
import torch
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. Page Configuration & Dark Tech CSS ---
st.set_page_config(page_title="NextRead", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', -apple-system, sans-serif; }
    .main-header { font-size: 4rem; font-weight: 900; color: #ffffff; text-align: center; margin-bottom: 0px; letter-spacing: -2px; }
    .sub-header { font-size: 1.2rem; color: #8b949e; text-align: center; margin-bottom: 40px; font-weight: 400; }
    .stTextInput>div>div>input { background-color: #161b22; color: #ffffff; border: 1px solid #30363d; border-radius: 8px; padding: 16px; font-size: 1.2rem; transition: all 0.2s ease-in-out; }
    .stTextInput>div>div>input:focus { border-color: #58a6ff; box-shadow: 0 0 0 1px #58a6ff; }
    .book-container { background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 20px; transition: transform 0.2s ease, border-color 0.2s ease; }
    .book-container:hover { border-color: #58a6ff; transform: translateY(-2px); }
    .book-title { font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-bottom: 4px; line-height: 1.2; }
    .book-author { font-size: 0.95rem; color: #58a6ff; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }
    .book-hook { font-size: 1.1rem; color: #8b949e; line-height: 1.6; margin-bottom: 0px; }
    .match-score { font-size: 0.8rem; font-weight: 700; color: #0d1117; background-color: #58a6ff; padding: 4px 10px; border-radius: 20px; display: inline-block; margin-bottom: 12px; }
    .diversity-badge { font-size: 0.75rem; font-weight: 600; color: #3fb950; background-color: #0d2119; border: 1px solid #3fb950; padding: 3px 8px; border-radius: 20px; display: inline-block; margin-left: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. File paths ---
DB_FILE = 'massive_clean_books.parquet'
FAISS_FILE = 'books_optimized.faiss'
MAP_FILE = 'vector_id_mapping.parquet'

for f in [DB_FILE, FAISS_FILE, MAP_FILE]:
    if not os.path.exists(f):
        st.error(f"🚨 Required file '{f}' not found. Make sure all pipeline files are present.")
        st.stop()

# --- 3. Load AI Engine (GPU-aware, cached for session) -----------------------

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource(show_spinner="Booting NextRead Engine...")
def load_ai_engine():
    bi_encoder    = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE)
    index         = faiss.read_index(FAISS_FILE)
    id_map        = pd.read_parquet(MAP_FILE)
    return bi_encoder, cross_encoder, index, id_map

bi_encoder, reranker, faiss_index, id_map_df = load_ai_engine()

# --- 4. Cover Fetcher ---------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def get_cover_url(title, author):
    fallback_url = "https://placehold.co/150x220/161b22/58a6ff/png?text=Cover\\nUnavailable"
    clean_title = re.sub(r'\(.*?\)', '', str(title)).strip()
    try:
        url = f"https://openlibrary.org/search.json?q={urllib.parse.quote(f'{clean_title} {author}')}&limit=1"
        res = requests.get(url, timeout=2).json()
        if res.get("numFound", 0) > 0 and "cover_i" in res["docs"][0]:
            return f"https://covers.openlibrary.org/b/id/{res['docs'][0]['cover_i']}-L.jpg"
    except:
        pass
    return fallback_url

# --- 5. MMR: Maximal Marginal Relevance ---------------------------------------
#
# The problem MMR solves: without it, your top 5 results could all be nearly
# identical books (e.g. Harry Potter 1-5 for a "magic school" query).
#
# How it works:
#   - Start with the single most relevant candidate
#   - For each next pick, score every remaining candidate as:
#       MMR = λ * relevance_score  -  (1-λ) * max_similarity_to_already_picked
#   - λ (lambda) controls the tradeoff:
#       λ=1.0  →  pure relevance (no diversity, same as without MMR)
#       λ=0.7  →  recommended: leans relevant but forces variety
#       λ=0.5  →  equal balance between relevance and diversity

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    relevance_scores: np.ndarray,
    top_k: int = 5,
    lambda_param: float = 0.7
) -> list:
    """
    Returns indices (into candidate_embeddings) of the top_k most diverse
    yet relevant candidates using the MMR algorithm.
    """
    # Normalize so dot product == cosine similarity
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = candidate_embeddings / norms

    selected  = []
    remaining = list(range(len(candidate_embeddings)))

    for _ in range(min(top_k, len(remaining))):
        if not selected:
            # First pick: the most relevant candidate, no diversity penalty yet
            best_idx = int(np.argmax(relevance_scores))
        else:
            selected_vecs  = normed[selected]    # shape: (k, dim)
            remaining_vecs = normed[remaining]   # shape: (r, dim)

            # Cosine similarity between each remaining book and every selected book
            sim_matrix = remaining_vecs @ selected_vecs.T  # shape: (r, k)

            # Worst-case similarity: how close is each remaining book to
            # the most similar thing we've already picked
            max_sim_to_selected = sim_matrix.max(axis=1)   # shape: (r,)

            remaining_relevance = relevance_scores[remaining]
            mmr_scores = (
                (lambda_param * remaining_relevance) -
                ((1 - lambda_param) * max_sim_to_selected)
            )
            best_idx = remaining[int(np.argmax(mmr_scores))]

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# --- 6. Series / Author Deduplication ----------------------------------------
#
# Keeps only the highest-scored book per author so we never return
# e.g. three books from the same series as three of our five results.

def deduplicate_by_author(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        df.sort_values(by=score_col, ascending=False)
          .drop_duplicates(subset=['Authors'], keep='first')
    )


# --- 7. Short Query Expansion -------------------------------------------------
#
# Bi-encoders work poorly on very short queries like "sad romance" because
# there's too little signal. We expand them by appending genre context terms.

QUERY_EXPANSIONS = {
    "romance":    "love story emotional heartbreak relationship",
    "thriller":   "suspense tension dangerous mystery",
    "fantasy":    "magic world-building adventure epic",
    "sci-fi":     "futuristic technology space exploration",
    "horror":     "frightening dark psychological fear",
    "mystery":    "detective investigation crime clues",
    "historical": "period setting based on real events past era",
    "adventure":  "journey quest action exploration",
    "sad":        "emotional grief loss melancholy",
    "funny":      "humour comedy lighthearted witty",
}

def expand_short_query(query: str) -> str:
    """Expand short queries with genre-related context terms."""
    if len(query.split()) >= 3:
        return query  # Long enough, no expansion needed
    query_lower = query.lower()
    expansions = [exp for kw, exp in QUERY_EXPANSIONS.items() if kw in query_lower]
    if expansions:
        return query + " — " + " ".join(expansions)
    return query


# --- 8. Sidebar Filters -------------------------------------------------------

with st.sidebar:
    st.markdown("### Search Parameters")
    pages_limit     = st.slider("Maximum Length (Pages)", 100, 2000, 500, step=50)
    era_filter      = st.radio("Timeline", ["Any Era", "Classics (Before 1900)", "20th Century", "Contemporary (2000+)"])
    prestige_filter = st.radio("Prestige", ["Any Rating", "Critically Acclaimed (4.0+)"])

    st.markdown("---")
    st.markdown("### Recommendation Style")
    lambda_val = st.slider(
        "Relevance ↔ Diversity",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Higher = more focused on your exact query. Lower = more varied recommendations."
    )
    st.caption(
        f"{'🎯 Focused' if lambda_val >= 0.85 else '⚖️ Balanced' if lambda_val >= 0.65 else '🌈 Diverse'}"
    )
    st.markdown("---")
    st.caption(f"Engine: **{'GPU ⚡' if DEVICE == 'cuda' else 'CPU'}**")


# --- 9. Main UI ---------------------------------------------------------------

st.markdown('<div class="main-header">NextRead</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Describe a world, a plot, or a concept. The engine will find it.</div>', unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="e.g., A gritty cyberpunk detective hunting a rogue AI...",
    label_visibility="collapsed"
)

if query and len(query.strip()) >= 3:
    with st.spinner("Searching millions of semantic dimensions..."):

        # ── Phase 1: Short Query Expansion ────────────────────────────────────
        expanded_query = expand_short_query(query.strip())
        was_expanded   = expanded_query != query.strip()

        # ── Phase 2: FAISS Bi-Encoder Retrieval ───────────────────────────────
        query_vector = bi_encoder.encode(
            [expanded_query],
            normalize_embeddings=True
        ).astype('float32')

        distances, indices = faiss_index.search(query_vector, 5000)
        matched_ids  = id_map_df.iloc[indices[0]]['Id'].tolist()
        id_list_str  = ", ".join(map(str, matched_ids))

        # ── Phase 3: DuckDB Metadata Filtering ────────────────────────────────
        y_logic = "1=1"
        if "Classics" in era_filter:       y_logic = "PublishYear < 1900"
        elif "20th Century" in era_filter: y_logic = "PublishYear >= 1900 AND PublishYear <= 2000"
        elif "Contemporary" in era_filter: y_logic = "PublishYear > 2000"
        p_logic = "Rating >= 4.0" if "Acclaimed" in prestige_filter else "Rating >= 3.0"

        sql = f"""
            SELECT Id, Name, Authors, Pages, PublishYear, Rating, Description
            FROM '{DB_FILE}'
            WHERE Id IN ({id_list_str})
            AND Pages <= {pages_limit}
            AND {y_logic}
            AND {p_logic}
            LIMIT 300
        """
        candidates_df = duckdb.execute(sql).df()

        if candidates_df.empty:
            st.warning("No books matched those filters. Try relaxing the page count or era.")
        else:
            # ── Phase 4: Cross-Encoder Reranking ──────────────────────────────
            corpus         = candidates_df['Description'].fillna("").tolist()
            sentence_pairs = [[query, doc] for doc in corpus]
            semantic_scores = reranker.predict(sentence_pairs)

            scaler = MinMaxScaler()
            norm_semantic = scaler.fit_transform(semantic_scores.reshape(-1, 1)).flatten()
            norm_rating   = scaler.fit_transform(candidates_df[['Rating']]).flatten()
            candidates_df['Cross_Score'] = (norm_semantic * 0.85) + (norm_rating * 0.15)

            # ── Phase 5: Author Deduplication ─────────────────────────────────
            # Take top 50 first so we don't accidentally drop a great book
            # just because a worse one by the same author appeared earlier.
            top50_df   = candidates_df.sort_values('Cross_Score', ascending=False).head(50)
            deduped_df = deduplicate_by_author(top50_df, 'Cross_Score').head(30).reset_index(drop=True)

            # ── Phase 6: MMR Diversity Reranking ──────────────────────────────
            candidate_descriptions = deduped_df['Description'].fillna("").tolist()
            candidate_embeddings   = bi_encoder.encode(
                candidate_descriptions,
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype('float32')

            mmr_indices = maximal_marginal_relevance(
                query_embedding      = query_vector,
                candidate_embeddings = candidate_embeddings,
                relevance_scores     = deduped_df['Cross_Score'].values,
                top_k                = 5,
                lambda_param         = lambda_val
            )
            final_df = deduped_df.iloc[mmr_indices].copy()

            # ── Phase 7: Results Display ───────────────────────────────────────
            st.markdown("---")
            if was_expanded:
                st.caption(f"🔍 Query expanded for better results: *\"{expanded_query}\"*")

            for rank, (_, row) in enumerate(final_df.iterrows(), 1):
                title     = str(row['Name'])
                author    = str(row['Authors'])
                full_desc = str(row['Description'])
                score     = min(row['Cross_Score'] * 100, 99.9)

                sentences = re.split(r'(?<=[.!?]) +', full_desc)
                hook = " ".join(sentences[:2]) if sentences else full_desc
                if len(hook) > 300:
                    hook = hook[:300] + "..."

                with st.container():
                    st.markdown('<div class="book-container">', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 6])

                    with col1:
                        st.image(get_cover_url(title, author), width=120)

                    with col2:
                        diversity_badge = '<span class="diversity-badge">✦ MMR Pick</span>' if rank > 1 else ''
                        st.markdown(
                            f'<div class="match-score">{score:.1f}% Context Match</div>{diversity_badge}',
                            unsafe_allow_html=True
                        )
                        st.markdown(f'<div class="book-title">{title}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="book-author">BY {author}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="book-hook">{hook}</div>', unsafe_allow_html=True)

                        with st.expander("View full description & stats"):
                            st.markdown(
                                f"**Rating:** {row['Rating']} &nbsp;|&nbsp; "
                                f"**Length:** {row['Pages']} pages &nbsp;|&nbsp; "
                                f"**Published:** {row['PublishYear']}"
                            )
                            st.write(full_desc)
                            st.link_button(
                                "View on Amazon",
                                f"https://www.amazon.com/s?k={urllib.parse.quote(f'{title} book {author}')}"
                            )

                    st.markdown('</div>', unsafe_allow_html=True)