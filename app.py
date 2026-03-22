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

# --- 1. Page Configuration & Editorial CSS ---
st.set_page_config(page_title="NextRead", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Lato:wght@300;400;700&display=swap');

    /* ── Palette ──────────────────────────────────────
       Background : #F5F1E8  (warm off-white)
       Surface    : #FFFFFF  (white cards/inputs)
       Text       : #1A1A1A  (near black)
       Secondary  : #555555
       Muted      : #999999
       Border     : #E8E4DC
       Accent     : #5A7D6A  (literary green)
       AccentDark : #4A6B59  (hover state)
    ────────────────────────────────────────────────── */

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Lato', sans-serif; }
    .stApp { background-color: #F5F1E8; color: #1A1A1A; }

    /* ── Hide chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 4rem;
        max-width: 1400px;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    /* ── Masthead ── */
    .masthead {
        text-align: center;
        padding: 2rem 0 1.25rem 0;
        border-bottom: 1px solid #E8E4DC;
        margin-bottom: 2rem;
    }
    .masthead-eyebrow {
        font-family: 'Lato', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #5A7D6A;
        margin-bottom: 0.75rem;
    }
    .masthead-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 5.5rem;
        font-weight: 700;
        color: #0F0F0F;
        line-height: 1;
        letter-spacing: -0.02em;
        margin-bottom: 0.85rem;
    }
    .masthead-subtitle {
        font-family: 'Lato', sans-serif;
        font-size: 0.9rem;
        font-weight: 300;
        color: #777777;
        letter-spacing: 0.02em;
    }

    /* ── Search label ── */
    .search-label {
        font-family: 'Lato', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #999999;
        margin-bottom: 0.5rem;
    }

    /* ── Search input ── */
    .stTextInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
        border: 1px solid #E8E4DC !important;
        border-radius: 4px !important;
        padding: 1rem 1.25rem !important;
        font-size: 1.15rem !important;
        font-family: 'Lato', sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: 0.02em !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #5A7D6A !important;
        box-shadow: 0 0 0 3px rgba(90, 125, 106, 0.12) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #BBBBBB !important;
        font-style: italic !important;
    }

    /* ── Filter label ── */
    .filter-label {
        font-family: 'Lato', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #AAAAAA;
        margin-bottom: 0.35rem;
    }

    /* ── Filter selects ── */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E8E4DC !important;
        border-radius: 4px !important;
        color: #333333 !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stSelectbox"] > div > div:hover {
        border-color: #5A7D6A !important;
    }
    div[data-testid="stSelectbox"] svg { color: #999999 !important; }

    /* ── Slider — accent green ── */
    [data-testid="stSlider"] div[role="slider"] {
        background-color: #5A7D6A !important;
        border-color: #5A7D6A !important;
    }
    [data-testid="stSlider"] > div > div > div > div {
        background-color: #5A7D6A !important;
    }
    .stSlider > div > div > div > div:nth-child(1) { background: #E8E4DC !important; }
    .stSlider > div > div > div > div:nth-child(2) { background: #5A7D6A !important; }
    [data-baseweb="slider"] [data-baseweb="thumb"] {
        background-color: #5A7D6A !important;
        border-color: #5A7D6A !important;
    }
    [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSlider"] div[data-baseweb="typo-label"],
    [data-testid="stSlider"] p,
    [data-testid="stSlider"] span {
        color: #5A7D6A !important;
        font-size: 0.9rem !important;
    }

    /* ── Search button ── */
    div[data-testid="stButton"] > button {
        font-family: 'Lato', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.28em !important;
        text-transform: uppercase !important;
        background-color: #3D2B1F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.75rem 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(61, 43, 31, 0.2) !important;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #2A1E15 !important;
        box-shadow: 0 6px 14px rgba(61, 43, 31, 0.25) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Divider & results header ── */
    .editorial-divider {
        border: none;
        border-top: 1px solid #E8E4DC;
        margin: 0.75rem 0 0.5rem 0;
    }
    .results-header {
        font-family: 'Lato', sans-serif;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: #AAAAAA;
        margin-bottom: 0.75rem;
    }

    /* ── Book entry — white card with shadow ── */
    .book-entry {
        display: grid;
        grid-template-columns: 200px 1fr;
        gap: 0 2rem;
        align-items: start;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    .book-entry:hover {
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.09);
        transform: translateY(-2px);
    }
    .book-cover img {
        width: 200px;
        height: 300px;
        object-fit: contain;
        background-color: #F5F1E8;
        border-radius: 4px;
        border: 1px solid #E8E4DC;
        display: block;
    }
    .book-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F0F0F;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }
    .book-author {
        font-family: 'Lato', sans-serif;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #5A7D6A;
        margin-bottom: 1.2rem;
    }
    .book-hook {
        font-family: 'Lato', sans-serif;
        font-size: 1.05rem;
        font-weight: 300;
        color: #444444;
        line-height: 1.8;
        margin-bottom: 1.2rem;
    }
    .book-meta-line {
        font-family: 'Lato', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
        color: #AAAAAA;
        letter-spacing: 0.04em;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.1rem;
    }
    .book-meta-line span { margin-right: 1.25rem; }
    .stars-filled { color: #C8960C; font-size: 1.1rem; }
    .stars-empty  { color: #D8D8D8; font-size: 1.1rem; }

    /* ── Amazon link hover ── */
    a.amazon-link {
        font-family: 'Lato', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4A6B59;
        text-decoration: none;
        border-bottom: 1.5px solid transparent;
        padding-bottom: 2px;
        transition: border-color 0.2s ease, color 0.2s ease;
    }
    a.amazon-link:hover {
        border-bottom-color: #4A6B59;
        color: #3A5A49;
    }

    /* ── Alert / warning ── */
    div[data-testid="stAlert"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E8E4DC !important;
        border-radius: 4px !important;
        color: #333333 !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stAlert"] p { color: #333333 !important; }
    div[data-testid="stAlert"] svg { color: #5A7D6A !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #5A7D6A !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. File paths ---
DB_FILE = 'massive_clean_books.parquet'
FAISS_FILE = 'books_semantic_index.faiss'
MAP_FILE = 'vector_id_mapping.parquet'

# --- Auto-download large files from HuggingFace Hub if not present -----------
# This runs on HuggingFace Spaces where the data files aren't in the repo.
# Locally, if files already exist they are skipped instantly.

HF_DATASET_REPO = "atp1111/nextread-data"

def download_data_files():
    try:
        from huggingface_hub import hf_hub_download
        files = {
            DB_FILE:    "massive_clean_books.parquet",
            FAISS_FILE: "books_optimized.faiss",
            MAP_FILE:   "vector_id_mapping.parquet",
        }
        for local_path, hf_filename in files.items():
            if not os.path.exists(local_path):
                with st.spinner(f"Downloading {hf_filename} from HuggingFace Hub..."):
                    hf_hub_download(
                        repo_id=HF_DATASET_REPO,
                        filename=hf_filename,
                        repo_type="dataset",
                        local_dir="."
                    )
    except Exception as e:
        st.error(f"🚨 Failed to download data files: {e}")
        st.stop()

download_data_files()

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
    fallback_url = "https://placehold.co/200x300/F5F1E8/AAAAAA/png?text=No+Cover"
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
    if len(query.split()) >= 6:
        return query  # Long enough, no expansion needed
    query_lower = query.lower()
    expansions = [exp for kw, exp in QUERY_EXPANSIONS.items() if kw in query_lower]
    if expansions:
        return query + " — " + " ".join(expansions)
    return query


# --- 8. Main UI ---------------------------------------------------------------

# Masthead
st.markdown("""
    <div class="masthead">
        <div class="masthead-title">NextRead</div>
        <div class="masthead-subtitle">Your next great book is in here somewhere.</div>
    </div>
""", unsafe_allow_html=True)

# ── Search bar ────────────────────────────────────────────────────────────────
st.markdown('<div class="search-label">What are you looking for?</div>', unsafe_allow_html=True)
query = st.text_input(
    "",
    placeholder="A quiet novel about grief and memory set in rural Japan...",
    label_visibility="collapsed"
)

# ── Inline pill filters ───────────────────────────────────────────────────────
col_era, col_rating, col_pages = st.columns([2, 1.2, 2])

with col_era:
    st.markdown('<div class="filter-label">Era</div>', unsafe_allow_html=True)
    era_filter = st.selectbox(
        "era",
        ["Any", "Pre-1900", "20th Century", "Contemporary"],
        label_visibility="collapsed"
    )

with col_rating:
    st.markdown('<div class="filter-label">Min Rating</div>', unsafe_allow_html=True)
    prestige_filter = st.selectbox(
        "rating",
        ["Any", "4.0+"],
        label_visibility="collapsed"
    )

with col_pages:
    st.markdown('<div class="filter-label">Max Pages</div>', unsafe_allow_html=True)
    pages_limit = st.slider(
        "pages",
        min_value=100,
        max_value=2000,
        value=800,
        step=50,
        label_visibility="collapsed"
    )

# ── Lambda fixed internally ───────────────────────────────────────────────────
lambda_val = 0.7

# Defaults
if not era_filter:      era_filter      = "Any"
if not prestige_filter: prestige_filter = "Any"

# ── Button centered using a single middle column ──────────────────────────────
_, btn_col, _ = st.columns([2, 1.2, 2])
with btn_col:
    search_clicked = st.button("Find My Next Book")

# ── Search logic ──────────────────────────────────────────────────────────────
if search_clicked and query and len(query.strip()) > 3:
    pass  # handled below

if search_clicked and (not query or len(query.strip()) <= 3):
    st.markdown(
        '<div style="background:#F0F0EE; border:1px solid #E2E2E0; border-radius:0; '
        'padding:0.75rem 1rem; font-family:\'Lato\',sans-serif; font-size:0.85rem; '
        'color:#333333; margin-top:1rem;">Please describe what you\'re looking for before searching.</div>',
        unsafe_allow_html=True
    )

if search_clicked and query and len(query.strip()) > 3:

    # Custom loading display
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
        <div style="text-align:center; padding:3rem 0;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:1.5rem;
                        font-weight:400; font-style:italic; color:#5A7D6A;
                        animation:pulse 1.8s ease-in-out infinite;">
                Searching 901,182 books...
            </div>
            <div style="font-family:'Lato',sans-serif; font-size:0.65rem;
                        letter-spacing:0.25em; text-transform:uppercase;
                        color:#BBBBBB; margin-top:0.75rem;">
                Reading between the lines
            </div>
        </div>
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
        </style>
    """, unsafe_allow_html=True)

    with st.spinner(""):

        # Phase 1: Short Query Expansion
        expanded_query = expand_short_query(query.strip())
        was_expanded   = expanded_query != query.strip()

        # Phase 2: FAISS Bi-Encoder Retrieval
        query_vector = bi_encoder.encode(
            [expanded_query],
            normalize_embeddings=True
        ).astype('float32')

        distances, indices = faiss_index.search(query_vector, 5000)
        matched_ids  = id_map_df.iloc[indices[0]]['Id'].tolist()
        id_list_str  = ", ".join(map(str, matched_ids))

        # Phase 3: DuckDB Metadata Filtering
        y_logic = "1=1"
        if era_filter == "Pre-1900":       y_logic = "PublishYear < 1900"
        elif era_filter == "20th Century": y_logic = "PublishYear >= 1900 AND PublishYear <= 2000"
        elif era_filter == "Contemporary": y_logic = "PublishYear > 2000"
        p_logic = "Rating >= 4.0" if prestige_filter == "4.0+" else "Rating >= 3.0"

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
            st.markdown(
                '<div style="background:#F3EEE8; border:1px solid #DDD5CA; border-radius:2px; '
                'padding:0.75rem 1rem; font-family:\'Lato\',sans-serif; font-size:0.85rem; '
                'color:#4A3F38; margin-top:1rem;">No books matched those filters. '
                'Try adjusting the era or page count.</div>',
                unsafe_allow_html=True
            )
        else:
            # Phase 4: Cross-Encoder Reranking
            corpus          = candidates_df['Description'].fillna("").tolist()
            sentence_pairs  = [[query, doc] for doc in corpus]
            semantic_scores = reranker.predict(sentence_pairs)

            scaler        = MinMaxScaler()
            norm_semantic = scaler.fit_transform(semantic_scores.reshape(-1, 1)).flatten()
            norm_rating   = scaler.fit_transform(candidates_df[['Rating']]).flatten()
            candidates_df['Cross_Score'] = (norm_semantic * 0.85) + (norm_rating * 0.15)

            # Phase 5: Author Deduplication
            top50_df   = candidates_df.sort_values('Cross_Score', ascending=False).head(50)
            deduped_df = deduplicate_by_author(top50_df, 'Cross_Score').head(30).reset_index(drop=True)

            # Phase 6: MMR Diversity Reranking
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

            # Phase 7: Results Display — pure HTML, no Streamlit widgets inside cards
            loading_placeholder.empty()
            st.markdown('<hr class="editorial-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="results-header">5 Recommendations · searched 901,182 books</div>',
                unsafe_allow_html=True
            )

            cards_html = ""
            total = len(final_df)
            for rank, (_, row) in enumerate(final_df.iterrows(), 1):
                title     = str(row['Name']).replace('"', '&quot;').replace('<', '&lt;')
                author    = str(row['Authors']).replace('"', '&quot;').replace('<', '&lt;')
                full_desc = str(row['Description'])
                rating    = round(float(row['Rating']), 1)
                pages     = int(row['Pages'])
                year      = int(row['PublishYear'])

                # 2-sentence hook
                sentences = re.split(r'(?<=[.!?]) +', full_desc)
                hook = " ".join(sentences[:3]) if sentences else full_desc
                if len(hook) > 480:
                    hook = hook[:480] + "..."
                hook = hook.replace('<', '&lt;').replace('>', '&gt;')

                # Stars — built with join so zero whitespace between spans
                full_stars  = int(rating // 1)
                empty_stars = 5 - full_stars
                filled = ''.join(['<span style="color:#C8960C;font-size:1.1rem;line-height:1;margin:0;padding:0;">★</span>' for _ in range(full_stars)])
                empty  = ''.join(['<span style="color:#D8D8D8;font-size:1.1rem;line-height:1;margin:0;padding:0;">☆</span>' for _ in range(empty_stars)])
                stars_html = f'<span style="display:inline-flex;gap:1px;align-items:center;">{filled}{empty}</span>'

                # Cover image URL
                cover_url = get_cover_url(str(row['Name']), str(row['Authors']))

                # Amazon URL
                amazon_url = f"https://www.amazon.com/s?k={urllib.parse.quote(title + ' ' + author)}"

                cards_html += f"""
                <div class="book-entry">
                    <div class="book-cover">
                        <img src="{cover_url}"
                             onerror="this.src='https://placehold.co/200x300/F5F1E8/AAAAAA/png?text=No+Cover'"/>
                    </div>
                    <div class="book-body">
                        <div class="book-title">{title}</div>
                        <div class="book-author">{author}</div>
                        <div class="book-hook">{hook}</div>
                        <div class="book-meta-line">
                            <span style="white-space:nowrap;">{stars_html} &nbsp;{rating}</span>
                            <span>{pages} pages</span>
                            <span>{year}</span>
                        </div>
                        <a href="{amazon_url}" target="_blank" class="amazon-link">
                            Find on Amazon ↗
                        </a>
                    </div>
                </div>
                """

            st.markdown(cards_html, unsafe_allow_html=True)
