import duckdb
import numpy as np
import pandas as pd
import faiss
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_vector_database():
    print("🚀 Initializing Vector Database Pipeline...")
    
    # 1. Load the pristine dataset
    # 1. Load the pristine dataset
    DB_FILE = 'massive_clean_books.parquet'
    print(f"📦 Loading {DB_FILE} into memory...")
    
    # We pull the Title, Author, AND Rating now
    df = duckdb.execute(f"SELECT Id, Name, Authors, Rating, Description FROM '{DB_FILE}'").df()
    
    # --- THE MAGIC UPGRADE: TEXTUAL RATING INJECTION ---
    print("⭐ Translating numeric ratings into NLP sentiment...")
    import numpy as np
    
    # If a book is rated highly, we inject hype words so the AI learns its prestige
    df['Rating_Text'] = np.where(df['Rating'] >= 4.2, "A highly rated, critically acclaimed book. ",
                        np.where(df['Rating'] >= 3.8, "A well-received popular book. ", ""))
    
    # --- THE GOD-TIER METADATA STITCH ---
    print("🪡 Stitching Metadata into Super-Strings...")
    df['Stitched_Text'] = (
        "Title: " + df['Name'].fillna("Unknown") + ". " + 
        "Author: " + df['Authors'].fillna("Unknown") + ". " + 
        df['Rating_Text'] + 
        "Summary: " + df['Description'].fillna("")
    )
    
    # We pass the Ultimate Stitched Text to the GPU
    descriptions = df['Stitched_Text'].tolist()
    ids = df['Id'].tolist()
    
    print(f"📊 Total Books to Encode: {len(descriptions):,}")
    
    # 2. Load the Bi-Encoder Model (Hardware Agnostic)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧠 Loading all-MiniLM-L6-v2...")
    print(f"⚙️ Compute Device detected: {device.upper()}")
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Batch Encoding Loop
    print("\n⚡ Starting massive GPU encoding job... (Go grab a coffee ☕)")
    batch_size = 256 # Optimized for RTX 3070 8GB VRAM
    all_embeddings = []
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(descriptions), batch_size), desc="Encoding Batches"):
        batch_text = descriptions[i : i + batch_size]
        # encode() returns a numpy array of vectors
        batch_embeddings = model.encode(batch_text, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(batch_embeddings)
        
    # 4. Stack into one giant matrix
    print("\n🧱 Stacking vectors into a single matrix...")
    embedding_matrix = np.vstack(all_embeddings).astype('float32')
    
    # 5. Build and Save the FAISS Index
    print("🗄️ Building the FAISS Vector Index...")
    dimension = embedding_matrix.shape[1] # Should be 384 for MiniLM
    
    # IndexFlatIP uses Inner Product (Cosine Similarity since we normalized)
    index = faiss.IndexFlatIP(dimension) 
    index.add(embedding_matrix)
    
    # Save the index to your hard drive
    faiss.write_index(index, "books_semantic_index.faiss")
    
    # Save the ID mapping so we know which vector belongs to which book
    pd.DataFrame({'Id': ids}).to_parquet('vector_id_mapping.parquet')
    
    end_time = time.time()
    minutes = round((end_time - start_time) / 60, 1)
    print(f"\n🎉 SUCCESS! Entire library encoded and saved in {minutes} minutes.")
    print("Files created: 'books_semantic_index.faiss' and 'vector_id_mapping.parquet'")

if __name__ == "__main__":
    build_vector_database()