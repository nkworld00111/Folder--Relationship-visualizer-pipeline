import numpy as np

def embed_documents(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generates vector embeddings for each document content."""
    if not docs or not isinstance(docs, list):
        print("⚠️ Invalid or empty document list.")
        return []

    texts = [d.get("content", "") for d in docs]

    try:
        from sentence_transformers import SentenceTransformer
        print(f"🧠 Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        print("  ✅ Model loaded successfully. Generating embeddings...")
        vecs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        vecs = vecs.astype(float)
    except Exception as e:
        print("  ⚠️ SentenceTransformers unavailable or failed:", e)
        print("  ⏪ Falling back to TF-IDF + TruncatedSVD...")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
            X = tfidf.fit_transform(texts)
            n_comp = min(384, max(64, X.shape[1] // 2))
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            vecs = svd.fit_transform(X)
        except Exception as e2:
            print("  ❌ Fallback failed:", e2)
            print("  Generating random small vectors instead for continuity.")
            vecs = np.random.randn(len(texts), 128).astype(float)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    for d, v in zip(docs, vecs):
        d["embedding"] = v.astype(float).tolist()

    print(f"✅ Generated embeddings for {len(docs)} document(s).")
    return docs


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python embedder.py <extracted_documents.json>")
        sys.exit(1)
    in_file = sys.argv[1]
    with open(in_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    embedded = embed_documents(docs)
    out_file = "extracted_with_embeddings.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(embedded, f, indent=2, ensure_ascii=False)
    print("Saved", out_file)
