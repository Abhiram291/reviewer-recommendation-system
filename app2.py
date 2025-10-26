# ==========================================
# 🧠 Cross-Document Reviewer Recommendation App with Evaluation
# ==========================================

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# ==========================================
# ⚙️ Config
# ==========================================
EMBEDDINGS_FILE = "author_embeddings_agg.npy"     
AUTHOR_NAMES_FILE = "author_names.npy"             
AUTHOR_PROFILES_FILE = "author_profiles.json"      
TOP_K = 5                                          
BACKUP_PER_AUTHOR = 2                              

# -----------------------------
# 🔹 Ground Truth for Test Papers
# Only include papers whose authors exist in the dataset
GROUND_TRUTH = {
    "2022-Deep_Architectures_for_Image_Compression_A_Critical_Review.pdf": ["Dipthi Mishra"],
    "mBERT based model for identification.pdf": ["Arun Chauhan"],
    "1299-1309.pdf" :["Om Prakash Patel","Aruna Tiwari"],
    
}

# ==========================================
# 🧩 Helper Functions
# ==========================================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

# ==========================================
# 🔹 Evaluation Functions
# ==========================================
def top_k_accuracy(preds, gt_authors, k=TOP_K):
    return any(author in gt_authors for author in preds[:k])

def precision_at_k(preds, gt_authors, k=TOP_K):
    correct = sum(1 for a in preds[:k] if a in gt_authors)
    return correct / k

def mean_reciprocal_rank(preds, gt_authors):
    for rank, author in enumerate(preds, start=1):
        if author in gt_authors:
            return 1 / rank
    return 0

# ==========================================
# 🧠 Load Data
# ==========================================
st.write("🔄 Loading author profiles and models...")

with open(AUTHOR_PROFILES_FILE, "r", encoding="utf-8") as f:
    author_texts = json.load(f)

author_names = np.load(AUTHOR_NAMES_FILE, allow_pickle=True)
author_embeddings = np.load(EMBEDDINGS_FILE)
author_embeddings = np.array([normalize(e.reshape(1, -1))[0] for e in author_embeddings])
reviewer_sim_matrix = cosine_similarity(author_embeddings)

# ==========================================
# 🧠 Load Models
# ==========================================
bert_model = SentenceTransformer('all-mpnet-base-v2')

tagged_docs = [TaggedDocument(preprocess(doc), [i]) for i, doc in enumerate(author_texts.values())]
d2v_model = Doc2Vec(vector_size=300, window=5, min_count=2, workers=4, epochs=40)
d2v_model.build_vocab(tagged_docs)
d2v_model.train(tagged_docs, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(list(author_texts.values()))
nmf_model = NMF(n_components=20, random_state=42)
nmf_topics = nmf_model.fit_transform(tfidf)

# ==========================================
# 🔍 Recommendation Methods
# ==========================================
def recommend_bert(paper_text, top_k=TOP_K, backup_per_author=BACKUP_PER_AUTHOR):
    paper_emb = bert_model.encode(paper_text, convert_to_numpy=True)
    paper_emb = normalize(paper_emb.reshape(1, -1))[0].reshape(1, -1)
    sims = cosine_similarity(paper_emb, author_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for i in top_indices:
        author = author_names[i]
        score = float(sims[i])
        reviewer_sims = reviewer_sim_matrix[i]
        backup_indices = reviewer_sims.argsort()[::-1]
        backup_authors = [author_names[j] for j in backup_indices if j != i][:backup_per_author]
        results.append({
            "author": author,
            "score": score,
            "backups": backup_authors
        })
    return results

def recommend_doc2vec(paper_text, top_k=TOP_K):
    paper_tokens = preprocess(paper_text)
    paper_vec = d2v_model.infer_vector(paper_tokens)
    author_vecs = np.array([d2v_model.dv[i] for i in range(len(author_names))])
    sims = cosine_similarity([paper_vec], author_vecs)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

def jaccard_similarity(text1, text2):
    w1, w2 = set(preprocess(text1)), set(preprocess(text2))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def recommend_jaccard(paper_text, top_k=TOP_K):
    sims = [jaccard_similarity(paper_text, author_texts[a]) for a in author_names]
    sims = np.array(sims)
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

def recommend_topic(paper_text, top_k=TOP_K):
    paper_vec = tfidf_vectorizer.transform([paper_text])
    paper_topic = nmf_model.transform(paper_vec)
    sims = np.dot(nmf_topics, paper_topic.T).flatten()
    sims = sims / (np.linalg.norm(nmf_topics, axis=1) * np.linalg.norm(paper_topic))
    top_indices = sims.argsort()[::-1][:top_k]
    return [(author_names[i], float(sims[i])) for i in top_indices]

# ==========================================
# 🎨 Streamlit UI
# ==========================================
st.title("🧠 Multi-Method Reviewer Recommendation System with Evaluation")
st.markdown("Upload a research paper PDF to find the most suitable reviewers using multiple NLP-based approaches.")

uploaded_file = st.file_uploader("📄 Upload a PDF paper", type=["pdf"])
tabs = st.tabs(["BERT (Cross-Doc)", "Doc2Vec", "Jaccard", "Topic Modeling", "Comparison", "Evaluation Metrics"])

if uploaded_file:
    paper_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("🔍 Processing the paper and generating recommendations..."):
        bert_results = recommend_bert(paper_text)
        doc2vec_results = recommend_doc2vec(paper_text)
        jaccard_results = recommend_jaccard(paper_text)
        topic_results = recommend_topic(paper_text)

    st.success("✅ Recommendations generated!")

    # ---------------- BERT Tab ----------------
    with tabs[0]:
        st.subheader("BERT (Cross-Document Embedding) Recommendations")
        for rank, rec in enumerate(bert_results, start=1):
            st.markdown(f"**{rank}. {rec['author']}**  \nSimilarity: `{rec['score']:.4f}`")
            st.markdown(f"_Backup Reviewers:_ {', '.join(rec['backups'])}")
            st.divider()

    # ---------------- Doc2Vec Tab ----------------
    with tabs[1]:
        st.subheader("Doc2Vec Recommendations")
        for rank, (author, score) in enumerate(doc2vec_results, start=1):
            st.write(f"{rank}. {author} (Similarity: {score:.4f})")

    # ---------------- Jaccard Tab ----------------
    with tabs[2]:
        st.subheader("Jaccard Similarity Recommendations")
        for rank, (author, score) in enumerate(jaccard_results, start=1):
            st.write(f"{rank}. {author} (Score: {score:.4f})")

    # ---------------- Topic Modeling Tab ----------------
    with tabs[3]:
        st.subheader("Topic Modeling (NMF) Recommendations")
        for rank, (author, score) in enumerate(topic_results, start=1):
            st.write(f"{rank}. {author} (Score: {score:.4f})")

    # ---------------- Comparison Tab ----------------
    with tabs[4]:
        st.subheader("Method Comparison Dashboard")
        import pandas as pd
        comp_data = {
            "BERT": [r['author'] for r in bert_results],
            "Doc2Vec": [a for a, _ in doc2vec_results],
            "Jaccard": [a for a, _ in jaccard_results],
            "TopicModel": [a for a, _ in topic_results]
        }
        df = pd.DataFrame(comp_data)
        st.dataframe(df)
        st.markdown("This table compares top reviewers across all four methods.")

    # ---------------- Evaluation Metrics Tab ----------------
    with tabs[5]:
        file_name = uploaded_file.name
        if file_name in GROUND_TRUTH:
            gt_authors = GROUND_TRUTH[file_name]
            bert_preds = [r['author'] for r in bert_results]
            doc2vec_preds = [a for a, _ in doc2vec_results]
            jaccard_preds = [a for a, _ in jaccard_results]
            topic_preds = [a for a, _ in topic_results]

            metrics = {}
            for method_name, preds in zip(
                ["BERT", "Doc2Vec", "Jaccard", "TopicModel"],
                [bert_preds, doc2vec_preds, jaccard_preds, topic_preds]
            ):
                metrics[method_name] = {
                    "Top-5 Accuracy": top_k_accuracy(preds, gt_authors, k=TOP_K),
                    "Precision@5": precision_at_k(preds, gt_authors, k=TOP_K),
                    "MRR": mean_reciprocal_rank(preds, gt_authors)
                }

            st.subheader("📊 Evaluation Metrics for Uploaded Paper")
            df_metrics = pd.DataFrame(metrics).T
            st.dataframe(df_metrics)
        else:
            st.info("⚠️ No ground truth available for this uploaded paper. Metrics cannot be computed.")
else:
    st.info("⬆️ Please upload a PDF to get reviewer recommendations.")
