### Reviewer-recommendation-system

A multi-method NLP-based system to recommend reviewers for research papers using BERT, Doc2Vec, Jaccard similarity, and Topic Modeling (NMF).”


## Features

- **BERT (Cross-Document Embedding):** Leverages sentence embeddings for semantic similarity.  
- **Doc2Vec:** Learns document-level embeddings from author profiles.  
- **Jaccard Similarity:** Measures word overlap between paper and author profiles.  
- **Topic Modeling (NMF):** Finds authors with similar research topics.
- *Comparison Dashboard:** Shows top recommended authors from all methods side-by-side.  


## Dataset

The dataset contains **author profiles** and their research papers in PDF format. It is used to train and evaluate the reviewer recommendation system.

### Structure

Dataset/
│
├─ Author_Name_1/
│ ├─ paper1.pdf
│ ├─ paper2.pdf
│ └─ ...
├─ Author_Name_2/
│ ├─ paper1.pdf
│ └─ ...
└─ ...


## Running the Project

The project consists of multiple Python scripts, each with a specific purpose. Follow the steps below to set up the system and generate reviewer recommendations.

---

### 1) Extract Author Texts from PDFs
File: `read_pdf.py`

- **Purpose:** Reads all PDF papers in the dataset and extracts text for each author.  
- **Output:** `author_profiles.json` – a JSON file containing concatenated text for each author.  

##Run the file ###2

--------------------------------------------------------------------------

### 2) Generate Author Embeddings
File: `author_embeddings.py`

-**Purpose:** Computes vector embeddings for each author using a SentenceTransformer (BERT-based).
-**Output**:
 `author_embeddings_agg.npy ` – embeddings for each author
 `author_names.npy ` – list of author names

----------------------------------------------------------------------------

### 3) Evaluate Recommendation Methods
File: `evaluation_methods.py`

- **Purpose:** Evaluates the performance of all recommendation methods on test papers using:
  - BERT embeddings
  - Doc2Vec
  - Jaccard Similarity
  - Topic Modeling (NMF)
  - Optional: Word Mover's Distance (WMD)
- Computes top-K recommendations for each method.
- Computes reviewer-reviewer similarity and generates a heatmap (`reviewer_similarity_heatmap.png`).

- We have inlcuded it in our app2.py file itself so it is not required to run this file

----------------------------------------------------------------------------

  ## 4) Run the Reviewer Recommendation App
File: `app2.py`

- **Purpose:** Streamlit app for uploading a research paper PDF and generating top reviewer recommendations.
-Upload a PDF in the Streamlit UI to get recommendations.
-The app displays top k reviewers with confidence score(c) for each method.


















