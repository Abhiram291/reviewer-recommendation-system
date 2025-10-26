<img width="431" height="768" alt="Screenshot 2025-10-26 at 1 07 26 PM" src="https://github.com/user-attachments/assets/b4c7c972-c9ba-4384-90e3-c45e5d0fdb22" /># Reviewer-recommendation-system
A multi-method NLP-based system to recommend reviewers for research papers using BERT, Doc2Vec, Jaccard similarity, and Topic Modeling (NMF).”


## Features

- **BERT (Cross-Document Embedding):** Leverages sentence embeddings for semantic similarity.  
- **Doc2Vec:** Learns document-level embeddings from author profiles.  
- **Jaccard Similarity:** Measures word overlap between paper and author profiles.  
- **Topic Modeling (NMF):** Finds authors with similar research topics.
- *Comparison Dashboard:** Shows top recommended authors from all methods side-by-side.  


## Dataset

The dataset contains **author profiles** and their research papers in PDF format. It is used to train and evaluate the reviewer recommendation system.

### **Structure**

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

### **1️⃣ Extract Author Texts from PDFs**
File: `read_pdf.py`

- **Purpose:** Reads all PDF papers in the dataset and extracts text for each author.  
- **Output:** `author_profiles.json` – a JSON file containing concatenated text for each author.  

##Run the file ###2

--------------------------------------------------------------------------

### ** 2️⃣ Generate Author Embeddings
File: `author_embeddings.py`

-**Purpose:** Computes vector embeddings for each author using a SentenceTransformer (BERT-based).
-**Output**:
 `author_embeddings_agg.npy ` – embeddings for each author
 `author_names.npy ` – list of author names

----------------------------------------------------------------------------

### ** 3️⃣ Evaluate Recommendation Methods**
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

  ## ** 4️⃣ Run the Reviewer Recommendation App**
File: `app2.py`

- **Purpose:** Streamlit app for uploading a research paper PDF and generating top reviewer recommendations.
-Upload a PDF in the Streamlit UI to get recommendations.
-The app displays top reviewers and backup reviewers for each method.



### ** 5️⃣ Evaluation Metrics Report**

- **Purpose:** Measures the performance of different reviewer recommendation methods on test papers.  
- **Metrics Used:**
  1. **Top-K Accuracy:** Checks if at least one ground-truth author is in the top-K recommendations.  
  2. **Precision@K:** Fraction of recommended authors in top-K that are correct.  
  3. **MRR (Mean Reciprocal Rank):** Average of reciprocal ranks of the first relevant author in the recommendations.

- **Example Output:**

| Method      | Top-5 Accuracy | Precision@5 | MRR  |
|------------|----------------|-------------|------|
| BERT       | TRUE           | 0.2         | 1    |
| Doc2Vec    | TRUE           | 0.2         | 1    |
| Jaccard    | TRUE           | 0.2         | 1    |
| TopicModel | TRUE           | 0.2         | 0.5  |

- **Limitations:**
  - Evaluation can only be performed on papers **whose authors are present in the dataset**.  
    - Reason: Papers outside the dataset may have authors not included in `author_profiles.json`, making comparison impossible.  
  - Metrics are sensitive to small dataset size; testing on a single paper may give skewed results.  
  - Some methods (e.g., Jaccard) may produce lower scores due to vocabulary mismatch or short paper length.  
  - Topic modeling depends on the number of topics chosen (`NUM_TOPICS`) and may vary across runs.

- **Files Used:**  
  - `evaluation_methods.py` – computes metrics and reviewer-reviewer similarity  
  - `author_profiles.json` – extracted author texts  
  - Test PDFs – papers used for evaluation




### ** 6️⃣ Limitations & Evaluation Rationale**

**1. Limitation of the Recommendation System**  
- The system recommends reviewers only from a precomputed list of authors with available profiles and embeddings.  
- This list is derived entirely from your dataset.  
- If a paper’s author is not in the dataset, the system has no information (embeddings or profile) to recommend that author.  
- Therefore, the system **cannot recommend authors outside the dataset**.

**2. Ground Truth Must Match Available Data**  
- Evaluation metrics like Top-K Accuracy, Precision@K, and MRR compare recommended authors against the ground truth:  
  - **Top-K Accuracy:** Checks if at least one ground-truth author appears in the top-K recommendations.  
  - **Precision@K:** Fraction of correct authors in the top-K.  
  - **MRR (Mean Reciprocal Rank):** Reciprocal rank of the first correct author.  
- If the ground truth includes authors not in the dataset, the system will never recommend them.  
  - This artificially lowers metrics and may give a **misleading impression of poor performance**, even if the system ranks the correct authors present in the dataset perfectly.

**3. Example**  
- Paper authors: Shankar Biradar, Sunil Saumya, Arun Chauhan  
- Only Arun Chauhan is in the dataset.  
- The system can only recommend Arun Chauhan.  
- Including Shankar Biradar or Sunil Saumya in the ground truth:  
  - Precision@5 = 1/5 (appears low)  
  - MRR is affected because the system “fails” to predict non-dataset authors  
✅ **Correct approach:** Include only authors present in the dataset in your ground truth.

**4. Key Takeaway**  
- Evaluation metrics reflect the system’s ability to rank **known authors**.  
- Including authors outside the dataset leads to misleading results.  
- For fair evaluation, **only test papers with at least one author present in the dataset should be used**.





### Streamlit App Preview
![Streamlit App]







