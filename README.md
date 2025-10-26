# reviewer-recommendation-system
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
