# Company Classifier

Assign companies to a fixed insurance taxonomy **without any labelled training data** using semantic embeddings

---

## Setup and Installation

**Clone the repository:**

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

**Create and activate a virtual environment:**

```bash
python3.12 -m venv venv

#Windows
venv\Scripts\activate

#Linux/macOS
source venv/bin/activate
```

**Install the required dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** At first run ~90MB Sentence Transformer Model will be downloaded

---

**To Run**

Execute the main script from the root directory

```bash
python main.py
```

---

## Output

The script will generate a new file specified in `config.py` (by default: `data/labeled_company_list.csv`).  
This CSV will contain all the original columns from the input company list plus a new column `insurance_label` with the assigned classifications.

---

## Problem Overview

In this document I explain the approach, reasoning, and results for a project that assigns companies to a fixed insurance taxonomy without any labelled training data.

The task was to classify companies into a predefined insurance taxonomy without using any labelled examples. Two main challenges appeared: first, there is no ground truth to train or directly measure performance against. This required a clear, repeatable validation plan. Second, the system had to be designed with scalability in mind so that it could handle very large datasets in a production environment.

---

## Approach

I followed a simple, iterative process: start with a reliable baseline, then move to a stronger semantic method while validating each change.

### Term Frequency–Inverse Document Frequency (TFIDF) with cosine similarity

I first built a baseline classifier using TFIDF vectors and cosine similarity. The baseline established a working pipeline quickly and was useful to confirm the overall logic. TF-IDF is fast, transparent and scales well, but it depends on matching exact words and often misses conceptual similarities.

### Improvements: Sentence embeddings (all-MiniLM-L6-v2)

To capture semantic meaning, I switched to pre-trained sentence embeddings using the all-MiniLM-L6-v2 model. This model understands context and similarity at a conceptual level, so it can match descriptions like “utility network connections design and construction” with labels such as “Civil Engineering Services” even when the words are different. This change proved essential for accurate zero-shot classification.

### Other ideas considered

I briefly considered unsupervised clustering to find natural groups in the data. While useful for exploration, clustering would require manual mapping of clusters to the fixed taxonomy and is therefore less relevant for this specific task.

---

## Validation Without Ground Truth

Validating results without labelled data was definitely a challenge. I used a mixed strategy to gain confidence in the output.

### Threshold tuning

I tested different cosine similarity thresholds. A low threshold (around 0.3) found many candidates but produced a lot of errors. After manually checking the confidence scores, the correct labels were almost always 0.4 or higher, so I raised the threshold to 0.4, which gave a much better balance, with higher precision and fewer incorrect assignments.

### Spot checks and qualitative review

I manually reviewed a sample of outputs. Some companies were clearly well matched. For example, a firm named “Loidholdhof Farm” received labels such as “Bakery Production Services” and “Gardening Services”, which matched the company description. Other entries, like an auto body shop under the name PATAGONIA, were correctly left without a label, indicating that the taxonomy did not cover those businesses.

### Looking at the ‘None’ Rate

In the full list, 48% of the companies were labeled as None. This is important as it suggests that the current taxonomy does not cover a large part of the dataset. Instead of showing a problem with the model, this result points to missing categories in the label set itself.

---

## Final Solution and Key Findings

The final setup uses the all-MiniLM-L6-v2 embeddings with a cosine similarity threshold of 0.4. This provides precise and meaningful assignments when a match exists.

Two key conclusions stand out:

The model is accurate at identifying relevant labels. When it assigns a label, the match is rarely wrong.  
The main limit is the taxonomy. The 48% None rate reveals missing categories such as automotive services, healthcare providers, hospitality and many common technology services.

---

## Scalability and Next Steps

The solution is ready for scale. Pre-computing embeddings for the taxonomy and storing them in a vector index (for example FAISS or Pinecone) allows fast similarity searches even at large scale. The process can run in parallel and be deployed on cloud infrastructure.

The most important next step is to expand the taxonomy. By adding the missing labels we identified, we can run the same process again and directly see improvements in coverage.

---

## Conclusion

This project shows that modern semantic methods make zero-shot classification useful and reliable. More importantly, the model helps find parts of the taxonomy that need improvement. Fixing these gaps will allow more companies to get meaningful labels and will improve data quality and analysis.

---

## References

Hugging Face. all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  
R. Author et al., Language Models to Support Multi-Label Classification of Industrial Data. https://arxiv.org/pdf/2504.15922
