import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class CompanyClassifier:
    # uses pre-trained sentence transformer model to understand
    
    def __init__(self, model_name: str, threshold: float):

        self.threshold = threshold
        print(f"Loading sentence transformer model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully")
        
        self.label_embeddings = None
        self.taxonomy_labels = None

    def fit(self, label_texts: pd.Series, taxonomy_labels: pd.Series):
        # we pre-calculate the label vectors for speed,this avoids having to re-encode the entire taxonomy for every prediction
        print("Encoding taxonomy labels into semantic vectors")
        self.label_embeddings = self.model.encode(
            label_texts.tolist(),
            convert_to_tensor=True,
            show_progress_bar=True
        )
        self.taxonomy_labels = taxonomy_labels.to_numpy()
        print("Taxonomy labels encoded and ready")

    def predict(self, company_texts: pd.Series) -> pd.Series:

        #turn the company descriptions into vectors
        print("Encoding company descriptions for semantic prediction")
        company_embeddings = self.model.encode(
            company_texts.tolist(),
            convert_to_tensor=True,
            show_progress_bar=True
        )

        # calculate the similarity between companies and all possible labels
        print("Calculating similarity and assigning labels")
        from sentence_transformers.util import cos_sim
        cosine_sim = cos_sim(company_embeddings, self.label_embeddings).cpu().numpy()

        assigned_labels = []
        for i in range(len(cosine_sim)):
            # find all labels that meet our similarity threshold

            best_idxs = np.where(cosine_sim[i] >= self.threshold)[0]
            
            if len(best_idxs) == 0:
                assigned_labels.append("None")
            else:
                # get both the labels and their corresponding scores
                labels = self.taxonomy_labels[best_idxs]
                scores = cosine_sim[i][best_idxs]
                
                # zip them together and sort by score, highest first
                label_score_pairs = list(zip(labels, scores))
                sorted_pairs = sorted(label_score_pairs, key=lambda x: x[1], reverse=True)
                
                ### DEBUG
                #formatted_labels = [f"{label} ({score:.2f})" for label, score in sorted_pairs]
                
                formatted_labels = [label for label, _ in sorted_pairs]
                final_string = ', '.join(formatted_labels)
                assigned_labels.append(final_string)

        print("Semantic prediction complete")
        return pd.Series(assigned_labels, index=company_texts.index)