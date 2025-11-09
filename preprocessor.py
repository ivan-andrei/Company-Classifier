import pandas as pd
import re
from typing import List, Tuple

def clean_text(text: str) -> str:
    
    #strips text of spaces and punctuation, and lowercases it
    
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # keep alphanumeric and spaces
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text

def preprocess_data(companies_df: pd.DataFrame, taxonomy_df: pd.DataFrame,
                    company_cols: List[str], taxonomy_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #prepares data for classification by combining text fields and cleaning them

    # preprocess companies data
    text_data = companies_df[company_cols].fillna('').astype(str)
    companies_df['combined_text'] = text_data.agg(' '.join, axis=1)

    companies_df['processed_text'] = companies_df['combined_text'].apply(clean_text)

    # preprocess taxonomy data
    taxonomy_df['processed_label'] = taxonomy_df[taxonomy_col].apply(clean_text)

    print("Text preprocessing completed")

    return companies_df, taxonomy_df