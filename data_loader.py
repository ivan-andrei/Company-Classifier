import pandas as pd
from typing import Tuple

def load_data(company_path: str, taxonomy_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    #loads company and taxonomy data from the specified csv files
    try:
        ### DEBUG: run only on 100 rows
        # companies_df = pd.read_csv(company_path, nrows=100)
        
        companies_df = pd.read_csv(company_path)
        taxonomy_df = pd.read_csv(taxonomy_path)
        print("Data loaded")
        return companies_df, taxonomy_df
    except FileNotFoundError as e:
        print(f"error: {e}. please ensure the paths are correct")
        return pd.DataFrame(), pd.DataFrame()