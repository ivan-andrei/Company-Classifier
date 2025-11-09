# file paths for the project
COMPANY_DATA_PATH = "data/ml_insurance_challenge.csv"
TAXONOMY_DATA_PATH = "data/insurance_taxonomy.csv"
OUTPUT_DATA_PATH = "data/labeled_company_list.csv"

# columns
COMPANY_TEXT_COLS = ['description', 'business_tags', 'sector', 'category', 'niche']
TAXONOMY_LABEL_COL = 'label'

# sentence model settings
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
# our final, data-driven threshold, chosen to maximize precision
SEMANTIC_THRESHOLD = 0.4