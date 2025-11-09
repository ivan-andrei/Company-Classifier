import config
from data_loader import load_data
from preprocessor import preprocess_data
from classifier import CompanyClassifier

def run():
    # load the data
    companies_df, taxonomy_df = load_data(config.COMPANY_DATA_PATH, config.TAXONOMY_DATA_PATH)
    if companies_df.empty or taxonomy_df.empty:
        return

    # clean up the text fields
    companies_df, taxonomy_df = preprocess_data(
        companies_df,
        taxonomy_df,
        config.COMPANY_TEXT_COLS,
        config.TAXONOMY_LABEL_COL
    )

    # initialize the classifier
    classifier = CompanyClassifier(
        model_name=config.SENTENCE_TRANSFORMER_MODEL,
        threshold=config.SEMANTIC_THRESHOLD
    )
    
    # pre-compute the embeddings for the taxonomy labels
    classifier.fit(
        label_texts=taxonomy_df['processed_label'],
        taxonomy_labels=taxonomy_df[config.TAXONOMY_LABEL_COL]
    )

    # get the label predictions for all companies
    predictions = classifier.predict(companies_df['processed_text'])
    
    ### DEBUG: print a summary of the results
    # total_companies = len(predictions)
    # none_count = predictions.value_counts().get("None", 0)
    # none_rows = predictions[predictions == "None"].head(100)
    # print(none_rows)
    # labeled_count = total_companies - none_count
    
    # print("\n--- Classification Summary ---")
    # print(f"Total Companies Processed: {total_companies}")
    # if total_companies > 0:
    #     labeled_percent = (labeled_count / total_companies) * 100
    #     none_percent = (none_count / total_companies) * 100
    #     print(f"Companies Labeled:         {labeled_count} ({labeled_percent:.1f}%)")
    #     print(f"Companies Unlabeled (None): {none_count} ({none_percent:.1f}%)")
    # print("----------------------------\n")
    
    # add the results to the dataframe and save the final file
    companies_df['insurance_label'] = predictions
    
    final_cols = list(companies_df.columns[:-3])
    final_cols.append('insurance_label')
    
    companies_df[final_cols].to_csv(config.OUTPUT_DATA_PATH, index=False)
    
    print(f"classification finished. final output saved to {config.OUTPUT_DATA_PATH}")
    print("\nsample of final results:")
    print(companies_df[final_cols].head())

if __name__ == "__main__":
    run()