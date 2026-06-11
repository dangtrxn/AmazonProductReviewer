import joblib
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scraper import scrape_product, preprocess
import pandas as pd
import numpy as np
import torch

# summarize function for summarization model
# max_input=1024, min_output=30, max_output=130
def summarize(text, max_input=1024, max_output=130, min_output=30):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_input,
        truncation=True,
        padding=True
    ).to('cpu')

    # helps save memory during inference
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_output,
            min_length=min_output,
            length_penalty=2.0,
            num_beams=2,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# function to summarize reviews
# combine up to max_reviews=10 and truncate to max_input_tokens=1024
def summarize_reviews(reviews, max_reviews=10, max_input_tokens=1024):
    if reviews.empty:
        return None
    
    combined = ' | '.join(reviews['text'].dropna().tolist()[:max_reviews])
    truncated = ' '.join(combined.split()[:max_input_tokens])

    return summarize(truncated)

# function to summarize descriptions
# combine all description bullet points and truncate to max_input_tokens=1024
def summarize_description(description: list, max_input_tokens=1024):
    if not description:
        return None
    
    combined = ' '.join(description)
    truncated = ' '.join(combined.split()[:max_input_tokens])

    return summarize(truncated, max_output=100, min_output=30)

# function to analyze entire product
# scrape all information, run sentiment analysis on reviews, summarize description and reviews, return prod_summary
def analyze(url:str):
    # scrape all necessary product information
    print(f'Scraping: {url}\n')
    product = scrape_product(url)
    reviews_df = product['reviews']
    reviews_df = preprocess(reviews_df)
    
    #check if reviews were properly scraped/found
    if reviews_df.empty:
        print('No reviews found')
        return {}
    
    # sentiment predictions using trained tfidf-logistic regression model
    X = reviews_df['title'].apply(lambda tokens: ''.join(tokens)) + ' ' + reviews_df['text'].apply(lambda tokens: ''.join(tokens))
    start_time = time.time()
    reviews_df['predictedSentiment'] = sentiment_pipeline.predict(X)
    print(f'Sentiment analysis done in {time.time() - start_time:.2f}s')
    sentiment_counts = reviews_df['predictedSentiment'].value_counts().to_dict()

    # summarize description
    print("Summarizing description: ")
    start_time = time.time()
    desc_summary = summarize_description(product['description'])
    print(f'Description summary done in {time.time() - start_time:.2f}s')

    # summarize reviews
    print("Summarizing reviews: ")
    review_summary = summarize_reviews(reviews_df)
    print(f'Review summary done in {time.time() - start_time:.2f}s')

    prod_summary = {
        "title": product['title'],
        "asin": product["asin"],
        "desc_summary": desc_summary,
        "avg_rating": round(reviews_df['rating'].sum()/reviews_df.shape[0],2),
        "total_reviews": len(reviews_df),
        "sentiment_counts": sentiment_counts,
        "review_summary": review_summary
    }

    return prod_summary

def main():
    print("---------------------------\n| Amazon Product Reviewer |\n---------------------------\n")
    url = input("Enter the url of the product to review: ")
    print()

    results_df = analyze(url)
    print("\n--------------------------------------\n| Refined Amazon Product Information |\n--------------------------------------")
    print(f"Product Title: {results_df['title']}")
    print(f"ASIN: {results_df['asin']}")
    print(f"Description Summary: {results_df['desc_summary']}")
    print(f"Total Reviews Analyzed: {results_df['total_reviews']}")
    print(f"Average Review Rating: {float(results_df['avg_rating'])}/5.0 stars")
    print(f"Sentiment Analysis Count Mapping: ")
    for k,v in results_df['sentiment_counts'].items():
        print(f"\t{k}: {v}")
    print(f"Summary of Reviews:")
    reviews_list = results_df['review_summary'].split("; ")
    for review in reviews_list:
        print(f"\t- {review}")

if __name__ == '__main__':
    # load tf-idf logistic regression classifier and finetuned bart summarizer models
    model_name = 'models/bart-finetuned-reviews-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    sentiment_pipeline = joblib.load('models/tfidf_logreg.joblib')

    main()