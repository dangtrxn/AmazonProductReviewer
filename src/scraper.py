# for scraping information
from bs4 import BeautifulSoup
import requests
import re

# for data processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  
import string
import emoji
import pandas as pd

header = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7", 
    "Accept-Encoding": "gzip, deflate, br, zstd", 
    "Accept-Language": "en-US,en;q=0.9", 
    "Priority": "u=0, i", 
    "Sec-Ch-Ua": "\"Chromium\";v=\"146\", \"Not-A.Brand\";v=\"24\", \"Google Chrome\";v=\"146\"", 
    "Sec-Ch-Ua-Mobile": "?0", 
    "Sec-Ch-Ua-Platform": "\"Windows\"", 
    "Sec-Fetch-Dest": "document", 
    "Sec-Fetch-Mode": "navigate", 
    "Sec-Fetch-Site": "cross-site", 
    "Sec-Fetch-User": "?1", 
    "Upgrade-Insecure-Requests": "1", 
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36", 
    "X-Amzn-Trace-Id": "Root=1-69bdf95c-20669cf436d2d5ad7eb97ab2"
}

# function to extract product title
# returns title as string or prints "Title not found"
def scrape_title(soup: BeautifulSoup):
    title_tag = soup.find("span", {"id" : "productTitle"})
    if title_tag:
        prod_title = title_tag.get_text(strip=True)  
    else: 
        prod_title = "Title not found"
    return prod_title

# function to extract product description
# returns description as list of strings or prints "Description not found"
def scrape_desc(soup: BeautifulSoup):
    prod_desc = []
    desc_section = soup.find("div", {"id": "feature-bullets"})
    if desc_section:
        for desc_bullet in desc_section.find_all("span", {"class" : "a-list-item"}):
            text = desc_bullet.get_text(strip=True)
            if text:
                prod_desc.append(text)
    else:
        prod_desc.append('Description not found')
    return prod_desc

# function to extract Amazon Standard Identification Number (ASIN)
# returns ASIN as string or None
def scrape_asin(soup: BeautifulSoup, url: str):
    asin = None

    # try to find asin in item details dropdown
    details_dropdown_table = soup.find("table", {"class" : "a-keyvalue prodDetTable"})
    if details_dropdown_table:
        for tr in details_dropdown_table.find_all("tr"):
            th = tr.find("th")
            td = tr.find('td')
            
            if th and td:
                label = th.get_text(strip=True)
                if "ASIN" in label:
                    asin = td.get_text(strip=True)
                    break
    
    # try to find asin in product details list
    if not asin:
        prod_details_list = soup.find('ul',{"class" : 'a-unordered-list a-nostyle a-vertical a-spacing-none detail-bullet-list'})
        if prod_details_list:
            for li in prod_details_list.find_all('li'):
                bold_text = li.find('span', {'class' : 'a-text-bold'})
                if bold_text:
                    label = bold_text.get_text(strip=True)
                    if "ASIN" in label:
                        span = li.find_all("span")
                        if len(span) > 1:
                            asin = span[-1].get_text(strip=True)
                            break

    #try to find asin in url using regex
    if not asin:
        asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
        if asin_match:
            asin = asin_match.group(1)

    return asin

# function to scrape product page reviews
# returns DataFrame with columns: [title, text, rating, helpfulVotes]
def scrape_reviews(soup: BeautifulSoup):
    review_list = soup.find_all('li', {'data-hook' : 'review'})
    review_data_list = []

    if review_list:
        for review in review_list:
            # only keep U.S. based reviews (for English only)
            review_date = review.find("span", {"data-hook" : "review-date"})
            if review_date:
                if "United States" not in review_date.get_text(strip=True):
                    continue

            # extract star rating
            star_rating = review.find('i', {'data-hook' : 'review-star-rating'})
            rating = None
            if star_rating:
                rating = float(star_rating.find('span').get_text(strip=True).split()[0])

            # extract title
            review_title = review.find("a", {'data-hook' : 'review-title'})
            title = None
            if review_title:
                title = review_title.find_all('span')[-1].get_text(strip=True)

            # extract body
            review_body = review.find("div", {"data-hook" : "review-collapsed"})
            body = None
            if review_body:
                body = review_body.find("span").get_text(strip=True)

            # extract helpful votes
            helpful_votes = review.find("span", {"data-hook" : "helpful-vote-statement"})
            votes = 0
            if helpful_votes:
                word_to_num = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                first_word = helpful_votes.get_text(strip=True).split()[0].lower()
                votes = word_to_num.get(first_word) or int(first_word)

            # append review data to list of reviews
            review_data_list.append([title,body,rating,votes])
    
    if not review_data_list:
        review_container_list = soup.find_all("div", {"data-hook" : "reviewContainer"})
        
        for review in review_container_list:
            # only keep U.S. based reviews (for English only)
            review_date = review.find("span", {"data-hook" : "review-date"})
            if review_date:
                if "United States" not in review_date.get_text(strip=True):
                    continue

            # extract star rating
            star_rating = review.find('i', {'data-hook' : 'review-star-rating'})
            rating = None
            if star_rating:
                rating = float(star_rating.find('span').get_text(strip=True).split()[0])

            # extract title
            review_title = review.find("h5", {'data-hook' : 'reviewTitle'})
            title = None
            if review_title:
                title = review_title.get_text(strip=True)

            # extract body
            review_body = review.find("div", {"data-hook" : "reviewRichContentContainer"})
            body = None
            if review_body:
                body = review_body.find("span").get_text(strip=True)

            # extract helpful votes
            helpful_votes = review.find("span", {"data-hook" : "helpful-vote-statement"})
            votes = 0
            if helpful_votes:
                word_to_num = {
                    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                first_word = helpful_votes.get_text(strip=True).split()[0].lower()
                votes = word_to_num.get(first_word) or int(first_word)

            # append review data to list of reviews
            review_data_list.append([title,body,rating,votes])

    df = pd.DataFrame(review_data_list, columns=['title','text','rating','helpfulVotes'])
    return df 

# function to scrape all necessary information from product: title, description, asin, and reviews
# returns prod_info dictionary
def scrape_product(url: str):
    r = requests.get(url, headers = header)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")

    # reviews_df = preprocess(scrape_reviews(soup))
    reviews_df = scrape_reviews(soup)
    product_info = {
        "title": scrape_title(soup), 
        "description": scrape_desc(soup),
        "asin": scrape_asin(soup, url),
        "reviews": reviews_df
    }
    return product_info

# function for preprocessing steps
# returns cleaned DataFrame: 
# lowercase and remove punctuation/emojis
def preprocess(review_df):
    stop_words = set(stopwords.words('english'))

    # helper function to clean text, includes lowercase, remove punctuation + emojis
    def clean(text: str):
        # lowercase
        text = text.lower()
        # remove punctuation
        trans = str.maketrans("","",string.punctuation)
        text = text.translate(trans)
        # remove emojis 
        text = emoji.replace_emoji(text,"")

        return text

    #clean title only
    review_df = review_df.copy()
    review_df["title"] = review_df["title"].fillna("")
    review_df["title"] = review_df["title"].apply(clean)

    return review_df

# function for preprocessing with stopword removal
# returns cleaned DataFrame: 
# lowercase, remove punctuation/emojis, tokenize, remove stopwords from text only
def preprocess_remove_stopwords(review_df):
    stop_words = set(stopwords.words('english'))

    # helper function to clean text, includes lowercase, remove punctuation + emojis, and tokenization
    def clean(text: str):
        # lowercase
        text = text.lower()
        # remove punctuation
        trans = str.maketrans("","",string.punctuation)
        text = text.translate(trans)
        # remove emojis 
        text = emoji.replace_emoji(text,"")
        # tokenize
        tokens = word_tokenize(text)

        return tokens

    # clean title, without stopword removal
    review_df = review_df.copy()
    review_df["title"] = review_df["title"].fillna("")
    review_df["title"] = review_df["title"].apply(clean)
    
    # clean text, with stopwords removed
    review_df["text"] = review_df["text"].apply(lambda text: [w for w in clean(text) if w not in stop_words])

    return review_df