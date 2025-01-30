from newsapi import NewsApiClient
import streamlit as st
import yfinance as yf 
import requests
import re
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
# import torch 
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd 

def get_company_name(ticker):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            return data["quotes"][0].get("longname", "Company name not found")
    return "Company name not found"

def remove_co_ltd(company_name):
    return re.sub(r'\s*Co\., Ltd\.$', '', company_name)
# company_name = get_company_name("000001.SS")
company_name = get_company_name("600519.SS")
company_name = remove_co_ltd(company_name)
print(company_name)

newsapi = NewsApiClient(api_key=st.secrets["NEWS_API_KEY"])

news = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')
#VADER
# analyzer = SentimentIntensityAnalyzer()
#HUGGGINGFACE
# sentiment_pipeline = pipeline("sentiment-analysis", device="mps")
#nltk 
sia = SentimentIntensityAnalyzer()

if news["totalResults"] == 0:
    print("No news articles found")
else:
    for article in news["articles"]:
        print(article["title"], article["content"], article["description"], article["url"])
        html_text = requests.get(article["url"]).text
        soup = BeautifulSoup(html_text, "html.parser") 
        text = soup.get_text()
        #VADER
        # score = analyzer.polarity_scores(article["content"])
        #HUGGINGFACE
        # result = sentiment_pipeline(text)
        # print(result)
        #NLTK 
        score = sia.polarity_scores(text)
        if score["compound"] < 0:
            print("negative")
        elif score["compound"] == 0:
            print("neutral")
        else:
            print("positive")
        
# print(torch.backends.mps.is_available())  # Should return True
# print(torch.backends.mps.is_built())  # Should return True