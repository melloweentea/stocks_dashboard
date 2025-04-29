import streamlit as st 
import pandas as pd
import numpy as np 
import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import cufflinks as cf 
from plotly.offline import iplot
import datetime 
from scipy.stats import norm
from financetoolkit import Toolkit
import re
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from newsapi import NewsApiClient
from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage

cf.go_offline()
# st.cache_data.clear()

st.set_page_config("Stock Dashboard", page_icon="📈")
st.sidebar.image("logo transparent.png")
st.sidebar.title("Stock Dashboard")

#initialize chatbot client
client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = None
    
if "ticker" not in st.session_state:
    st.session_state["ticker"] = None
    
if "start_date" not in st.session_state:
    st.session_state["start_date"] = None
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

@st.cache_data
def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start_date, end_date)
    df.reset_index(inplace=True)
    df.columns = df.columns.droplevel(1)  # Drop the second level (Ticker)
    return df 

def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Open"], name="stock open"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="stock close"))
    fig.layout.update(title_text="Stock Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)    
    
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

def get_response(user_input):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant on a stock analysis website"},
            {"role": "user", "content": user_input},
        ],
        stream=True
    )
    return response
    # return response.choices[0].message.content

def stream_response(response):
    for chunk in response:
        yield chunk.choices[0].delta.content

with st.sidebar.form("my form"):
    st.write("Find your preferred stock by entering the following info")
    ticker = st.text_input("ticker symbol (in all caps)", key="t")
    st.session_state["ticker"] = ticker
    start_date = st.text_input("start date in the format YYYY-MM-DD", key="sd")
    st.session_state["start_date"] = start_date
    end_date = st.text_input("end date in the format YYYY-MM-DD", key="ed")
    submitted = st.form_submit_button("download")
    if submitted:
        try:
            df = load_data(ticker, start_date, end_date)
            st.session_state["dataframe"] = df
        except ValueError:
            st.write("wrong input, try again")
        except Exception:
            st.write("invalid ticker entered")
            
        if df.empty:
            st.error("wrong date format or invalid ticker entered, try again")
        else:
            st.info("download complete")

if st.sidebar.button("clear file", type="primary"):
    st.session_state["dataframe"] = None 

choice = st.sidebar.radio("Select what you want to do:", ["technical indicators", "fundamental analysis", "forecast", "sentiment analysis", "chatbot"])

if choice == "technical indicators":
    st.title("Technical Indicators")
    
    volume_flag = st.checkbox(label="Add Volume")
    
    with st.expander("Simple Moving Averages"):
        sma_flag = st.checkbox(label="Add SMA")
        sma_periods = st.number_input(label="SMA Periods", min_value=1, max_value=100, value=20, step=1)

    with st.expander("Bollinger Bands"):
        bb_flag = st.checkbox(label="Add Bollinger Bands")
        bb_periods = st.number_input(label="BB periods", min_value=1, max_value=50, value=20, step=1)
        bb_std = st.number_input(label="number of standard deviations", min_value=1, max_value=4, value=2, step=1)
    
    with st.expander("Relative Strength Index"):
        rsi_flag = st.checkbox(label="Add RSI")
        rsi_periods = st.number_input(label="RSI periods", min_value=1, max_value=50, value=20, step=1)
        rsi_upper = st.number_input(label="RSI upper", min_value=50, max_value=90, value=70, step=1)
        rsi_lower = st.number_input(label="RSI lower", min_value=10, max_value=50, value=30, step=1)
    
    with st.expander("MACD"):
        macd_flag = st.checkbox(label="Add MACD")
        fast_period = st.number_input(label="Fast period", min_value=1, max_value=50, value=12, step=1)
        slow_period = st.number_input(label="Slow period", min_value=1, max_value=50, value=26, step=1)
        signal_period = st.number_input(label="Signal period", min_value=1, max_value=50, value=9, step=1)
    
    if st.session_state["dataframe"] is not None:
        qf = cf.QuantFig(st.session_state["dataframe"], title="Annotated Stock Price")
        
        if volume_flag:
            qf.add_volume()
        if sma_flag:
            qf.add_sma(periods=sma_periods)
        if bb_flag:
            qf.add_bollinger_bands(periods=bb_periods, boll_std=bb_std)
        if rsi_flag:
            qf.add_rsi(periods=rsi_periods, rsi_upper=rsi_upper, rsi_lower=rsi_lower, showbands=True)
        if macd_flag:
            qf.add_macd(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        
        fig = qf.iplot(asFigure=True)
        st.plotly_chart(fig)
    
elif choice == "fundamental analysis":
    st.title("Fundamental Analysis")

    companies = Toolkit(
        tickers=[st.session_state["ticker"]],
        api_key=st.secrets["FMP_API_KEY"],
        start_date=st.session_state["start_date"]
    )
    
    income_statement = companies.get_income_statement()
    st.header("Income Statement")
    st.write(income_statement)
    
    profitability_ratios = companies.ratios.collect_profitability_ratios()
    st.header("Profitability Ratios")
    st.write(profitability_ratios)
    
    efficiency_ratios = companies.ratios.collect_efficiency_ratios()
    st.header("Efficiency Ratios")
    st.write(efficiency_ratios)
    
    liquidity_ratios = companies.ratios.collect_liquidity_ratios()
    st.header("Liquidity Ratios")
    st.write(liquidity_ratios)
    
    solvency_ratios = companies.ratios.collect_solvency_ratios()
    st.header("Solvency Ratios")
    st.write(solvency_ratios)
    
    valuation_ratios = companies.ratios.collect_valuation_ratios()
    st.header("Valuation Ratios")
    st.write(valuation_ratios)

elif choice == "forecast":
    st.title("Stock Forecasting")
    st.subheader("Fbprophet")
    st.subheader("Raw data")
    st.write(st.session_state["dataframe"])
    
    n_years = st.slider("Fbprophet Years of prediction", 1, 4)
    period = n_years * 365 
    
    if st.session_state["dataframe"] is not None:
        plot_raw_data(st.session_state["dataframe"])
        
        #forecasting 
        df_train = st.session_state["dataframe"]
        df_train = df_train[["Date", "Close"]]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        # st.write(df_train["y"].dtypes)
        
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        
        st.subheader("Forecasted data")
        st.write(forecast)
        
        st.subheader(f"fbprophet {n_years} year forecast")
        fig_forecast = plot_plotly(m, forecast)
        st.plotly_chart(fig_forecast)
        
        st.subheader("forecast components")
        fig_comp = m.plot_components(forecast)
        st.write(fig_comp)
    
    st.subheader("Monte carlo")
    n_days = st.slider("Days of prediction", 1, 365, value=50)
    n_runs = st.slider("Number of runs", 1, 20, value=5)
    
    if st.session_state["dataframe"] is not None:
        df = st.session_state["dataframe"]
        log_return = np.log(1 + df["Close"].pct_change())
        
        u = log_return.mean()
        var = log_return.var()
        drift = u - (var / 2)
        
        stdev = log_return.std()
        days = n_days
        trials = n_runs
        Z = norm.ppf(np.random.rand(days, trials))
        daily_returns = np.exp(drift + stdev * Z)
            
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = df["Close"].iloc[-1]
        for t in range(1, days):
            price_paths[t] = price_paths[t-1] * daily_returns[t]
        
        paths_df = pd.DataFrame()
        for idx, path in enumerate(price_paths):
            paths_df[idx] = path
            
        paths_df = paths_df.transpose()
        paths_df["Date"] = pd.date_range(start=df["Date"].iloc[-1], periods=n_days)
        paths_df.set_index("Date", inplace=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close Price"))
        for runs in range(n_runs):
            fig.add_trace(go.Scatter(x=paths_df.index, y=paths_df[runs], name=f"Prediction {runs+1}"))
        fig.layout.update(title_text="Monte Carlo Predictions", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)    

elif choice == "sentiment analysis":
    st.title("Sentiment Analysis")
    company_name = get_company_name(st.session_state["ticker"])
    company_name = remove_co_ltd(company_name)
    # print(company_name)

    newsapi = NewsApiClient(api_key=st.secrets["NEWS_API_KEY"])

    news = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')
    #VADER
    # analyzer = SentimentIntensityAnalyzer()
    #HUGGGINGFACE
    # sentiment_pipeline = pipeline("sentiment-analysis", device="mps")
    #nltk 
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    st.header(company_name)
    st.info(f"Number of articles found: {news['totalResults']}")
    if news["totalResults"] == 0:
        st.error("No news articles found")
    else:
        for article in news["articles"]:
            with st.expander(f"Article: {article['title']}", expanded=True):
                st.subheader(article["title"])
                st.markdown(f"**Description**: {article['description']}")
                st.markdown(f"**Content preview**: {article['content']}")
                st.link_button("Link to article", article["url"])
                
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
                    st.error("negative")
                elif score["compound"] == 0:
                    st.warning("neutral")
                else:
                    st.success("positive")

elif choice == "chatbot":
    st.title("Chatbot")
    # for message in st.session_state.chat_history:
    #     if isinstance(message, HumanMessage):
    #         with st.chat_message("Human"):
    #             st.markdown(message.content)
    #     else:
    #         with st.chat_message("AI"):
    #             st.markdown(message.content)
            
    user_query = st.chat_input("Your question")
    if user_query is not None and user_query != "":
        # st.session_state.chat_history.append(HumanMessage(user_query))
    
        with st.chat_message("Human"):
            st.markdown(user_query)
            st.session_state.user_query = user_query
            
        with st.chat_message("AI"):
            ai_response = st.write_stream(stream_response(get_response(user_query)))
        # st.session_state.chat_history.append(AIMessage(ai_response))
            
    