import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources if not already downloaded
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    # Analyze sentiment
    sentiment_scores = sia.polarity_scores(text)
    
    # Interpret sentiment
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, sentiment_scores

# Streamlit UI
st.title("Sentiment Analysis App")

# User input for text
user_input = st.text_area("Enter a text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment, scores = analyze_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Positive: {scores['pos']:.2f}")
        st.write(f"Negative: {scores['neg']:.2f}")
        st.write(f"Neutral: {scores['neu']:.2f}")
        st.write(f"Compound Score: {scores['compound']:.2f}")
    else:
        st.warning("Please enter some text for analysis.")
