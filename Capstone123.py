# -*- coding: utf-8 -*-
import gdown
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Download dataset
file_id = '1E4W1RvNGgyawc6I4TxQk76n289FX9kCK'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, 'dataset social media.xlsx', quiet=False)

# ===============================
# === DATA PREPARATION ====
# ===============================

@st.cache
def load_data():
    df = pd.read_excel('dataset social media.xlsx', sheet_name='Working File')
    # Clean up columns
    for col in ['Platform', 'Post Type', 'Audience Gender', 'Age Group', 'Sentiment', 'Time Periods', 'Weekday Type']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    # Drop irrelevant columns
    drop_cols = ['Post ID', 'Date', 'Time', 'Audience Location', 'Audience Continent', 'Audience Interests', 'Campaign ID', 'Influencer ID']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    # Convert timestamp and time-related features
    df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'], errors='coerce')
    df = df.dropna(subset=['Post Timestamp'])
    df['Post Hour'] = df['Post Timestamp'].dt.hour
    df['Post Day Name'] = df['Post Timestamp'].dt.day_name()
    return df

df = load_data()

# ===============================
# === SENTIMENT ANALYSIS ====
# ===============================

nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(caption):
    score = vader_analyzer.polarity_scores(caption)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# ===============================
# === MAIN STREAMLIT APP ====
# ===============================

# App Title
st.markdown("""
<div style='text-align: center;'>
    <span style="font-size:3em;">📊</span><br>
    <span style="font-size:1.8em; font-weight: bold;">Social Media Caption & Posting Analytics</span><br>
    <span style="font-size:1.2em; color:gray;">Boost Your Engagement with Smart Caption Analysis and Optimal Posting Times</span>
</div>
""", unsafe_allow_html=True)

# Social Media Logos
st.markdown("""
<div style='text-align: center; display: flex; justify-content: center; gap: 20px;'>
    <div>
        <img src="https://drive.google.com/uc?id=1-W5rrAD5ltgBM4uH6PQd4U6PPdp6pTiZ" width='100' />
        <p>Facebook</p>
    </div>
    <div>
        <img src="https://drive.google.com/uc?id=12SltF6Pp-XXJSfLegFIy-ZkS2eTjWmWw" width='100' />
        <p>Instagram</p>
    </div>
    <div>
        <img src="https://drive.google.com/uc?id=19G9XyD8-VviezDanqxJUBDWIXnziaDRn" width='100' />
        <p>LinkedIn</p>
    </div>
    <div>
        <img src="https://drive.google.com/uc?id=1sEOKIcqOwvugj2xEuRwa6vnQO5daNZNo" width='100' />
        <p>X</p>
    </div>
</div>
""", unsafe_allow_html=True)

# User Input Form
with st.form(key='input_form'):
    caption_input = st.text_area("Enter Your Caption:")
    post_type_input = st.selectbox("Select Post Type", sorted(df['Post Type'].unique()))
    audience_gender_input = st.selectbox("Select Audience Gender", sorted(df['Audience Gender'].unique()))
    age_group_input = st.selectbox("Select Age Group", sorted(df['Age Group'].unique()))
    platforms = list(sorted(df['Platform'].unique())) + ['All']
    platform_input = st.selectbox("Select Platform", platforms)
    submit_button = st.form_submit_button(label='Check Recommendations')

if submit_button:
    # 1. Sentiment Analysis
    sentiment_result = analyze_sentiment(caption_input)
    st.success(f"Predicted Sentiment for Your Caption: **{sentiment_result}**")

    # Add other recommendations and analysis logic below

# Save & run the app
if __name__ == "__main__":
    st.title("Streamlit Capstone")
