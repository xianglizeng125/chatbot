import os
import zipfile
import gdown
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, TFDistilBertModel
import tensorflow as tf
from tensorflow.keras.models import load_model

# ====== CONFIG ======
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")
MAX_LEN = 100
GOOGLE_DRIVE_ZIP_ID = "1fKrPXGWGMn0mkwLPGOAgEH7f9_efR1Sd"
ZIP_PATH = "nlp_assets.zip"
EXTRACT_PATH = "nlp_assets"

# ====== DOWNLOAD & LOAD MODEL/TOKENIZER ======
@st.cache_resource
def download_and_load_assets():
    if not os.path.exists(EXTRACT_PATH):
        with st.spinner("📥 Downloading model & tokenizer..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ZIP_ID}"
            try:
                gdown.download(url, ZIP_PATH, quiet=False)
                with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                    zip_ref.extractall(EXTRACT_PATH)
            except Exception as e:
                st.error(f"❌ Error downloading or extracting ZIP: {e}")
                return None, None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(EXTRACT_PATH, "tokenizer_distilbert"), local_files_only=True)
        bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        bert_model.trainable = False
        model = load_model(os.path.join(EXTRACT_PATH, "model.keras"), compile=False)
        st.success("✅ Model and tokenizer loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading tokenizer/model: {e}")
        return None, None, None

    return tokenizer, bert_model, model

tokenizer, bert_model, sentiment_model = download_and_load_assets()

# ====== MENU & DATA ======
menu_actual = [
    "soondubu jjigae", "prawn soondubu jjigae", "kimchi jjigae", "tofu jjigae",
    "samgyeopsal", "spicy samgyeopsal", "woo samgyup", "spicy woo samgyup",
    "bulgogi", "dak bulgogi", "spicy dak bulgogi", "meltique tenderloin", "odeng",
    "beef soondubu jjigae", "pork soondubu jjigae"
]

menu_aliases = {
    "soondubu": "soondubu jjigae", "suundobu": "soondubu jjigae",
    "beef soondubu": "beef soondubu jjigae", "pork soondubu": "pork soondubu jjigae",
    "soondubu jigae": "soondubu jjigae"
}

keyword_aliases = {
    "spicy": "spicy", "meat": "meat", "soup": "soup", "seafood": "seafood",
    "beef": "beef", "pork": "pork", "bbq": "bbq", "non-spicy": "non_spicy", "tofu": "tofu_based"
}

menu_categories = {
    "spicy": ["spicy samgyeopsal", "spicy woo samgyup", "spicy dak bulgogi", "kimchi jjigae", "budae jjigae"],
    "meat": ["samgyeopsal", "woo samgyup", "bulgogi", "dak bulgogi", "saeng galbi", "meltique tenderloin"],
    "soup": ["kimchi jjigae", "tofu jjigae", "budae jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae", "prawn soondubu jjigae"],
    "seafood": ["prawn soondubu jjigae", "odeng"],
    "beef": ["bulgogi", "beef soondubu jjigae", "meltique tenderloin"],
    "pork": ["samgyeopsal", "spicy samgyeopsal", "pork soondubu jjigae"],
    "bbq": ["samgyeopsal", "woo samgyup", "bulgogi"],
    "non_spicy": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "meltique tenderloin", "odeng"],
    "tofu_based": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae"]
}

@st.cache_data
def load_data():
    df = pd.read_csv("review_sentiment.csv")
    df = df[df["sentiment"] == "positive"]
    df["menu"] = df["menu"].str.lower().replace(menu_aliases)

    menu_stats = df.groupby("menu").agg(
        count=("menu", "count"),
        avg_sentiment=("compound_score", "mean")
    ).reset_index()

    scaler = MinMaxScaler()
    menu_stats[["count_norm", "sentiment_norm"]] = scaler.fit_transform(menu_stats[["count", "avg_sentiment"]])
    menu_stats["score"] = (menu_stats["count_norm"] + menu_stats["sentiment_norm"]) / 2
    return menu_stats

menu_stats = load_data()
if menu_stats is None:
    st.stop()

# ====== UTILS ======
def correct_spelling(text):
    return str(TextBlob(text).correct())

def detect_category(text):
    text = text.lower()
    for keyword, category in keyword_aliases.items():
        if keyword in text:
            return category
    return None

def fuzzy_match_menu(text, menu_list):
    text = text.lower()
    for menu in menu_list:
        if all(word in text for word in menu.split()):
            return menu
    return None

def detect_negative_rule(text):
    return any(neg in text for neg in ["don't", "not", "dislike", "too", "hate", "worst", "bad"])

def is_category_only_input(text):
    return all(word in keyword_aliases for word in text.lower().split())

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=MAX_LEN)
    bert_output = bert_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    ).last_hidden_state
    preds = sentiment_model.predict(bert_output, verbose=0)
    return int(preds[0][0] > 0.5)

# ====== UI ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("👩‍🍳 GoStop Korean BBQ Menu Recommender")
st.markdown("Ask something like **'recommend me non-spicy food'** or **'how about odeng?'**")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your request here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    raw_input = user_input.lower()
    corrected_input = user_input if len(user_input.split()) <= 2 else correct_spelling(user_input)
    corrected_lower = corrected_input.lower()

    matched_menu = fuzzy_match_menu(raw_input, menu_actual)
    category = detect_category(raw_input)
    is_category_input = is_category_only_input(corrected_lower)
    explicit_negative = detect_negative_rule(raw_input)

    sentiment_pred = predict_sentiment(corrected_input)
    is_negative = sentiment_pred == 0 if matched_menu or category else False
    show_mood = any(word in raw_input for word in ["love", "like", "want", "enjoy"]) and not is_negative

    response = ""
    if show_mood:
        response += "😊 You're in a great mood!\n\n"
    elif is_negative:
        response += "😕 Got it. Let's avoid that.\n\n"

    if matched_menu:
        if is_negative:
            response += f"❌ You don't like **{matched_menu}**. Let's try something else."
        else:
            row = menu_stats[menu_stats["menu"] == matched_menu].iloc[0]
            response += f"🍽️ **{matched_menu.title()}** has **{row['count']} reviews** with average sentiment **{row['avg_sentiment']:.2f}**. Recommended!"
    elif category:
        suggestions = menu_stats.copy()
        if is_negative:
            suggestions = suggestions[~suggestions["menu"].isin(menu_categories.get(category, []))]
        else:
            suggestions = suggestions[suggestions["menu"].isin(menu_categories.get(category, []))]

        if suggestions.empty:
            response += "🙁 Sorry, no menu found for that."
        else:
            top = suggestions.sort_values("score", ascending=False).head(3)
            response += f"🔥 Here are top picks in **{category}** category:<br><table><tr><th>Rank</th><th>Menu</th><th>Sentiment</th><th>Reviews</th></tr>"
            for i, (_, row) in enumerate(top.iterrows(), 1):
                response += f"<tr><td>{i}</td><td>{row['menu'].title()}</td><td>{row['avg_sentiment']:.2f}</td><td>{int(row['count'])}</td></tr>"
            response += "</table>"
    else:
        top = menu_stats.sort_values("score", ascending=False).head(3)
        response += "🤔 Not sure what you meant. Here's our top 3 picks:<br><table><tr><th>Rank</th><th>Menu</th><th>Sentiment</th><th>Reviews</th></tr>"
        for i, (_, row) in enumerate(top.iterrows(), 1):
            response += f"<tr><td>{i}</td><td>{row['menu'].title()}</td><td>{row['avg_sentiment']:.2f}</td><td>{int(row['count'])}</td></tr>"
        response += "</table>"

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for sender, message in st.session_state.chat_history:
    if sender == "Bot":
        st.markdown(f"**{sender}:** {message}", unsafe_allow_html=True)
    else:
        st.markdown(f"**{sender}:** {message}")
