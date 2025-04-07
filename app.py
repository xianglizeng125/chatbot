import os
import zipfile
import gdown
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer
from tensorflow.keras.models import load_model

# ========= CONFIG =========
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")
MAX_LEN = 100
GOOGLE_DRIVE_ZIP_ID = "1fKrPXGWGMn0mkwLPGOAgEH7f9_efR1Sd"
ZIP_PATH = "nlp_assets.zip"
EXTRACT_PATH = "nlp_assets"

# ========= CUSTOM FUNCTION (for Lambda Layer) =========
def extract_bert_embeddings(inputs):
    return inputs[:, 0, :]

# ========= DOWNLOAD & LOAD =========
@st.cache_resource
def download_and_load_assets():
    if not os.path.exists(EXTRACT_PATH):
        with st.spinner("📥 Downloading model & tokenizer..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ZIP_ID}"
            gdown.download(url, ZIP_PATH, quiet=False)
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_PATH)

    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(EXTRACT_PATH, "tokenizer_distilbert"), local_files_only=True)
        model = load_model(
            os.path.join(EXTRACT_PATH, "model.keras"),
            custom_objects={"extract_bert_embeddings": extract_bert_embeddings}
        )
        st.success("✅ Model and tokenizer loaded!")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load model/tokenizer: {e}")
        return None, None

tokenizer, sentiment_model = download_and_load_assets()

# ========= SIDEBAR =========
with st.sidebar:
    if os.path.exists("gostop.jpeg"):
        st.image(Image.open("gostop.jpeg"), use_container_width=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ========= MENU & ALIASES =========
menu_actual = [
    "soondubu jjigae", "prawn soondubu jjigae", "kimchi jjigae", "tofu jjigae",
    "samgyeopsal", "spicy samgyeopsal", "woo samgyup", "spicy woo samgyup",
    "bulgogi", "dak bulgogi", "spicy dak bulgogi", "meltique tenderloin", "odeng",
    "beef soondubu jjigae", "pork soondubu jjigae"
]

menu_aliases = {
    "soondubu": "soondubu jjigae",
    "suundobu": "soondubu jjigae",
    "beef soondubu": "beef soondubu jjigae",
    "pork soondubu": "pork soondubu jjigae",
    "soondubu jigae": "soondubu jjigae"
}

keyword_aliases = {
    "spicy": "spicy", "meat": "meat", "soup": "soup", "seafood": "seafood",
    "beef": "beef", "pork": "pork", "bbq": "bbq", "non-spicy": "non_spicy", "tofu": "tofu_based"
}

menu_categories = {
    "spicy": ["spicy samgyeopsal", "spicy woo samgyup", "spicy dak bulgogi", "kimchi jjigae"],
    "meat": ["samgyeopsal", "woo samgyup", "bulgogi", "dak bulgogi", "meltique tenderloin"],
    "soup": ["kimchi jjigae", "tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae", "prawn soondubu jjigae"],
    "seafood": ["prawn soondubu jjigae", "odeng"],
    "beef": ["bulgogi", "beef soondubu jjigae", "meltique tenderloin"],
    "pork": ["samgyeopsal", "spicy samgyeopsal", "pork soondubu jjigae"],
    "bbq": ["samgyeopsal", "woo samgyup", "bulgogi"],
    "non_spicy": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "meltique tenderloin", "odeng"],
    "tofu_based": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae"]
}

# ========= LOAD SENTIMENT DATA =========
@st.cache_data
def load_data():
    if not os.path.exists("review_sentiment.csv"):
        st.error("❌ 'review_sentiment.csv' not found.")
        return None

    df = pd.read_csv("review_sentiment.csv")
    df["menu"] = df["menu"].str.lower().replace(menu_aliases)
    df = df[df["sentiment"] == "positive"]

    stats = df.groupby("menu").agg(
        count=("menu", "count"),
        avg_sentiment=("compound_score", "mean")
    ).reset_index()

    scaler = MinMaxScaler()
    stats[["count_norm", "sentiment_norm"]] = scaler.fit_transform(
        stats[["count", "avg_sentiment"]]
    )
    stats["score"] = (stats["count_norm"] + stats["sentiment_norm"]) / 2
    return stats

menu_stats = load_data()
if menu_stats is None:
    st.stop()

# ========= UTILS =========
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
    negative_keywords = ["don't", "not", "dislike", "too", "hate", "worst", "bad"]
    return any(neg in text for neg in negative_keywords)

def is_category_only_input(text):
    return all(word in keyword_aliases for word in text.lower().split())

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=MAX_LEN)
    preds = sentiment_model.predict([inputs['input_ids'], inputs['attention_mask']], verbose=0)
    return int(preds[0][0] > 0.5)

# ========= CHATBOT UI =========
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
    matched_menu = fuzzy_match_menu(raw_input, menu_actual)
    category = detect_category(raw_input)
    is_category_input = is_category_only_input(corrected_input)
    explicit_negative = detect_negative_rule(raw_input)

    sentiment_pred = None
    is_negative = False
    if matched_menu:
        sentiment_pred = predict_sentiment(corrected_input)
        is_negative = explicit_negative or sentiment_pred == 0
    elif category or is_category_input:
        sentiment_pred = predict_sentiment(corrected_input)
        is_negative = explicit_negative or sentiment_pred == 0
    else:
        sentiment_pred = predict_sentiment(corrected_input)
        is_negative = sentiment_pred == 0

    show_mood = any(word in raw_input for word in ["love", "like", "want", "enjoy"]) and not is_negative
    sentiment_note = "😊 You're in a good mood! " if show_mood else "😕 Gotcha! " if is_negative else ""

    response = ""
    recommendations = None

    if matched_menu:
        if is_negative:
            recommendations = menu_stats[menu_stats["menu"] != matched_menu].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"You don't like **{matched_menu}**? Try these instead:"
        else:
            if matched_menu in menu_stats["menu"].values:
                row = menu_stats[menu_stats["menu"] == matched_menu].iloc[0]
                response = sentiment_note + f"🍽️ **{matched_menu}** has **{row['count']} reviews** with avg sentiment **{row['avg_sentiment']:.2f}**. Recommended!"
            else:
                response = f"✅ **{matched_menu}** is on the menu!"
    elif category:
        matched = menu_categories.get(category, [])
        if is_negative:
            recommendations = menu_stats[~menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = f"🙅 Avoiding **{category}**? Here’s something else:"
        else:
            recommendations = menu_stats[menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"🔥 You might like these **{category}** dishes:"
    else:
        recommendations = menu_stats.sort_values("score", ascending=False).head(3)
        response = sentiment_note + "🤔 Unsure what you meant. Here’s our top 3!"

    if recommendations is not None:
        response += "<table><thead><tr><th>Rank</th><th>Menu</th><th>Sentiment</th><th>Reviews</th></tr></thead><tbody>"
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            response += f"<tr><td>{i}</td><td>{row['menu'].title()}</td><td>{row['avg_sentiment']:.2f}</td><td>{int(row['count'])}</td></tr>"
        response += "</tbody></table>"

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# ========= DISPLAY CHAT =========
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**{sender}:** {msg}")
    else:
        st.markdown(msg, unsafe_allow_html=True)
