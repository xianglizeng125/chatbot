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

# ====== CONFIG ======
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")
MAX_LEN = 100
GOOGLE_DRIVE_ZIP_ID = "1fKrPXGWGMn0mkwLPGOAgEH7f9_efR1Sd"
ZIP_PATH = "nlp_assets.zip"
EXTRACT_PATH = "nlp_assets"

# ====== DEFINE CUSTOM FUNCTION USED IN MODEL ======
def extract_bert_embeddings(inputs):
    return inputs[:, 0, :]

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
                return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(EXTRACT_PATH, "tokenizer_distilbert"), local_files_only=True)
        model = load_model(
            os.path.join(EXTRACT_PATH, "model.keras"),
            custom_objects={"extract_bert_embeddings": extract_bert_embeddings}
        )
        st.success("✅ Model and tokenizer loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading tokenizer/model: {e}")
        return None, None

    return tokenizer, model

tokenizer, sentiment_model = download_and_load_assets()

# ====== SIDEBAR ======
with st.sidebar:
    if os.path.exists("gostop.jpeg"):
        st.image(Image.open("gostop.jpeg"), use_container_width=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ====== MENU & ALIASES ======
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

# ====== DATA LOADER ======
@st.cache_data
def load_data():
    if not os.path.exists("review_sentiment.csv"):
        st.error("❌ 'review_sentiment.csv' not found.")
        return None

    df = pd.read_csv("review_sentiment.csv")
    df["menu"] = df["menu"].str.lower().replace(menu_aliases)
    df = df[df["sentiment"] == "positive"]

    menu_stats = df.groupby("menu").agg(
        count=("menu", "count"),
        avg_sentiment=("compound_score", "mean")
    ).reset_index()

    scaler = MinMaxScaler()
    menu_stats[["count_norm", "sentiment_norm"]] = scaler.fit_transform(
        menu_stats[["count", "avg_sentiment"]]
    )
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
    negative_keywords = ["don't", "not", "dislike", "too", "hate", "worst", "bad"]
    return any(neg in text for neg in negative_keywords)

def is_category_only_input(text):
    words = text.lower().split()
    return all(word in keyword_aliases for word in words)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=MAX_LEN)
    preds = sentiment_model.predict(inputs['input_ids'], verbose=0)
    return int(preds[0][0] > 0.5)

# ====== CHATBOT UI ======
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
    is_negative = False
    sentiment_pred = "SKIPPED"

    if matched_menu and not explicit_negative:
        is_negative = False
    elif matched_menu and explicit_negative:
        is_negative = True
    elif matched_menu:
        sentiment_pred = predict_sentiment(corrected_input)
        is_negative = sentiment_pred == 0
    elif category and not explicit_negative and is_category_input:
        is_negative = False
    elif category and explicit_negative:
        is_negative = True
    else:
        sentiment_pred = predict_sentiment(corrected_input)
        is_negative = sentiment_pred == 0

    st.session_state.chat_history.append(("You", user_input))

    if matched_menu:
        if is_negative:
            response = f"❌ Oh no! Sounds like you don't like **{matched_menu}**. Let's try something else?"
        elif matched_menu in menu_stats["menu"].values:
            row = menu_stats[menu_stats["menu"] == matched_menu].iloc[0]
            response = f"🍽️ **{matched_menu.title()}** has **{row['count']} reviews** with average sentiment **{row['avg_sentiment']:.2f}**. Recommended! 🎉"
        else:
            response = f"✅ Great! **{matched_menu}** is a tasty choice!"
    elif category:
        suggestions = menu_stats[menu_stats["menu"].isin(menu_actual)].copy()
        if is_negative:
            suggestions = suggestions[~suggestions["menu"].isin(menu_categories.get(category, []))]
        else:
            suggestions = suggestions[suggestions["menu"].isin(menu_categories.get(category, []))]

        if suggestions.empty:
            response = "🙁 Sorry, I couldn't find any matching menu!"
        else:
            top = suggestions.sort_values("score", ascending=False).iloc[0]
            response = f"🍽️ How about trying **{top['menu']}**?"
    else:
        if is_negative:
            response = "Got it! You don't like that. Let me think of something else next time."
        else:
            top = menu_stats.sort_values("score", ascending=False).iloc[0]
            response = f"🤔 Not sure what you meant, but maybe try **{top['menu']}**?"

    st.session_state.chat_history.append(("Bot", response))

# ====== DISPLAY CHAT ======
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")
