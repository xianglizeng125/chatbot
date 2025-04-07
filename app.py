
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
from tensorflow.keras.models import load_model

# ====== CONFIG ======
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")
MAX_LEN = 100
GOOGLE_DRIVE_ZIP_ID = "1fKrPXGWGMn0mkwLPGOAgEH7f9_efR1Sd"
ZIP_PATH = "nlp_assets.zip"
EXTRACT_PATH = "nlp_assets"

def extract_bert_embeddings(inputs):
    return inputs[:, 0, :]

# ====== DOWNLOAD & LOAD ASSETS ======
@st.cache_resource
def download_and_load_assets():
    if not os.path.exists(EXTRACT_PATH):
        with st.spinner("📥 Downloading model & tokenizer..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ZIP_ID}"
            gdown.download(url, ZIP_PATH, quiet=False)
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_PATH)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(EXTRACT_PATH, "tokenizer_distilbert"), local_files_only=True)
    bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    sentiment_model = load_model(
        os.path.join(EXTRACT_PATH, "model.keras"),
        custom_objects={"extract_bert_embeddings": extract_bert_embeddings},
        compile=False
    )
    return tokenizer, bert_model, sentiment_model

tokenizer, bert_model, sentiment_model = download_and_load_assets()

# SIDEBAR
with st.sidebar:
    if os.path.exists("gostop.jpeg"):
        st.image(Image.open("gostop.jpeg"), use_container_width=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Menu & aliases
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
    if not os.path.exists("review_sentiment.csv"):
        st.error("❌ 'review_sentiment.csv' not found.")
        return None
    df = pd.read_csv("review_sentiment.csv")
    df["menu"] = df["menu"].str.lower().replace(menu_aliases)
    df = df[df["sentiment"] == "positive"]
    stats = df.groupby("menu").agg(count=("menu", "count"), avg_sentiment=("compound_score", "mean")).reset_index()
    scaler = MinMaxScaler()
    stats[["count_norm", "sentiment_norm"]] = scaler.fit_transform(stats[["count", "avg_sentiment"]])
    stats["score"] = (stats["count_norm"] + stats["sentiment_norm"]) / 2
    return stats

menu_stats = load_data()
if menu_stats is None:
    st.stop()

def correct_spelling(text): return str(TextBlob(text).correct())
def detect_category(text): return next((v for k, v in keyword_aliases.items() if k in text.lower()), None)
def fuzzy_match_menu(text, menu_list): return next((m for m in menu_list if all(w in text.lower() for w in m.split())), None)
def detect_negative_rule(text): return any(neg in text for neg in ["don't", "not", "dislike", "too", "hate", "worst", "bad"])
def is_category_only_input(text): return all(word in keyword_aliases for word in text.lower().split())

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=MAX_LEN)
    bert_output = bert_model(**inputs).last_hidden_state
    preds = sentiment_model.predict(bert_output, verbose=0)
    return int(preds[0][0] > 0.5)

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

    if matched_menu:
        sentiment_pred = predict_sentiment(corrected_input)
        is_negative = sentiment_pred == 0 if not explicit_negative else True
    elif category and is_category_input:
        is_negative = explicit_negative
    else:
        sentiment_pred = predict_sentiment(corrected_input)
        is_negative = sentiment_pred == 0

    show_mood = any(w in raw_input for w in ["love", "like", "want", "enjoy"]) and not is_negative
    sentiment_note = "😊 You're in a good mood! " if show_mood else "😕 Got it. " if is_negative else ""

    st.session_state.chat_history.append(("You", user_input))
    recommended = None

    if matched_menu:
        matched_menu = matched_menu.strip().lower()
        if is_negative:
            recommended = menu_stats[menu_stats["menu"] != matched_menu].sort_values("score", ascending=False).head(3)
            response = f"{sentiment_note}❌ You don't like <strong>{matched_menu.title()}</strong>? Try these:"
        elif matched_menu in menu_stats["menu"].values:
            row = menu_stats[menu_stats["menu"] == matched_menu].iloc[0]
            response = f"{sentiment_note}🍽️ <strong>{matched_menu.title()}</strong> has <strong>{row['count']} reviews</strong> with avg sentiment <strong>{row['avg_sentiment']:.2f}</strong>."
        else:
            recommended = menu_stats.sort_values("score", ascending=False).head(3)
            response = f"{sentiment_note}❓ Not sure about that menu. Here's our top picks!"
    elif category:
        matched = menu_categories.get(category, [])
        if is_negative:
            recommended = menu_stats[~menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = f"🙅 Avoiding <strong>{category.replace('_', ' ').title()}</strong>? Try these:"
        else:
            recommended = menu_stats[menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = f"{sentiment_note}🔥 You might like these <strong>{category.replace('_', ' ').title()}</strong> dishes:"
    else:
        recommended = menu_stats.sort_values("score", ascending=False).head(3)
        response = f"{sentiment_note}🤔 Couldn't find anything specific. Here's our top 3!"

    if recommended is not None:
        response += "<table style='width:100%; border-collapse: collapse;'>"
        response += "<thead><tr><th style='text-align:left;'>#</th><th style='text-align:left;'>Menu</th><th>Sentiment</th><th>Reviews</th></tr></thead><tbody>"
        for i, (_, row) in enumerate(recommended.iterrows(), 1):
            response += f"<tr><td>{i}</td><td>{row['menu'].title()}</td><td>{row['avg_sentiment']:.2f}</td><td>{int(row['count'])}</td></tr>"
        response += "</tbody></table>"

    st.session_state.chat_history.append(("Bot", response))

for sender, msg in st.session_state.chat_history:
    if sender == "Bot":
        st.markdown(msg, unsafe_allow_html=True)
    else:
        st.markdown(f"**{sender}:** {msg}")
