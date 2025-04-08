import os
import zipfile
import gdown
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, TFDistilBertModel
from tensorflow.keras.models import load_model

# ========== CONFIG ==========
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")
MAX_LEN = 100
GOOGLE_DRIVE_ZIP_ID = "13kZQFNNE8BI9Ix519l6fZ9lxIpKr4m9J"
ZIP_PATH = "nlp_assets.zip"
EXTRACT_PATH = "nlp_assets"

# ========== DOWNLOAD & LOAD ==========
@st.cache_resource
def download_and_load_assets():
    if not os.path.exists(EXTRACT_PATH):
        with st.spinner("📥 Downloading model & tokenizer..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ZIP_ID}"
            gdown.download(url, ZIP_PATH, quiet=False)
            with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_PATH)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(EXTRACT_PATH, "tokenizer_distilbert"),
            local_files_only=True
        )
        bert_model = TFDistilBertModel.from_pretrained(
            os.path.join(EXTRACT_PATH, "bert_model"),
            local_files_only=True
        )
        sentiment_model = load_model(os.path.join(EXTRACT_PATH, "model.keras"))
        return tokenizer, bert_model, sentiment_model
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}")
        return None, None, None

tokenizer, bert_model, sentiment_model = download_and_load_assets()

# ✅ Perbaikan penting agar tidak error jika gagal load
if not all([tokenizer, bert_model, sentiment_model]):
    st.error("❌ Failed to load model/tokenizer. App will stop.")
    st.stop()

# ========== SIDEBAR ==========
with st.sidebar:
    if os.path.exists("gostop.jpeg"):
        st.image(Image.open("gostop.jpeg"), use_container_width=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ========== MENU ==========
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

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("review_sentiment.csv")
    df = df[df["sentiment"] == "positive"]
    df["menu"] = df["menu"].str.lower().replace(menu_aliases)

    stats = df.groupby("menu").agg(
        count=("menu", "count"),
        avg_sentiment=("compound_score", "mean")
    ).reset_index()

    scaler = MinMaxScaler()
    stats[["count_norm", "sentiment_norm"]] = scaler.fit_transform(stats[["count", "avg_sentiment"]])
    stats["score"] = (stats["count_norm"] + stats["sentiment_norm"]) / 2
    return stats

menu_stats = load_data()

# ========== UTILS ==========
def correct_spelling(text):
    return str(TextBlob(str(text)).correct())

def detect_category(text):
    text = str(text).lower()
    for keyword, category in keyword_aliases.items():
        if keyword in text:
            return category
    return None

def fuzzy_match_menu(text, menu_list):
    text = str(text).lower()
    for menu in menu_list:
        if all(word in text for word in menu.split()):
            return menu
    return None

def detect_negative_rule(text):
    text = str(text).lower()
    return any(neg in text for neg in ["don't", "not", "dislike", "too", "hate", "worst", "bad"])

def is_category_only_input(text):
    return all(word in keyword_aliases for word in text.lower().split())

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=MAX_LEN)
    bert_output = bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
    preds = sentiment_model.predict(bert_output, verbose=0)
    return int(preds[0][0] > 0.5)

# ========== CHATBOT ==========
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
    is_negative = False

    if matched_menu:
        is_negative = explicit_negative or predict_sentiment(corrected_input) == 0
    elif category:
        is_negative = explicit_negative
    else:
        is_negative = predict_sentiment(corrected_input) == 0

    show_mood = any(word in raw_input for word in ["love", "like", "want", "enjoy"]) and not is_negative
    sentiment_note = "😊 Awesome! You're in a good mood! " if show_mood else "😕 No worries! I got you. " if is_negative else ""

    recommended = None
    if matched_menu:
        if is_negative:
            recommended = menu_stats[menu_stats["menu"] != matched_menu].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"❌ You don't like <strong>{matched_menu.title()}</strong>? Try these:"
        elif matched_menu in menu_stats["menu"].values:
            row = menu_stats[menu_stats["menu"] == matched_menu].iloc[0]
            response = sentiment_note + f"🍽️ <strong>{matched_menu.title()}</strong> has <strong>{row['count']} reviews</strong> with average sentiment <strong>{row['avg_sentiment']:.2f}</strong>. Recommended!"
        else:
            recommended = menu_stats.sort_values("score", ascending=False).head(3)
            response = sentiment_note + "🤔 Hmm, not sure about that. Here are our top picks!"
    elif category:
        matched = menu_categories.get(category, [])
        if is_negative:
            recommended = menu_stats[~menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"🙅 Skipping <strong>{category.replace('_',' ')}</strong>? Try these:"
        else:
            recommended = menu_stats[menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"🔥 Great! Top <strong>{category.replace('_',' ')}</strong> picks:"
    else:
        recommended = menu_stats.sort_values("score", ascending=False).head(3)
        response = sentiment_note + "🤔 Couldn't match it well. Here's our top 3!"

    if recommended is not None:
        response += "<table><thead><tr><th>Rank</th><th>Menu</th><th>Sentiment</th><th>Reviews</th></tr></thead><tbody>"
        for i, row in enumerate(recommended.itertuples(), 1):
            response += f"<tr><td>{i}</td><td>{row.menu.title()}</td><td>{row.avg_sentiment:.2f}</td><td>{row.count}</td></tr>"
        response += "</tbody></table>"

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for sender, msg in st.session_state.chat_history:
    if sender == "Bot":
        st.markdown(msg, unsafe_allow_html=True)
    else:
        st.markdown(f"**{sender}:** {msg}")
