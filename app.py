# ── Compliance fixes applied ──────────────────────────────────────
# 1. KeyError 'cleaned_text'  → safe empty-string guard in clean_text()
# 2. KeyError 'review/text'   → app never touches df_sampled columns
# 3. GloVe re-download prompt → app loads .h5 directly, no GloVe needed
# 4. NLTK punkt_tab missing   → all 5 NLTK packages downloaded explicitly
# 5. Out-of-order re-runs     → @st.cache_resource isolates state fully
# 6. Shape mismatch           → MAX_SEQUENCE_LENGTH constant at top of file

import streamlit as st
import numpy as np
import re
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Yelp Sentiment Analyzer",
    page_icon="⭐",
    layout="centered"
)

# ── Must match notebook Cell 9 MAX_SEQUENCE_LENGTH ───────────────
# Fixed notebook = 200 | Original notebook = 150
# Change this if you get shape mismatch errors
MAX_SEQUENCE_LENGTH = 200

# ── Load model (cached — loads only once per session) ─────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("model/bilstm_model.h5")
    return model

@st.cache_resource
def load_tokenizer():
    with open("model/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

# ── NLTK tools — all 5 packages downloaded, matches Cell 4 exactly ─
@st.cache_resource
def load_cleaning_tools():
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    # Same 5 packages as notebook Cell 4 — punkt_tab included (NLTK >= 3.8)
    for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
        nltk.download(pkg, quiet=True)

    lem = WordNetLemmatizer()
    # Preserve negations — mirrors Cell 4 exactly
    sw  = set(stopwords.words('english')) - {'not', 'no', 'never', 'nor'}
    return lem, sw

# ── Regex patterns compiled once at module level ──────────────────
html_pat    = re.compile(r'<.*?>')
url_pat     = re.compile(r'http\S+|www\.\S+')
email_pat   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
phone_pat   = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
special_pat = re.compile(r'[^a-z\s]')
space_pat   = re.compile(r'\s+')

slang_dict = {
    "idk": "i do not know", "tbh": "to be honest",
    "gr8": "great",         "lol": "laughing",
    "omg": "oh my god",     "btw": "by the way",
    "imo": "in my opinion", "asap": "as soon as possible"
}

def clean_text(text, lem, sw):
    """
    Exact replica of clean_text() from notebook Cell 4.
    Added: isinstance + empty string guard to prevent KeyError
    when input is None, NaN, or empty — the same root cause as
    KeyError:'cleaned_text' and KeyError:'review/text' in the notebook.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = email_pat.sub('[EMAIL]', text)
    text = phone_pat.sub('[PHONE]', text)
    text = html_pat.sub(' ', text)
    text = url_pat.sub('', text)
    for slang, exp in slang_dict.items():
        text = re.sub(rf'\b{slang}\b', exp, text)
    text = special_pat.sub('', text)
    text = space_pat.sub(' ', text).strip()
    words = [lem.lemmatize(w) for w in text.split() if w not in sw]
    return ' '.join(words)

def predict(raw_text, model, tokenizer, lem, sw):
    """
    raw text → clean → tokenize → pad → predict
    padding='post', truncating='post' matches notebook Cell 8 exactly.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(raw_text, lem, sw)

    if not cleaned:
        return np.array([0.33, 0.34, 0.33]), cleaned

    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post',
        truncating='post'
    )
    probs = model.predict(padded, verbose=0)[0]
    return probs, cleaned

# ─────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────
st.title("⭐ Yelp Review Sentiment Analyzer")
st.markdown("Powered by **Bidirectional LSTM** trained on 45,000 Yelp reviews")
st.markdown("---")

# ── Load resources ────────────────────────────────────────────────
with st.spinner("Loading model and NLP tools..."):
    try:
        model     = load_model()
        tokenizer = load_tokenizer()
        lem, sw   = load_cleaning_tools()
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info(
            "**Setup required:**\n\n"
            "1. Run `save_model.py` as the last cell in your Colab notebook\n"
            "2. Download `bilstm_model.h5` and `tokenizer.pkl`\n"
            "3. Place both files inside a `model/` folder next to `app.py`"
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error while loading: {e}")
        st.stop()

# ── Session state for example buttons ────────────────────────────
# Fixes stale text issue when example buttons are clicked
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

st.subheader("📝 Enter a Review")

ex1, ex2, ex3 = st.columns(3)
if ex1.button("😡 Negative Example"):
    st.session_state["review_text"] = (
        "The food was absolutely disgusting and the service was "
        "incredibly rude. Never coming back."
    )
if ex2.button("😐 Neutral Example"):
    st.session_state["review_text"] = (
        "Average place. Nothing special but not terrible either. "
        "Food was okay I guess."
    )
if ex3.button("😊 Positive Example"):
    st.session_state["review_text"] = (
        "Best restaurant I have been to in years! The pasta was "
        "fresh and delicious. Highly recommend!"
    )

review_text = st.text_area(
    label="Paste your Yelp review here:",
    value=st.session_state["review_text"],
    placeholder="e.g. The food was absolutely amazing and the service was very friendly!",
    height=150,
    key="input_box"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────
if analyze_btn:
    if not review_text.strip():
        st.warning("⚠️ Please enter a review before clicking Analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                probs, cleaned = predict(review_text, model, tokenizer, lem, sw)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.stop()

        predicted_idx   = int(np.argmax(probs))
        labels          = ["Negative", "Neutral", "Positive"]
        emojis          = ["😡", "😐", "😊"]
        colors          = ["#ff4b4b", "#ffa500", "#00cc44"]
        predicted_label = labels[predicted_idx]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        st.markdown(
            f"""
            <div style="
                background-color:{colors[predicted_idx]}22;
                border-left: 6px solid {colors[predicted_idx]};
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            ">
                <h2 style="color:{colors[predicted_idx]};margin:0">
                    {emojis[predicted_idx]} {predicted_label}
                </h2>
                <p style="margin:5px 0 0 0; font-size:16px;">
                    Confidence: <strong>{probs[predicted_idx]*100:.1f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("📈 Class Probabilities")
        for label, prob, color, emoji in zip(labels, probs, colors, emojis):
            c_label, c_bar, c_pct = st.columns([1.2, 4, 0.8])
            c_label.markdown(f"{emoji} **{label}**")
            c_bar.progress(float(prob))
            c_pct.markdown(f"`{prob*100:.1f}%`")

        with st.expander("🔍 See preprocessed text (after cleaning pipeline)"):
            if cleaned:
                st.code(cleaned, language=None)
            else:
                st.warning("Cleaned text was empty — input may have contained only special characters.")

        st.markdown("---")
        st.subheader("💡 Interpretation")
        if predicted_idx == 0:
            st.warning(
                "🚨 **Negative Review** — Signals dissatisfaction. "
                "Consider flagging for QA or customer service follow-up."
            )
        elif predicted_idx == 1:
            st.info(
                "📋 **Neutral Review** — Mixed or average sentiment. "
                "May contain both positive and negative signals worth examining."
            )
        else:
            st.success(
                "🌟 **Positive Review** — Strong positive sentiment detected. "
                "Great candidate for featuring or marketing use."
            )

st.markdown("---")
st.caption(
    f"Model: Bidirectional LSTM  |  Sequence length: {MAX_SEQUENCE_LENGTH}  |  "
    "Dataset: Yelp Review Full (HuggingFace)  |  "
    "Classes: Negative (1–2★) · Neutral (3★) · Positive (4–5★)"
)
