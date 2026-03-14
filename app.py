import streamlit as st
import numpy as np
import re
import pickle
import os
import sys
import types

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(
    page_title="Yelp Sentiment Analyzer",
    page_icon="⭐",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────
MAX_SEQUENCE_LENGTH = 200
VOCAB_SIZE          = 25000
EMBEDDING_DIM       = 100
LSTM_UNITS          = 64
DROPOUT_RATE        = 0.5

WEIGHTS_ID     = "18XUT9YKVeyDRZ91bL-LAwft5p5C8veo8"
TOKENIZER_ID   = "1kunztOgHS8Yoy78BWcXlVgoy3fjWInCg"
WEIGHTS_PATH   = "models/bilstm_weights.weights.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# ── File Download ─────────────────────────────────────────────────
def download_file(file_id, output_path, label):
    try:
        import gdown
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise ValueError(f"{label} download appears incomplete.")
    except Exception as e:
        st.error(
            f"❌ Failed to download {label}.\n\n"
            f"**Error:** `{e}`\n\n"
            "This may be due to Google Drive quota limits. "
            "Please try again in a few minutes."
        )
        st.stop()

def ensure_files():
    if not os.path.exists(WEIGHTS_PATH) or os.path.getsize(WEIGHTS_PATH) < 1000:
        with st.spinner("⬇️ Downloading model weights (first run only)..."):
            download_file(WEIGHTS_ID, WEIGHTS_PATH, "model weights")
    if not os.path.exists(TOKENIZER_PATH) or os.path.getsize(TOKENIZER_PATH) < 100:
        with st.spinner("⬇️ Downloading tokenizer (first run only)..."):
            download_file(TOKENIZER_ID, TOKENIZER_PATH, "tokenizer")

# ── Model Loading ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Embedding, Bidirectional,
        LSTM, Dense, Dropout, SpatialDropout1D
    )
    from tensorflow.keras.regularizers import l2

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), name='Input')
    x   = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM,
                    trainable=False, name='GloVe_Embedding')(inp)
    x   = SpatialDropout1D(0.4, name='SpatialDropout')(x)
    x   = Bidirectional(
              LSTM(LSTM_UNITS, return_sequences=True,
                   dropout=0.3, recurrent_dropout=0.2),
              name='Bi_LSTM_1')(x)
    x   = Bidirectional(
              LSTM(32, return_sequences=False,
                   dropout=0.3, recurrent_dropout=0.2),
              name='Bi_LSTM_2')(x)
    x   = Dropout(DROPOUT_RATE, name='Dropout_1')(x)
    x   = Dense(64, activation='relu',
                kernel_regularizer=l2(0.001), name='Dense_1')(x)
    x   = Dropout(DROPOUT_RATE, name='Dropout_2')(x)
    out = Dense(3, activation='softmax', name='Output')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.load_weights(WEIGHTS_PATH)
    return model

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    import tensorflow as tf

    # ── Fix: patch keras.src.legacy so pickle.load() doesn't crash ──
    # tokenizer.pkl was saved in Keras 3 (Colab) but we run Keras 2
    # (tensorflow==2.15.0). Inject fake module path before unpickling.
    real_cls = tf.keras.preprocessing.text.Tokenizer

    for mod_name in [
        "keras.src",
        "keras.src.legacy",
        "keras.src.legacy.preprocessing",
        "keras.src.legacy.preprocessing.text",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    sys.modules["keras.src.legacy.preprocessing.text"].Tokenizer = real_cls
    # ────────────────────────────────────────────────────────────────

    with open(TOKENIZER_PATH, "rb") as f:
        tok = pickle.load(f)

    return tok

@st.cache_resource(show_spinner=False)
def load_cleaning_tools():
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    lem = WordNetLemmatizer()
    sw  = set(stopwords.words('english')) - {'not', 'no', 'never', 'nor'}
    return lem, sw

# ── Text Cleaning ─────────────────────────────────────────────────
html_pat    = re.compile(r'<.*?>')
url_pat     = re.compile(r'http\S+|www\.\S+')
email_pat   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
phone_pat   = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
special_pat = re.compile(r'[^a-z\s]')
space_pat   = re.compile(r'\s+')

SLANG = {
    "idk": "i do not know", "tbh": "to be honest",
    "gr8": "great",          "lol": "laughing",
    "omg": "oh my god",      "btw": "by the way",
    "imo": "in my opinion",  "asap": "as soon as possible"
}

def clean_text(text, lem, sw):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = email_pat.sub('[EMAIL]', text)
    text = phone_pat.sub('[PHONE]', text)
    text = html_pat.sub(' ', text)
    text = url_pat.sub('', text)
    for slang, exp in SLANG.items():
        text = re.sub(rf'\b{slang}\b', exp, text)
    text = special_pat.sub('', text)
    text = space_pat.sub(' ', text).strip()
    words = [lem.lemmatize(w) for w in text.split() if w not in sw]
    return ' '.join(words)

# ── Prediction ────────────────────────────────────────────────────
def predict(raw_text, model, tokenizer, lem, sw):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(raw_text, lem, sw)
    if not cleaned:
        return np.array([0.33, 0.34, 0.33]), cleaned

    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        seq, maxlen=MAX_SEQUENCE_LENGTH,
        padding='post', truncating='post'
    )
    probs = model.predict(padded, verbose=0)[0]
    return probs, cleaned

# ── UI ────────────────────────────────────────────────────────────
st.title("⭐ Yelp Review Sentiment Analyzer")
st.markdown("Powered by **Bidirectional LSTM** trained on 45,000 Yelp reviews")
st.markdown("---")

ensure_files()

with st.spinner("Loading model and NLP tools..."):
    try:
        model     = load_model()
        tokenizer = load_tokenizer()
        lem, sw   = load_cleaning_tools()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Unexpected error while loading: {e}")
        st.exception(e)
        st.stop()

if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

st.subheader("📝 Enter a Review")

ex1, ex2, ex3 = st.columns(3)
if ex1.button("😡 Negative"):
    st.session_state["review_text"] = (
        "The food was absolutely disgusting and the service was "
        "incredibly rude. Never coming back."
    )
    st.rerun()
if ex2.button("😐 Neutral"):
    st.session_state["review_text"] = (
        "Average place. Nothing special but not terrible either. "
        "Food was okay I guess."
    )
    st.rerun()
if ex3.button("😊 Positive"):
    st.session_state["review_text"] = (
        "Best restaurant I have been to in years! The pasta was "
        "fresh and delicious. Highly recommend!"
    )
    st.rerun()

review_text = st.text_area(
    label="Paste your Yelp review here:",
    value=st.session_state["review_text"],
    placeholder="e.g. The food was absolutely amazing...",
    height=150,
    key="input_box"
)

_, col2, _ = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze_btn:
    if not review_text.strip():
        st.warning("⚠️ Please enter a review before clicking Analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                probs, cleaned = predict(review_text, model, tokenizer, lem, sw)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)
                st.stop()

        labels = ["Negative", "Neutral", "Positive"]
        emojis = ["😡", "😐", "😊"]
        colors = ["#ff4b4b", "#ffa500", "#00cc44"]
        predicted_idx = int(np.argmax(probs))

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.markdown(
            f"""
            <div style="
                background-color:{colors[predicted_idx]}22;
                border-left:6px solid {colors[predicted_idx]};
                padding:20px; border-radius:8px; margin-bottom:20px;">
                <h2 style="color:{colors[predicted_idx]};margin:0">
                    {emojis[predicted_idx]} {labels[predicted_idx]}
                </h2>
                <p style="margin:5px 0 0 0;font-size:16px;">
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

        with st.expander("🔍 Preprocessed text (after cleaning pipeline)"):
            if cleaned:
                st.code(cleaned, language=None)
            else:
                st.warning("Cleaned text was empty.")

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
                "May contain both positive and negative signals."
            )
        else:
            st.success(
                "🌟 **Positive Review** — Strong positive sentiment. "
                "Great candidate for featuring or marketing use."
            )

st.markdown("---")
st.caption(
    f"Model: Bidirectional LSTM  |  Sequence length: {MAX_SEQUENCE_LENGTH}  |  "
    "Dataset: Yelp Review Full (HuggingFace)  |  "
    "Classes: Negative (1–2★) · Neutral (3★) · Positive (4–5★)"
)
