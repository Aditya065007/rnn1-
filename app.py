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
    page_title="Sentient — Yelp Review Intelligence",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Design System ─────────────────────────────────────────────────
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --ink:        #0D0D0D;
    --ink-60:     rgba(13,13,13,0.6);
    --ink-20:     rgba(13,13,13,0.08);
    --paper:      #F5F2ED;
    --paper-dark: #EDE9E2;
    --red:        #C8392B;
    --amber:      #D4720C;
    --green:      #1A6B42;
    --red-tint:   rgba(200,57,43,0.08);
    --amber-tint: rgba(212,114,12,0.08);
    --green-tint: rgba(26,107,66,0.08);
    --serif:      'Instrument Serif', Georgia, serif;
    --sans:       'DM Sans', system-ui, sans-serif;
    --mono:       'DM Mono', monospace;
    --radius:     4px;
    --radius-lg:  10px;
}

/* ── Streamlit overrides ── */
.stApp {
    background-color: var(--paper) !important;
    font-family: var(--sans) !important;
}
.stApp > header { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
.stButton > button {
    font-family: var(--sans) !important;
}
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stStatusWidget"] { display: none; }
#MainMenu { display: none; }
footer { display: none; }
.stSpinner { display: none !important; }
div[data-testid="stMarkdownContainer"] p {
    font-family: var(--sans);
    color: var(--ink);
}

/* ── Page wrapper ── */
.page-wrapper {
    min-height: 100vh;
    background: var(--paper);
    display: flex;
    flex-direction: column;
}

/* ── Masthead ── */
.masthead {
    border-bottom: 1px solid var(--ink-20);
    padding: 0 48px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    background: var(--paper);
    z-index: 100;
}
.masthead-logo {
    font-family: var(--serif);
    font-size: 22px;
    color: var(--ink);
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.masthead-logo span {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: var(--ink-60);
    text-transform: uppercase;
    background: var(--ink-20);
    padding: 3px 8px;
    border-radius: 100px;
}
.masthead-badge {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-60);
}

/* ── Hero ── */
.hero {
    padding: 80px 48px 64px;
    max-width: 760px;
    margin: 0 auto;
    width: 100%;
}
.hero-eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--ink-60);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-eyebrow::before {
    content: '';
    display: block;
    width: 32px;
    height: 1px;
    background: var(--ink-60);
}
.hero-title {
    font-family: var(--serif);
    font-size: clamp(42px, 6vw, 68px);
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-bottom: 20px;
}
.hero-title em {
    font-style: italic;
    color: var(--ink-60);
}
.hero-subtitle {
    font-size: 16px;
    line-height: 1.6;
    color: var(--ink-60);
    font-weight: 300;
    max-width: 480px;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: var(--ink-20);
    max-width: 760px;
    margin: 0 auto;
    width: 100%;
}

/* ── Main content ── */
.main-content {
    max-width: 760px;
    margin: 0 auto;
    width: 100%;
    padding: 56px 48px;
}

/* ── Section label ── */
.section-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-60);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--ink-20);
}

/* ── Quick-fill chips ── */
.chips-row {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.chip {
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 100px;
    border: 1px solid var(--ink-20);
    background: transparent;
    color: var(--ink-60);
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.chip:hover {
    background: var(--ink);
    color: var(--paper);
    border-color: var(--ink);
}

/* ── Textarea ── */
.stTextArea textarea {
    font-family: var(--sans) !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    font-weight: 300 !important;
    color: var(--ink) !important;
    background: var(--paper-dark) !important;
    border: 1px solid var(--ink-20) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important;
    resize: vertical !important;
    transition: border-color 0.15s ease !important;
    box-shadow: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--ink) !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: var(--ink-60) !important;
    font-style: italic;
}
.stTextArea label { display: none !important; }
div[data-baseweb="textarea"] {
    border: none !important;
}

/* ── Analyze button ── */
.stButton > button {
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 14px 36px !important;
    width: 100% !important;
    height: auto !important;
    transition: opacity 0.15s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    opacity: 0.8 !important;
    background: var(--ink) !important;
    color: var(--paper) !important;
}
.stButton > button:active {
    opacity: 0.6 !important;
    transform: scale(0.99) !important;
}

/* ── Result panel ── */
.result-panel {
    margin-top: 48px;
    border-top: 1px solid var(--ink-20);
    padding-top: 48px;
    animation: fadeUp 0.4s ease forwards;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Verdict block ── */
.verdict-block {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 32px;
    align-items: start;
    margin-bottom: 48px;
}
.verdict-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.verdict-label.neg { color: var(--red); }
.verdict-label.neu { color: var(--amber); }
.verdict-label.pos { color: var(--green); }

.verdict-title {
    font-family: var(--serif);
    font-size: clamp(36px, 5vw, 54px);
    line-height: 1.0;
    letter-spacing: -0.02em;
}
.verdict-title.neg { color: var(--red); }
.verdict-title.neu { color: var(--amber); }
.verdict-title.pos { color: var(--green); }

.verdict-meta {
    font-size: 13px;
    color: var(--ink-60);
    margin-top: 8px;
    font-weight: 300;
}

.confidence-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 100px;
    font-family: var(--mono);
    font-size: 20px;
    font-weight: 500;
    letter-spacing: -0.01em;
    min-width: 100px;
    justify-content: center;
    margin-top: 4px;
}
.confidence-pill.neg {
    background: var(--red-tint);
    color: var(--red);
    border: 1px solid rgba(200,57,43,0.2);
}
.confidence-pill.neu {
    background: var(--amber-tint);
    color: var(--amber);
    border: 1px solid rgba(212,114,12,0.2);
}
.confidence-pill.pos {
    background: var(--green-tint);
    color: var(--green);
    border: 1px solid rgba(26,107,66,0.2);
}
.confidence-sub {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
    margin-top: 4px;
    color: var(--ink-60);
}

/* ── Distribution bars ── */
.dist-section {
    margin-bottom: 40px;
}
.dist-row {
    display: grid;
    grid-template-columns: 88px 1fr 52px;
    align-items: center;
    gap: 16px;
    margin-bottom: 14px;
}
.dist-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-60);
}
.dist-track {
    height: 3px;
    background: var(--ink-20);
    border-radius: 100px;
    overflow: hidden;
    position: relative;
}
.dist-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.dist-fill.neg { background: var(--red); }
.dist-fill.neu { background: var(--amber); }
.dist-fill.pos { background: var(--green); }
.dist-pct {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 500;
    text-align: right;
}
.dist-pct.neg { color: var(--red); }
.dist-pct.neu { color: var(--amber); }
.dist-pct.pos { color: var(--green); }

/* ── Insight card ── */
.insight-card {
    padding: 24px 28px;
    border-radius: var(--radius-lg);
    margin-bottom: 20px;
}
.insight-card.neg {
    background: var(--red-tint);
    border: 1px solid rgba(200,57,43,0.15);
}
.insight-card.neu {
    background: var(--amber-tint);
    border: 1px solid rgba(212,114,12,0.15);
}
.insight-card.pos {
    background: var(--green-tint);
    border: 1px solid rgba(26,107,66,0.15);
}
.insight-card-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.insight-card-label.neg { color: var(--red); }
.insight-card-label.neu { color: var(--amber); }
.insight-card-label.pos { color: var(--green); }
.insight-card-text {
    font-size: 14px;
    line-height: 1.6;
    font-weight: 300;
    color: var(--ink);
}
.insight-card-text strong { font-weight: 600; }

/* ── Preprocessed text ── */
.preprocessed-box {
    margin-top: 12px;
    background: var(--paper-dark);
    border: 1px solid var(--ink-20);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
}
.preprocessed-box code {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    line-height: 1.8 !important;
    color: var(--ink-60) !important;
    background: transparent !important;
}

/* ── Toggle (expander) ── */
.streamlit-expanderHeader {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--ink-60) !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.streamlit-expanderHeader:hover { color: var(--ink) !important; }
.streamlit-expanderContent {
    border: none !important;
    padding: 12px 0 0 !important;
}

/* ── Footer ── */
.page-footer {
    border-top: 1px solid var(--ink-20);
    padding: 28px 48px;
    max-width: 760px;
    margin: auto auto 0;
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: gap;
}
.footer-meta {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    color: var(--ink-60);
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}
.footer-dot {
    color: var(--ink-20);
}

/* ── Loading state ── */
.loading-block {
    padding: 48px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
}
.loading-spinner {
    width: 28px;
    height: 28px;
    border: 2px solid var(--ink-20);
    border-top-color: var(--ink);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.loading-text {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-60);
}

/* ── Noise texture overlay ── */
.paper-grain {
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.025;
    z-index: 999;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    background-size: 180px;
}

/* ── Misc ── */
.stAlert { display: none !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.element-container { margin: 0 !important; }
</style>

<div class="paper-grain"></div>
"""

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
        st.markdown(
            f'<div style="font-family:var(--mono,monospace);font-size:12px;'
            f'color:#C8392B;padding:20px;border:1px solid rgba(200,57,43,0.2);'
            f'border-radius:8px;background:rgba(200,57,43,0.05);">'
            f'Download failed: {e}</div>',
            unsafe_allow_html=True
        )
        st.stop()

def ensure_files():
    if not os.path.exists(WEIGHTS_PATH) or os.path.getsize(WEIGHTS_PATH) < 1000:
        with st.spinner(""):
            st.markdown(
                '<div class="loading-block">'
                '<div class="loading-spinner"></div>'
                '<div class="loading-text">Fetching model weights</div>'
                '</div>', unsafe_allow_html=True
            )
            download_file(WEIGHTS_ID, WEIGHTS_PATH, "model weights")
    if not os.path.exists(TOKENIZER_PATH) or os.path.getsize(TOKENIZER_PATH) < 100:
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
    x   = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True,
                              dropout=0.3, recurrent_dropout=0.2),
                        name='Bi_LSTM_1')(x)
    x   = Bidirectional(LSTM(32, return_sequences=False,
                              dropout=0.3, recurrent_dropout=0.2),
                        name='Bi_LSTM_2')(x)
    x   = Dropout(DROPOUT_RATE, name='Dropout_1')(x)
    x   = Dense(64, activation='relu',
                kernel_regularizer=l2(0.001), name='Dense_1')(x)
    x   = Dropout(DROPOUT_RATE, name='Dropout_2')(x)
    out = Dense(3, activation='softmax', name='Output')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(WEIGHTS_PATH)
    return model

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    import tensorflow as tf
    real_cls = tf.keras.preprocessing.text.Tokenizer
    for mod_name in ["keras.src", "keras.src.legacy",
                     "keras.src.legacy.preprocessing",
                     "keras.src.legacy.preprocessing.text"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["keras.src.legacy.preprocessing.text"].Tokenizer = real_cls
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

def predict(raw_text, model, tokenizer, lem, sw):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    cleaned = clean_text(raw_text, lem, sw)
    if not cleaned:
        return np.array([0.33, 0.34, 0.33]), cleaned
    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    probs  = model.predict(padded, verbose=0)[0]
    return probs, cleaned

# ══════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════

# Inject CSS
st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# ── Masthead ──────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-logo">
        Sentient
        <span>Beta</span>
    </div>
    <div class="masthead-badge">Bi‑LSTM · Yelp Reviews</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Sentiment Intelligence</div>
    <h1 class="hero-title">
        What does your<br><em>customer</em> really feel?
    </h1>
    <p class="hero-subtitle">
        Paste any Yelp review and our neural network classifies the 
        underlying sentiment — negative, neutral, or positive — 
        with word-level precision.
    </p>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────────
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Download files
ensure_files()

# Load model silently
with st.spinner(""):
    try:
        model     = load_model()
        tokenizer = load_tokenizer()
        lem, sw   = load_cleaning_tools()
    except Exception as e:
        st.markdown(
            f'<div style="font-family:monospace;font-size:12px;color:#C8392B;'
            f'padding:20px;border:1px solid rgba(200,57,43,0.2);border-radius:8px;">'
            f'Model failed to load: {e}</div>',
            unsafe_allow_html=True
        )
        st.stop()

# ── Input section ─────────────────────────────────────────────────
st.markdown('<div class="section-label">Review Input</div>', unsafe_allow_html=True)

# Quick-fill buttons
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("↘ Negative example", key="neg_btn"):
        st.session_state["review_text"] = (
            "The food was absolutely disgusting and the service was "
            "incredibly rude. Never coming back."
        )
        st.rerun()
with col2:
    if st.button("→ Neutral example", key="neu_btn"):
        st.session_state["review_text"] = (
            "Average place. Nothing special but not terrible either. "
            "Food was okay I guess."
        )
        st.rerun()
with col3:
    if st.button("↗ Positive example", key="pos_btn"):
        st.session_state["review_text"] = (
            "Best restaurant I have been to in years! The pasta was "
            "fresh and delicious. Highly recommend!"
        )
        st.rerun()

review_text = st.text_area(
    label="Review",
    value=st.session_state["review_text"],
    placeholder="Paste a customer review here — the longer the better...",
    height=160,
    key="input_box"
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([0.15, 0.7, 0.15])
with btn_col:
    analyze_btn = st.button("Analyze Sentiment →", key="analyze", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────
if analyze_btn:
    if not review_text.strip():
        st.markdown(
            '<div style="font-family:var(--mono,monospace);font-size:11px;'
            'letter-spacing:0.1em;text-transform:uppercase;color:#D4720C;'
            'padding:16px 0;border-top:1px solid rgba(212,114,12,0.2);margin-top:20px;">'
            '⚠ Please enter a review before analyzing.</div>',
            unsafe_allow_html=True
        )
    else:
        with st.spinner(""):
            try:
                probs, cleaned = predict(review_text, model, tokenizer, lem, sw)
            except Exception as e:
                st.markdown(
                    f'<div style="font-family:monospace;font-size:12px;color:#C8392B;">'
                    f'Prediction failed: {e}</div>',
                    unsafe_allow_html=True
                )
                st.stop()

        labels    = ["Negative", "Neutral", "Positive"]
        css_keys  = ["neg", "neu", "pos"]
        icons     = ["↘", "→", "↗"]
        descs     = [
            "Signals customer dissatisfaction — flag for QA and service recovery.",
            "Mixed or lukewarm sentiment — examine for specific pain points.",
            "Strong approval signal — excellent for marketing and featuring."
        ]

        idx   = int(np.argmax(probs))
        key   = css_keys[idx]
        conf  = probs[idx] * 100

        # Verdict block
        st.markdown(f"""
        <div class="result-panel">
            <div class="section-label">Analysis Result</div>
            <div class="verdict-block">
                <div>
                    <div class="verdict-label {key}">{icons[idx]} {key.upper()}</div>
                    <div class="verdict-title {key}">{labels[idx]}</div>
                    <div class="verdict-meta">
                        Classified from {len(review_text.split())} input words
                    </div>
                </div>
                <div>
                    <div class="confidence-pill {key}">{conf:.0f}%</div>
                    <div class="confidence-sub">Confidence</div>
                </div>
            </div>

            <div class="section-label">Score Distribution</div>
            <div class="dist-section">
        """, unsafe_allow_html=True)

        for i, (lbl, k, p) in enumerate(zip(labels, css_keys, probs)):
            pct = p * 100
            st.markdown(f"""
            <div class="dist-row">
                <div class="dist-label">{lbl}</div>
                <div class="dist-track">
                    <div class="dist-fill {k}" style="width:{pct:.1f}%"></div>
                </div>
                <div class="dist-pct {k}">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Insight card
        st.markdown(f"""
            <div class="section-label">Operational Insight</div>
            <div class="insight-card {key}">
                <div class="insight-card-label {key}">Recommended Action</div>
                <div class="insight-card-text">{descs[idx]}</div>
            </div>
        """, unsafe_allow_html=True)

        # Preprocessed text (collapsible)
        st.markdown(
            '<div class="section-label" style="margin-top:32px;">Pipeline Output</div>',
            unsafe_allow_html=True
        )
        with st.expander("View preprocessed tokens"):
            if cleaned:
                st.markdown(
                    f'<div class="preprocessed-box"><code>{cleaned}</code></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-family:monospace;font-size:11px;color:#888;">'
                    'No tokens remaining after preprocessing.</div>',
                    unsafe_allow_html=True
                )

        st.markdown("</div>", unsafe_allow_html=True)  # close result-panel

st.markdown("</div>", unsafe_allow_html=True)  # close main-content

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="page-footer">
    <div class="footer-meta">
        <span>Bi‑LSTM</span>
        <span class="footer-dot">·</span>
        <span>GloVe 100d</span>
        <span class="footer-dot">·</span>
        <span>45 000 reviews</span>
        <span class="footer-dot">·</span>
        <span>Seq len 200</span>
        <span class="footer-dot">·</span>
        <span>3-class</span>
    </div>
    <div class="footer-meta">
        <span>Yelp Review Full · HuggingFace</span>
    </div>
</div>
""", unsafe_allow_html=True)
