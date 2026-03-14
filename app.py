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

# ── CSS — surgical overrides only, widgets left intact ────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #F5F2ED !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 720px !important;
    padding-top: 0 !important;
    padding-bottom: 48px !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
}

/* Textarea */
textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
    color: #0D0D0D !important;
    background-color: #EDEAE4 !important;
    border: 1.5px solid transparent !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
}
textarea:focus {
    border-color: #0D0D0D !important;
    outline: none !important;
    box-shadow: none !important;
}
textarea::placeholder {
    color: rgba(13,13,13,0.38) !important;
    font-style: italic;
}
.stTextArea label { display: none !important; }

/* Analyze button */
.stButton button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #F5F2ED !important;
    background-color: #0D0D0D !important;
    border: 1.5px solid #0D0D0D !important;
    border-radius: 4px !important;
    padding: 12px 28px !important;
    height: auto !important;
    line-height: 1.4 !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: opacity 0.15s !important;
}
.stButton button:hover,
.stButton button:focus,
.stButton button:active {
    opacity: 0.72 !important;
    color: #F5F2ED !important;
    background-color: #0D0D0D !important;
    border-color: #0D0D0D !important;
    box-shadow: none !important;
}

/* Chip buttons live inside .chip-wrap */
.chip-wrap .stButton button {
    font-size: 11px !important;
    background-color: transparent !important;
    color: rgba(13,13,13,0.52) !important;
    border: 1.5px solid rgba(13,13,13,0.15) !important;
    border-radius: 100px !important;
    padding: 7px 16px !important;
    letter-spacing: 0.06em !important;
    text-transform: none !important;
}
.chip-wrap .stButton button:hover,
.chip-wrap .stButton button:focus,
.chip-wrap .stButton button:active {
    background-color: #0D0D0D !important;
    color: #F5F2ED !important;
    border-color: #0D0D0D !important;
    opacity: 1 !important;
}

/* Expander */
details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: rgba(13,13,13,0.45) !important;
    padding: 0 0 8px 0 !important;
}
details { border: none !important; background: transparent !important; }

/* Spinner */
.stSpinner p {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: rgba(13,13,13,0.45) !important;
}

.stAlert { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── File Download ─────────────────────────────────────────────────
def download_file(file_id, output_path, label):
    try:
        import gdown
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={file_id}",
                       output_path, quiet=False, fuzzy=True)
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise ValueError("Download incomplete.")
    except Exception as e:
        st.error(f"Download failed ({label}): {e}")
        st.stop()

def ensure_files():
    if not os.path.exists(WEIGHTS_PATH) or os.path.getsize(WEIGHTS_PATH) < 1000:
        with st.spinner("Downloading model weights…"):
            download_file(WEIGHTS_ID, WEIGHTS_PATH, "weights")
    if not os.path.exists(TOKENIZER_PATH) or os.path.getsize(TOKENIZER_PATH) < 100:
        with st.spinner("Downloading tokenizer…"):
            download_file(TOKENIZER_ID, TOKENIZER_PATH, "tokenizer")


# ── Model Loading ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Embedding, Bidirectional,
                                         LSTM, Dense, Dropout, SpatialDropout1D)
    from tensorflow.keras.regularizers import l2

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), name='Input')
    x   = Embedding(VOCAB_SIZE, EMBEDDING_DIM, trainable=False,
                    name='GloVe_Embedding')(inp)
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
    m   = Model(inputs=inp, outputs=out)
    m.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
    m.load_weights(WEIGHTS_PATH)
    return m

@st.cache_resource(show_spinner=False)
def load_tokenizer():
    import tensorflow as tf
    real_cls = tf.keras.preprocessing.text.Tokenizer
    for mod in ["keras.src", "keras.src.legacy",
                "keras.src.legacy.preprocessing",
                "keras.src.legacy.preprocessing.text"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    sys.modules["keras.src.legacy.preprocessing.text"].Tokenizer = real_cls
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_nlp():
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
_html    = re.compile(r'<.*?>')
_url     = re.compile(r'http\S+|www\.\S+')
_email   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
_phone   = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
_special = re.compile(r'[^a-z\s]')
_space   = re.compile(r'\s+')
SLANG = {"idk":"i do not know","tbh":"to be honest","gr8":"great",
         "lol":"laughing","omg":"oh my god","btw":"by the way",
         "imo":"in my opinion","asap":"as soon as possible"}

def clean_text(text, lem, sw):
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = _email.sub('[EMAIL]', t)
    t = _phone.sub('[PHONE]', t)
    t = _html.sub(' ', t)
    t = _url.sub('', t)
    for s, e in SLANG.items():
        t = re.sub(rf'\b{s}\b', e, t)
    t = _special.sub('', t)
    t = _space.sub(' ', t).strip()
    return ' '.join(lem.lemmatize(w) for w in t.split() if w not in sw)

def predict(raw, model, tok, lem, sw):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    cleaned = clean_text(raw, lem, sw)
    if not cleaned:
        return np.array([0.33, 0.34, 0.33]), cleaned
    seq    = tok.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post', truncating='post')
    return model.predict(padded, verbose=0)[0], cleaned


# ══════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════

# ── Top bar ───────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:20px 0;border-bottom:1px solid rgba(13,13,13,0.1);
            margin-bottom:52px;">
  <div style="font-family:'Instrument Serif',Georgia,serif;font-size:20px;
              color:#0D0D0D;letter-spacing:-0.01em;display:flex;
              align-items:center;gap:10px;">
    Sentient
    <span style="font-family:'DM Mono',monospace;font-size:9px;font-weight:500;
                 letter-spacing:0.14em;text-transform:uppercase;
                 background:rgba(13,13,13,0.07);color:rgba(13,13,13,0.45);
                 padding:3px 9px;border-radius:100px;">Beta</span>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;
              letter-spacing:0.1em;text-transform:uppercase;
              color:rgba(13,13,13,0.35);">Bi‑LSTM · Yelp</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:52px;">
  <div style="font-family:'DM Mono',monospace;font-size:10px;
              letter-spacing:0.16em;text-transform:uppercase;
              color:rgba(13,13,13,0.4);margin-bottom:18px;
              display:flex;align-items:center;gap:10px;">
    <div style="width:28px;height:1px;background:rgba(13,13,13,0.25);"></div>
    Sentiment Intelligence
  </div>
  <h1 style="font-family:'Instrument Serif',Georgia,serif;
             font-size:clamp(38px,6vw,58px);line-height:1.06;
             letter-spacing:-0.02em;color:#0D0D0D;margin:0 0 16px;
             font-weight:400;">
    What does your<br>
    <em style="color:rgba(13,13,13,0.4);">customer</em> really feel?
  </h1>
  <p style="font-size:15px;line-height:1.65;color:rgba(13,13,13,0.52);
            font-weight:300;max-width:460px;margin:0;">
    Paste any Yelp review. Our Bi-LSTM neural network reads it and 
    returns a sentiment verdict —&nbsp;
    <strong style="font-weight:500;color:#0D0D0D;">negative, neutral,</strong>
    &nbsp;or&nbsp;
    <strong style="font-weight:500;color:#0D0D0D;">positive</strong>
    &nbsp;— with confidence scores.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Bootstrap ─────────────────────────────────────────────────────
ensure_files()
with st.spinner("Initialising model…"):
    try:
        model     = load_model()
        tokenizer = load_tokenizer()
        lem, sw   = load_nlp()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ── Input ─────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:10px;
            letter-spacing:0.16em;text-transform:uppercase;
            color:rgba(13,13,13,0.4);margin-bottom:14px;
            display:flex;align-items:center;gap:10px;">
  Review Input
  <div style="flex:1;height:1px;background:rgba(13,13,13,0.1);"></div>
</div>
""", unsafe_allow_html=True)

if "rv" not in st.session_state:
    st.session_state.rv = ""

# Chip buttons
st.markdown('<div class="chip-wrap">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("↘  Negative", key="b_neg"):
        st.session_state.rv = ("The food was absolutely disgusting and the "
                               "service was incredibly rude. Never coming back.")
        st.rerun()
with c2:
    if st.button("→  Neutral", key="b_neu"):
        st.session_state.rv = ("Average place. Nothing special but not terrible "
                               "either. Food was okay I guess.")
        st.rerun()
with c3:
    if st.button("↗  Positive", key="b_pos"):
        st.session_state.rv = ("Best restaurant I have been to in years! "
                               "The pasta was fresh and delicious. Highly recommend!")
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

review_text = st.text_area(
    label="review",
    value=st.session_state.rv,
    placeholder="Paste a Yelp review here…",
    height=148,
    key="ta",
    label_visibility="collapsed"
)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    go = st.button("Analyze Sentiment →", key="go", use_container_width=True)


# ── Result ────────────────────────────────────────────────────────
if go:
    if not review_text.strip():
        st.markdown("""
        <div style="margin-top:24px;padding:14px 18px;
                    border-left:3px solid #D4720C;
                    background:rgba(212,114,12,0.06);
                    border-radius:0 6px 6px 0;
                    font-family:'DM Mono',monospace;font-size:11px;
                    letter-spacing:0.08em;text-transform:uppercase;
                    color:#D4720C;">
          ⚠ Please enter a review before analyzing.
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing…"):
            try:
                probs, cleaned = predict(review_text, model, tokenizer, lem, sw)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        idx      = int(np.argmax(probs))
        labels   = ["Negative",              "Neutral",               "Positive"]
        icons    = ["↘",                     "→",                     "↗"]
        colors   = ["#C8392B",               "#D4720C",               "#1A6B42"]
        bgs      = ["rgba(200,57,43,0.07)",  "rgba(212,114,12,0.07)", "rgba(26,107,66,0.07)"]
        borders  = ["rgba(200,57,43,0.18)",  "rgba(212,114,12,0.18)", "rgba(26,107,66,0.18)"]
        insights = [
            "Signals customer dissatisfaction. Flag for QA team and service recovery.",
            "Mixed or lukewarm sentiment. Scan for specific pain points worth addressing.",
            "Strong positive signal. Great candidate for featured placement or marketing."
        ]

        C  = colors[idx]
        BG = bgs[idx]
        BD = borders[idx]

        # Divider
        st.markdown("""
        <div style="height:1px;background:rgba(13,13,13,0.1);margin:40px 0;"></div>
        """, unsafe_allow_html=True)

        # Label
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:10px;
                    letter-spacing:0.16em;text-transform:uppercase;
                    color:rgba(13,13,13,0.4);margin-bottom:28px;
                    display:flex;align-items:center;gap:10px;">
          Analysis Result
          <div style="flex:1;height:1px;background:rgba(13,13,13,0.1);"></div>
        </div>
        """, unsafe_allow_html=True)

        # Verdict
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr auto;
                    gap:24px;align-items:start;margin-bottom:40px;">
          <div>
            <div style="font-family:'DM Mono',monospace;font-size:10px;
                        letter-spacing:0.15em;text-transform:uppercase;
                        color:{C};margin-bottom:10px;">
              {icons[idx]} {labels[idx].upper()}
            </div>
            <div style="font-family:'Instrument Serif',Georgia,serif;
                        font-size:clamp(32px,5vw,50px);line-height:1.0;
                        letter-spacing:-0.02em;color:{C};font-weight:400;">
              {labels[idx]}
            </div>
            <div style="font-size:13px;color:rgba(13,13,13,0.4);
                        margin-top:8px;font-weight:300;">
              {len(review_text.split())} words analyzed
            </div>
          </div>
          <div style="text-align:center;padding-top:4px;">
            <div style="font-family:'DM Mono',monospace;font-size:28px;
                        font-weight:500;letter-spacing:-0.02em;color:{C};
                        background:{BG};border:1.5px solid {BD};
                        border-radius:12px;padding:14px 20px;min-width:90px;">
              {probs[idx]*100:.0f}%
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:9px;
                        letter-spacing:0.12em;text-transform:uppercase;
                        color:rgba(13,13,13,0.3);margin-top:6px;">
              Confidence
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Score bars
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:10px;
                    letter-spacing:0.16em;text-transform:uppercase;
                    color:rgba(13,13,13,0.4);margin-bottom:16px;
                    display:flex;align-items:center;gap:10px;">
          Score Distribution
          <div style="flex:1;height:1px;background:rgba(13,13,13,0.1);"></div>
        </div>
        """, unsafe_allow_html=True)

        for i, (lbl, p, col) in enumerate(zip(labels, probs, colors)):
            is_w   = (i == idx)
            weight = "600" if is_w else "400"
            op     = "1"   if is_w else "0.45"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:78px 1fr 46px;
                        align-items:center;gap:14px;margin-bottom:12px;
                        opacity:{op};">
              <div style="font-family:'DM Mono',monospace;font-size:10px;
                          letter-spacing:0.08em;text-transform:uppercase;
                          color:rgba(13,13,13,0.55);font-weight:{weight};">
                {lbl}
              </div>
              <div style="height:3px;background:rgba(13,13,13,0.1);
                          border-radius:100px;overflow:hidden;">
                <div style="width:{p*100:.1f}%;height:100%;
                            background:{col};border-radius:100px;"></div>
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:12px;
                          font-weight:{weight};color:{col};text-align:right;">
                {p*100:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Insight
        st.markdown(f"""
        <div style="height:28px;"></div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;
                    letter-spacing:0.16em;text-transform:uppercase;
                    color:rgba(13,13,13,0.4);margin-bottom:14px;
                    display:flex;align-items:center;gap:10px;">
          Operational Insight
          <div style="flex:1;height:1px;background:rgba(13,13,13,0.1);"></div>
        </div>
        <div style="padding:20px 24px;background:{BG};border:1.5px solid {BD};
                    border-radius:10px;margin-bottom:28px;">
          <div style="font-family:'DM Mono',monospace;font-size:9px;
                      letter-spacing:0.16em;text-transform:uppercase;
                      color:{C};margin-bottom:8px;">Recommended Action</div>
          <div style="font-size:14px;line-height:1.65;color:#0D0D0D;
                      font-weight:300;">{insights[idx]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessed tokens
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:10px;
                    letter-spacing:0.16em;text-transform:uppercase;
                    color:rgba(13,13,13,0.4);margin-bottom:12px;
                    display:flex;align-items:center;gap:10px;">
          Pipeline Output
          <div style="flex:1;height:1px;background:rgba(13,13,13,0.1);"></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View preprocessed tokens"):
            if cleaned:
                st.markdown(f"""
                <div style="background:#EDEAE4;border-radius:8px;
                            padding:16px 20px;margin-top:4px;">
                  <code style="font-family:'DM Mono',monospace;font-size:12px;
                               line-height:1.8;color:rgba(13,13,13,0.55);
                               background:transparent;">{cleaned}</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-family:'DM Mono',monospace;font-size:11px;
                            color:rgba(13,13,13,0.35);padding:12px 0;">
                  No tokens remaining after preprocessing.
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:72px;padding-top:24px;
            border-top:1px solid rgba(13,13,13,0.1);
            display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:8px;">
  <div style="font-family:'DM Mono',monospace;font-size:10px;
              letter-spacing:0.08em;color:rgba(13,13,13,0.3);
              display:flex;gap:14px;flex-wrap:wrap;">
    <span>Bi‑LSTM</span><span style="opacity:.3">·</span>
    <span>GloVe 100d</span><span style="opacity:.3">·</span>
    <span>45 000 reviews</span><span style="opacity:.3">·</span>
    <span>Seq 200</span><span style="opacity:.3">·</span>
    <span>3-class</span>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;
              letter-spacing:0.08em;color:rgba(13,13,13,0.25);">
    Yelp Review Full · HuggingFace
  </div>
</div>
""", unsafe_allow_html=True)
