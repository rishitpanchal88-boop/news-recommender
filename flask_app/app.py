"""
News Recommendation System - Flask Backend

Run this with:  python app.py
Then open:       http://127.0.0.1:5000 in your browser
"""

import os
import re
import json
import sys
import time
import string
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse
from flask import Flask, render_template, redirect, url_for, request, flash, abort, jsonify
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
# Every data file is resolved against this file's location, never against the
# current working directory. A bare `pd.read_csv("x.csv")` only works if you
# happen to be sitting in exactly one folder, so the app dies with a
# FileNotFoundError the moment someone launches it from anywhere else -- which
# is precisely what happens when the project folder is copied or moved.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Overridable so the (large) data files can be kept outside the source tree,
# e.g. on a network share, without editing any code.
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)

# Guarantee the sibling modules (models, news_api, premium) are importable no
# matter how the app was started. Python already does this automatically for
# `python app.py` and `python flask_app/app.py`, but NOT for `python -m`, an
# interactive session, or any harness that imports app.py by file path -- those
# would otherwise die with a bare "No module named 'models'".
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
# Explicit path: a bare load_dotenv() searches from the current working
# directory upward, so running the app from the project root would silently miss
# flask_app/.env and drop every API key without any error being raised.
load_dotenv(os.path.join(BASE_DIR, ".env"))

from models import db, User, Interaction, RevenueLog
import news_api
import premium as premium_engine

# ----------------------------------------------------------------------
# APP SETUP
# ----------------------------------------------------------------------
def _resolve_secret_key():
    """Return the key used to sign session cookies.

    There is deliberately no hardcoded fallback. A default secret committed in
    source means anyone who reads the repo can forge a session cookie and
    impersonate any account. In production we generate a secure random one if
    not explicitly provided, ensuring the service starts reliably.
    """
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    import secrets
    ephemeral_key = secrets.token_hex(32)
    if os.environ.get("FLASK_ENV", "development") == "production":
        print("NOTICE: Production mode active without explicit SECRET_KEY - using generated secure key.")
    else:
        print("WARNING: SECRET_KEY is not set - generated an ephemeral one. "
              "Sessions will not survive a restart. Add SECRET_KEY to .env.")
    return ephemeral_key


app = Flask(__name__)
app.config["SECRET_KEY"] = _resolve_secret_key()

# Support reverse proxies (Render, Railway, Fly.io, Heroku, Nginx, Cloudflare)
try:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
except Exception:
    pass

# SQLite by default. Point DATABASE_URL at Postgres for a real deployment --
# Render's free tier gives an ephemeral disk, so a SQLite file there is wiped
# on every redeploy (losing users, interactions and the live_article archive).
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///db.sqlite3"
)
# Heroku/Render hand out postgres:// URLs; SQLAlchemy 2.x wants postgresql://
if app.config["SQLALCHEMY_DATABASE_URI"].startswith("postgres://"):
    app.config["SQLALCHEMY_DATABASE_URI"] = app.config["SQLALCHEMY_DATABASE_URI"].replace(
        "postgres://", "postgresql://", 1
    )
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


def _ensure_schema_columns():
    """Add new columns to user and interaction tables if they don't already exist.
    SQLite doesn't support ALTER TABLE ADD COLUMN IF NOT EXISTS, so we inspect and add individually."""
    try:
        import sqlite3
        db_path = os.path.join(app.instance_path, "db.sqlite3")
        if not os.path.exists(db_path):
            return
        conn = sqlite3.connect(db_path)
        
        # User table columns
        cursor = conn.execute("PRAGMA table_info(user)")
        existing_user_cols = {row[1] for row in cursor.fetchall()}
        for col, typedef in [
            ("country", "VARCHAR(60) DEFAULT ''"),
            ("state_region", "VARCHAR(80) DEFAULT ''"),
            ("city", "VARCHAR(80) DEFAULT ''"),
            ("preferred_language", "VARCHAR(10) DEFAULT 'en'"),
        ]:
            if col not in existing_user_cols:
                conn.execute(f"ALTER TABLE user ADD COLUMN {col} {typedef}")

        # Interaction table columns
        try:
            cursor = conn.execute("PRAGMA table_info(interaction)")
            existing_int_cols = {row[1] for row in cursor.fetchall()}
            for col, typedef in [
                ("article_key", "VARCHAR(256) DEFAULT ''"),
                ("title", "VARCHAR(500) DEFAULT ''"),
                ("source_name", "VARCHAR(200) DEFAULT ''"),
                ("url_to_image", "VARCHAR(1000) DEFAULT ''"),
                ("url", "VARCHAR(1000) DEFAULT ''"),
                ("category", "VARCHAR(100) DEFAULT ''"),
            ]:
                if col not in existing_int_cols:
                    conn.execute(f"ALTER TABLE interaction ADD COLUMN {col} {typedef}")
        except Exception:
            pass

        conn.commit()
        conn.close()
    except Exception:
        pass


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ----------------------------------------------------------------------
# LOAD DATA ONCE AT STARTUP (MEMORY-OPTIMIZED & RESILIENT)
# ----------------------------------------------------------------------
def _find_data_file(filename):
    """Search for data file in DATA_DIR, BASE_DIR, and parent workspace directory.

    A gzip-compressed copy (<filename>.gz) is accepted as a fallback and
    preferred last: pandas reads .csv.gz transparently, so the large
    artifacts can live in git as .gz (GitHub caps files at 100MB; the
    raw CSV is ~516MB). The plain file always wins when both exist, so
    a local checkout with the notebook outputs behaves identically.
    """
    candidates = []
    for base in (DATA_DIR, BASE_DIR, os.path.dirname(BASE_DIR)):
        candidates.append(os.path.join(base, filename))
        candidates.append(os.path.join(base, filename + ".gz"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _build_fallback_dataset():
    """Construct a clean, self-contained fallback dataset with sample news
    articles across major categories if large CSV files are not present."""
    print("  [Data] Initializing lightweight starter dataset...")
    sample_data = [
        {
            "article_id": 1001,
            "title": "Global Climate Summit Reaches Landmark Renewable Energy Pact",
            "description": "World leaders agree on accelerated transition goals to triple global renewable energy capacity by 2030 with dedicated financing for emerging markets.",
            "source_name": "Reuters",
            "category": "Science",
            "author": "Climate Desk",
            "url": "https://www.reuters.com",
            "url_to_image": "https://images.unsplash.com/photo-1497435334941-8c899ee9e8e9?w=800&q=80",
            "published_at": "2026-10-04",
            "cleaned_text": "global climate summit reaches landmark renewable energy pact world leader agree accelerated transition goal triple renewable energy capacity",
            "credibility_label": "Real",
            "fake_news_prediction": "Real",
        },
        {
            "article_id": 1002,
            "title": "Breakthrough in Quantum Computing Achieves Practical Error Correction",
            "description": "Researchers demonstrate fault-tolerant logical qubits operating continuously, paving the way for commercial cryptographic and molecular simulation algorithms.",
            "source_name": "MIT Technology Review",
            "category": "Technology",
            "author": "Tech Research Team",
            "url": "https://technologyreview.com",
            "url_to_image": "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800&q=80",
            "published_at": "2026-10-04",
            "cleaned_text": "breakthrough quantum computing achieves practical error correction researcher demonstrate faulttolerant logical qubit commercial cryptographic",
            "credibility_label": "Real",
            "fake_news_prediction": "Real",
        },
        {
            "article_id": 1003,
            "title": "Central Banks Announce Coordinated Policy Framework on Digital Currencies",
            "description": "Major economic authorities outline interoperability standards for sovereign digital currencies to reduce cross-border settlement friction.",
            "source_name": "Bloomberg",
            "category": "Finance",
            "author": "Financial Markets",
            "url": "https://bloomberg.com",
            "url_to_image": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=800&q=80",
            "published_at": "2026-10-03",
            "cleaned_text": "central bank announce coordinated policy framework digital currency economic authority outline interoperability standard sovereign settlement",
            "credibility_label": "Real",
            "fake_news_prediction": "Real",
        },
        {
            "article_id": 1004,
            "title": "Championship Finals Deliver Thrilling Double-Overtime Victory",
            "description": "Underdog squad secures international title following a spectacular defensive display in the closing seconds of sudden death overtime.",
            "source_name": "BBC Sport",
            "category": "Sports",
            "author": "Sports Desk",
            "url": "https://bbc.com/sport",
            "url_to_image": "https://images.unsplash.com/photo-1461896836934-ffe607ba8211?w=800&q=80",
            "published_at": "2026-10-04",
            "cleaned_text": "championship final deliver thrilling doubleovertime victory underdog squad secure international title spectacular defensive display",
            "credibility_label": "Real",
            "fake_news_prediction": "Real",
        },
        {
            "article_id": 1005,
            "title": "Clinical Trials of Universal Vaccine Show Broad Immune Response",
            "description": "Phase 3 clinical data demonstrates sustained multi-strain neutralization without significant adverse effects across diverse demographic cohorts.",
            "source_name": "The Lancet",
            "category": "Health",
            "author": "Medical Science Desk",
            "url": "https://thelancet.com",
            "url_to_image": "https://images.unsplash.com/photo-1584515979956-d9f6e5d09982?w=800&q=80",
            "published_at": "2026-10-02",
            "cleaned_text": "clinical trial universal vaccine show broad immune response phase clinical data demonstrates sustained neutralization adverse effect cohort",
            "credibility_label": "Real",
            "fake_news_prediction": "Real",
        },
        {
            "article_id": 1006,
            "title": "International Space Agency Unveils Lunar Base Construction Blueprints",
            "description": "Collaborative mission architecture outlines robotic 3D-printing habitats leveraging lunar regolith for sustainable long-duration exploration.",
            "source_name": "Associated Press",
            "category": "Science",
            "author": "Aerospace Division",
            "url": "https://apnews.com",
            "url_to_image": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=800&q=80",
            "published_at": "2026-10-01",
            "cleaned_text": "international space agency unveils lunar base construction blueprint collaborative mission architecture robotic habitat sustainable exploration",
            "credibility_label": "Real",
            "fake_news_prediction": "Real",
        },
    ]
    df = pd.DataFrame(sample_data)
    df["category"] = df["category"].fillna("")
    df = df.set_index("article_id", drop=False)
    return df


print("Loading article data...")
_csv_path = _find_data_file("cleaned_data_with_fake_news.csv")

if _csv_path:
    try:
        # Optimize memory usage: load only needed columns and downcast types
        _cols_to_use = [
            "article_id", "source_name", "author", "title", "description",
            "url", "url_to_image", "published_at", "category", "cleaned_text",
            "credibility_label", "fake_news_prediction"
        ]
        # Inspect available columns first
        _preview = pd.read_csv(_csv_path, nrows=1)
        _actual_cols = [c for c in _cols_to_use if c in _preview.columns]
        
        ARTICLES_DF = pd.read_csv(_csv_path, usecols=_actual_cols)
        # Fill missing text so templates never render "nan", and store
        # the low-cardinality label columns as categorical -- as plain
        # object columns they triple this DataFrame's RAM footprint,
        # which matters on 512MB-1GB cloud instances.
        for _c in ("author", "title", "description", "url",
                   "url_to_image", "published_at"):
            if _c in _actual_cols:
                ARTICLES_DF[_c] = ARTICLES_DF[_c].fillna("")
        for _c in ("source_name", "category", "credibility_label",
                   "fake_news_prediction"):
            if _c in _actual_cols:
                ARTICLES_DF[_c] = ARTICLES_DF[_c].fillna("").astype("category")
        ARTICLES_DF = ARTICLES_DF.set_index("article_id", drop=False)
        print(f"  {len(ARTICLES_DF)} articles loaded (memory optimized)")
    except Exception as e:
        print(f"  Warning: Failed to load CSV ({e}); using fallback starter dataset.")
        ARTICLES_DF = _build_fallback_dataset()
else:
    print("  Notice: Full dataset CSV not found; running with starter dataset & live news aggregation.")
    ARTICLES_DF = _build_fallback_dataset()

ALL_ARTICLE_IDS = ARTICLES_DF["article_id"].tolist()

print("Loading similarity table...")
_sim_path = _find_data_file("similarity_table.csv")
SIMILARITY_LOOKUP = {}

if _sim_path:
    try:
        t_sim = time.time()
        SIMILARITY_DF = pd.read_csv(_sim_path)
        for a, s, r, sc in SIMILARITY_DF.itertuples(index=False):
            SIMILARITY_LOOKUP.setdefault(a, []).append((s, sc))
        print(f"  {len(SIMILARITY_LOOKUP)} articles loaded into fast similarity lookup in {round(time.time() - t_sim, 2)}s")
    except Exception as e:
        print(f"  Warning: Could not load similarity table ({e}); building in-memory lookup.")
else:
    print("  Building in-memory similarity lookup...")

print("Loading recommendation TF-IDF...")
_tfidf_pkl = _find_data_file("tfidf_vectorizer.pkl")
_tfidf_npz = _find_data_file("tfidf_matrix.npz")

if _tfidf_pkl and _tfidf_npz:
    try:
        with open(_tfidf_pkl, "rb") as f:
            REC_TFIDF = pickle.load(f)
        REC_TFIDF_MATRIX = scipy.sparse.load_npz(_tfidf_npz).astype(np.float32)
        REC_TFIDF_MATRIX_T = REC_TFIDF_MATRIX.T.tocsr()
        print(f"  Recommendation matrix: {REC_TFIDF_MATRIX.shape}")
    except Exception as e:
        print(f"  Warning: Failed loading TF-IDF files ({e}); generating in-memory vectorizer.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        REC_TFIDF = TfidfVectorizer(stop_words="english", max_features=5000)
        _texts = ARTICLES_DF["cleaned_text"].fillna(ARTICLES_DF["title"]).tolist()
        REC_TFIDF_MATRIX = REC_TFIDF.fit_transform(_texts).astype(np.float32)
        REC_TFIDF_MATRIX_T = REC_TFIDF_MATRIX.T.tocsr()
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    REC_TFIDF = TfidfVectorizer(stop_words="english", max_features=5000)
    _texts = ARTICLES_DF["cleaned_text"].fillna(ARTICLES_DF["title"]).tolist()
    REC_TFIDF_MATRIX = REC_TFIDF.fit_transform(_texts).astype(np.float32)
    REC_TFIDF_MATRIX_T = REC_TFIDF_MATRIX.T.tocsr()
    print(f"  Generated recommendation matrix: {REC_TFIDF_MATRIX.shape}")

# If SIMILARITY_LOOKUP is empty, populate from TF-IDF dot product
if not SIMILARITY_LOOKUP and len(ALL_ARTICLE_IDS) > 0:
    for i, aid in enumerate(ALL_ARTICLE_IDS[:500]):
        vec = REC_TFIDF_MATRIX[i]
        sims = (vec @ REC_TFIDF_MATRIX_T).toarray().flatten()
        top_idx = np.argsort(-sims)[1:6]
        SIMILARITY_LOOKUP[aid] = [(ALL_ARTICLE_IDS[idx], round(float(sims[idx]), 4)) for idx in top_idx if sims[idx] > 0]

# API keys come from the environment only -- never from a literal in source.
# Both keys were previously hardcoded here as fallbacks, which leaks them to
# anyone who reads the file. They live in .env locally and in your host's
# dashboard when deployed. Both are optional: with no NewsAPI key the feed
# falls back to Google News RSS / GDELT, and with no Gemini key the
# fake-news checker falls back to the web-coverage verdict.
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
print(f"  News API key: {'configured' if NEWS_API_KEY else 'NOT SET (using RSS/GDELT)'}")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
try:
    from google import genai
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    print("  Gemini API client ready")
except Exception:
    GEMINI_CLIENT = None
    print("  Gemini API not available, using web-coverage fallback only")

# Models tried in order for the fake-news checker. Free-tier quota is per
# model per day, so if one is rate-limited we fall through to the next.
# "lite"/flash-lite models carry the largest free daily quotas.
GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# Social / user-generated platforms -- not real news outlets. Used only to
# decide the last-resort verdict when every Gemini model is unavailable.
SHADY_SOURCES = {
    "facebook", "facebook.com", "twitter", "twitter.com", "x", "x.com",
    "reddit", "reddit.com", "instagram", "instagram.com", "youtube",
    "youtube.com", "tiktok", "tiktok.com", "telegram", "whatsapp", "threads",
}

# Verdict cache: repeated checks of the same text must not burn Gemini quota.
_VERDICT_CACHE = {}
_VERDICT_TTL = 6 * 60 * 60  # 6 hours

# Curated category menu -- these are real values from the dataset's
# `category` column. "Cinema" and "Business" are friendlier menu labels
# that link to their real underlying category value.
NAV_CATEGORIES = [
    ("Sports", "Sports"),
    ("Business", "Finance"),
    ("Cinema", "Movies"),
    ("Technology", "Technology"),
    ("Health", "Health"),
    ("Education", "Education"),
    ("Politics", "Politics"),
    ("Science", "Science"),
    ("Travel", "Travel"),
    ("Food", "Food"),
]

LANGUAGES = [
    ("en", "English"),
    ("hi", "Hindi"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("gu", "Gujarati"),
]

# Dataset category value -> NewsAPI top-headlines category.
# Only categories NewsAPI supports are mapped; the rest fall back to a
# keyword search of the news archive instead.
NEWS_CATEGORY_MAP = {
    "Sports": "sports",
    "Finance": "business",
    "Movies": "entertainment",
    "Technology": "technology",
    "Health": "health",
    "Science": "science",
    "Education": "education",
    "Politics": "politics",
    "Travel": "travel",
    "Food": "food",
}

TOP_NEWS_COUNT = 6
FEED_COUNT = 12

# Text cleaning -- same pipeline used in both notebooks, needed here to
# clean whatever a user types into the fake-news checker.
try:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()

    def clean_text(text):
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in STOP_WORDS and w.isalpha()]
        tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
        return " ".join(tokens)
except Exception as e:
    print(f"WARNING: NLTK setup failed ({e}); falling back to basic cleaning.")

    def clean_text(text):
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def get_article(article_id):
    if article_id not in ARTICLES_DF.index:
        return None
    return ARTICLES_DF.loc[article_id].to_dict()


def get_recommendations(article_id, top_n=5):
    if article_id not in SIMILARITY_LOOKUP:
        return []
    neighbors = SIMILARITY_LOOKUP[article_id][:top_n]
    results = []
    for sim_id, score in neighbors:
        article = get_article(sim_id)
        if article:
            article["similarity_score"] = score
            results.append(article)
    return results


def _live_pipeline_kwargs():
    """The shared set of arguments every live-news pipeline call needs."""
    return dict(
        clean_text_fn=clean_text,
        rec_tfidf=REC_TFIDF,
        rec_tfidf_matrix=REC_TFIDF_MATRIX,
        rec_tfidf_matrix_t=REC_TFIDF_MATRIX_T,
        all_article_ids=ALL_ARTICLE_IDS,
        get_article_fn=get_article,
    )


def get_live_articles(category="general"):
    """Ensure the live-news cache is warm for a category (defaults to general),
    then return ALL cached live articles for it.  Every route goes through this
    single entry point so the API quota is only consumed once per cache window
    per category.  The multi-source aggregator (NewsAPI + Google RSS + GDELT)
    picks up whichever sources are available, so this works even without a
    NewsAPI key."""
    news_api.get_live_news(
        api_key=NEWS_API_KEY, count=TOP_NEWS_COUNT, category=category,
        **_live_pipeline_kwargs()
    )
    return news_api.get_all_live_articles(category)


def get_cached_live_articles_only(category="general"):
    """Return ONLY already-cached live articles for a category.  Never
    triggers network fetches — used by routes that must not block."""
    return news_api.get_all_live_articles(category)


def _rehydrate_live_article(article):
    """Rebuild the derived fields a live-article card needs when it was
    restored from the durable SQLite table rather than the memory cache.

    Rows rebuilt by news_api.load_article() carry the display fields but not
    the two computed ones: _rec_vec (TF-IDF, needed for live-to-live
    "Similar Live Coverage") and similar_from_dataset (archive matches). Both
    are recomputed here from cleaned_text using the already-loaded
    recommendation vectorizer -- the same maths process_articles() does, so a
    restored card behaves identically to a cached one. Never raises.
    """
    if article is None:
        return None

    cleaned = article.get("cleaned_text") or ""

    if article.get("_rec_vec") is None and cleaned:
        try:
            article["_rec_vec"] = REC_TFIDF.transform([cleaned]).astype(np.float32)
        except Exception:
            article["_rec_vec"] = None

    if not article.get("similar_from_dataset") and article.get("_rec_vec") is not None:
        try:
            sims = (article["_rec_vec"] @ REC_TFIDF_MATRIX_T).toarray().flatten()
            for i in np.argsort(-sims)[:5]:
                if sims[i] <= 0:
                    continue
                match = get_article(ALL_ARTICLE_IDS[i])
                if match:
                    match["similarity_score"] = round(float(sims[i]), 4)
                    article["similar_from_dataset"].append(match)
        except Exception:
            pass

    return article


def resolve_live_article(live_id):
    """Resolve a /live-article/<live_id> id to a renderable article.

    Checks the volatile caches first, then falls back to the durable
    live_article table and rehydrates the computed fields. Returns None only
    if the id was genuinely never persisted -- in which case there is nothing
    to show and the caller should redirect.

    Deliberately makes NO network calls: a warm-up fetch cannot resurrect an
    evicted id anyway, it just burns API quota on every dead link.
    """
    article = news_api.get_cached_live_article(live_id)
    if article is None:
        return None
    # A memory hit already has _rec_vec / similar_from_dataset, so this is a
    # no-op for it; a DB hit gets them rebuilt here.
    return _rehydrate_live_article(article)


_FULL_ARTICLE_CACHE = {}
_FULL_ARTICLE_TTL = 10 * 60  # seconds


def _is_readable_text(text):
    """Reject pages whose 'extracted' text is actually compressed/obfuscated
    bytes (some sites serve that to bots). A real article is mostly ASCII
    letters and contains a reasonable number of English words."""
    if not text or len(text) < 150:
        return False
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if letters / len(text) < 0.60:
        return False
    if len(re.findall(r"[A-Za-z]{2,}", text)) < 30:
        return False
    return True


def _extract_from_html(html, url):
    """Try several article extractors on raw HTML. Returns clean article text
    or "" if nothing readable was extracted. Never raises."""
    import trafilatura

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        include_links=False,
        output_format="txt",
    )
    text = (text or "").strip()
    if _is_readable_text(text):
        return text

    try:
        from bs4 import BeautifulSoup
        from readability import Document

        doc = Document(html, url=url)
        soup = BeautifulSoup(doc.summary(html_partial=True), "lxml")
        text2 = soup.get_text(separator="\n", strip=True)
        if _is_readable_text(text2):
            return text2
    except Exception:
        pass
    return ""


def _wayback_snapshot(url):
    """Find the closest archived snapshot of a URL via the Wayback Machine.
    Returns the snapshot URL, or "" if none exists. Never raises."""
    try:
        import requests

        r = requests.get(
            "https://archive.org/wayback/available",
            params={"url": url},
            timeout=20,
        )
        snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("status") == "200" and snap.get("url"):
            return snap["url"]
    except Exception:
        pass
    return ""


def _strip_junk_prefix(text):
    """Drop leading lines that are just site navigation/header leftovers."""
    if not text:
        return text
    lines = text.split("\n")
    junk = ("please try another search", "skip to content", "skip navigation",
            "sign in", "subscribe", "newsletter", "menu")
    while lines:
        first = lines[0].strip()
        if first.lower() in junk or any(first.lower().startswith(j) for j in junk):
            lines.pop(0)
        else:
            break
    return "\n".join(lines).strip()


def fetch_full_article(url):
    """Fetch and extract the main article text from an external URL.

    Strategy (with a short in-memory cache):
      1. Fetch the live page (trafilatura) and extract readable text.
      2. Fall back to the Wayback Machine's closest snapshot of the page
         (for sites that block bots, like Investing.com).

    Returns the extracted article text (str), or "" if nothing readable was
    found. Never raises."""
    if not url:
        return ""
    try:
        cached = _FULL_ARTICLE_CACHE.get(url)
        if cached and (time.time() - cached[0]) < _FULL_ARTICLE_TTL:
            return cached[1]

        text = ""
        try:
            import trafilatura

            html = trafilatura.fetch_url(url)
            if html:
                text = _extract_from_html(html, url)
        except Exception:
            pass

        if not text:
            snapshot = _wayback_snapshot(url)
            if snapshot:
                try:
                    import trafilatura

                    html = trafilatura.fetch_url(snapshot)
                    if html:
                        text = _extract_from_html(html, snapshot)
                except Exception:
                    pass

        if not text:
            return ""
        _FULL_ARTICLE_CACHE[url] = (time.time(), text)
        if len(_FULL_ARTICLE_CACHE) > 300:
            _FULL_ARTICLE_CACHE.clear()
        return _strip_junk_prefix(text)
    except Exception:
        return ""


def get_live_recommendations(article_id, top_n=3):
    """Find cached live (NewsAPI) articles most similar to a dataset
    article, using the same recommendation TF-IDF space. Cheap: the live
    vectors were already computed during process_articles() and are cached
    in-memory, so this is only a handful of dot products."""
    article = get_article(article_id)
    if article is None:
        return []
    cleaned = article.get("cleaned_text") or ""
    if not cleaned:
        return []
    try:
        query_vec = REC_TFIDF.transform([cleaned]).astype(np.float32)
    except Exception:
        return []

    scored = []
    for live in get_live_articles():
        live_vec = live.get("_rec_vec")
        if live_vec is None:
            continue
        try:
            score = float((query_vec @ live_vec.T).toarray()[0, 0])
        except Exception:
            continue
        if score <= 0:
            continue
        scored.append((score, live))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, live in scored[:top_n]:
        live["similarity_score"] = round(score, 4)
        results.append(live)
    return results


def filter_live_articles(keywords):
    """Return cached live articles whose title/description/source contains
    ANY of the given keywords. Uses OR matching so even partial queries
    return results. Searches across all cached categories."""
    words = [w.lower() for w in re.findall(r'[A-Za-z0-9]+', keywords or '') if len(w) >= 2]
    if not words:
        return []
    matches = []
    for cat in ["general", "technology", "business", "sports", "entertainment",
                "health", "science", "education", "politics"]:
        for a in get_cached_live_articles_only(category=cat):
            haystack = "{} {} {}".format(
                a.get("title", ""), a.get("description", ""), a.get("source_name", "")
            ).lower()
            if any(w in haystack for w in words):
                matches.append(a)
    seen = set()
    unique = []
    for a in matches:
        lid = a.get("live_id")
        if lid not in seen:
            seen.add(lid)
            unique.append(a)
    return unique


def search_web_news(query, max_results=12):
    """Search the wider news archive via the multi-source aggregator
    (NewsAPI + Google RSS + GDELT). Returns live-card dicts for display.

    Every hit is pushed through news_api.process_articles() -- the same
    pipeline the category feeds use. That is what registers the article in
    news_api._live_article_index, which is the ONLY place the
    /live-article/<live_id> route looks. Building the card by hand here
    left search results un-indexed, so clicking one bounced back to the
    home page with "This live article is no longer available."

    Going through process_articles() also gives each search hit the extras
    its detail page expects: _rec_vec for "Similar Live Coverage" chaining
    and similar_from_dataset for dataset matches.

    Returns [] on failure -- never raises."""
    if not query.strip():
        return []
    try:
        results = news_api.search_everything(
            NEWS_API_KEY, query, page_size=max_results * 2, raw=False
        )
        # Re-shape the simplified search dicts into the legacy shape
        # process_articles() expects, then let it normalise + index them.
        legacy = [
            {
                "title": r.get("title", ""),
                "description": r.get("description", ""),
                "content": "",
                "url": r.get("link", ""),
                "urlToImage": r.get("image", ""),
                "publishedAt": r.get("date", ""),
                "author": "",
                "source": {"name": r.get("source") or "Unknown"},
            }
            for r in results
        ]
        processed = news_api.process_articles(legacy, **_live_pipeline_kwargs())
        for art in processed:
            art["category"] = "Web Search"
        return processed[:max_results]
    except Exception:
        return []


def find_related_articles(raw_text, top_n=5):
    """For arbitrary user-typed text (the fake-news checker), find the
    most similar real articles by vectorizing with the SAME TF-IDF
    vectorizer used for the recommendation engine, then comparing
    against every article's vector directly.
    """
    cleaned = clean_text(raw_text)
    if not cleaned:
        return []
    query_vec = REC_TFIDF.transform([cleaned]).astype(np.float32)
    sims = (query_vec @ REC_TFIDF_MATRIX_T).toarray().flatten()
    top_idx = np.argsort(-sims)[:top_n]
    results = []
    for i in top_idx:
        if sims[i] <= 0:
            continue
        aid = ALL_ARTICLE_IDS[i]
        article = get_article(aid)
        if article:
            article["similarity_score"] = float(sims[i])
            results.append(article)
    return results


def build_evidence_text(web_results):
    """Turn gathered web coverage into a compact evidence string for Gemini,
    flagging coverage from verified/trusted sources."""
    lines = []
    for n in web_results[:6]:
        source = (n.get("source") or "").strip()
        title = n.get("title") or ""
        verified = " [verified source]" if source.lower() in news_api.TRUSTED_SOURCES else ""
        lines.append(f"- {source}{verified}: {title}")
    return "\n".join(lines)


def _extract_retry_delay(msg):
    """Pull the suggested retry seconds out of a Gemini rate-limit error."""
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s", msg, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def _call_gemini(raw_text, context):
    """Ask Gemini to return a definitive verdict on the text, cross-checked
    against any gathered web coverage. Gemini is FORCED to answer either Real
    or Fake -- no "Uncertain" is allowed.

    Tries the GEMINI_MODELS chain so one model's daily free-tier quota doesn't
    take the checker down, retrying once on transient rate-limit errors.
    Returns a parsed {label, confidence, analysis} dict. Raises if every
    model is unavailable."""
    prompt = (
        "You are a professional fact-checking AI assistant. Analyze the input below. "
        "It may be a news article, a headline, a claim, a rumor, or any descriptive "
        "text about something that happened.\n\n"
        "You MUST give a definitive verdict: the input is either REAL news or FAKE "
        "news. Do NOT answer 'Uncertain' and do not hedge. If you cannot fully verify "
        "it, make your best judgment based on plausibility, language quality, "
        "sensationalism, named entities, consistency, and any coverage below -- but "
        "always pick exactly one label.\n\n"
        "Base your judgment on:\n"
        "(1) the claims, language quality, sensationalism and logical consistency "
        "of the text itself;\n"
        "(2) any recent web coverage given below. A story widely reported by credible "
        "outlets leans REAL; a sensational claim with no credible reporting leans FAKE.\n\n"
    )
    prompt += f"Input text:\n\"\"\"\n{raw_text[:3000]}\n\"\"\"\n\n"
    if context:
        prompt += (
            "Recent web coverage found for this story (use as additional evidence "
            "to verify or contradict the claims, not as ground truth):\n"
            f"{context[:2500]}\n\n"
        )
    prompt += (
        "Respond ONLY with this JSON (no markdown, no extra text):\n"
        '{"label":"Real" or "Fake",'
        '"confidence":number 0.0-1.0 representing how sure you are of your label,'
        '"description":"1-2 sentence explanation of your verdict and the evidence you relied on"}'
    )

    last_error = None
    for idx, model in enumerate(GEMINI_MODELS):
        retry_attempts = (0, 1) if idx == len(GEMINI_MODELS) - 1 else (0,)
        for attempt in retry_attempts:
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                raw = response.text.strip()
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
                data = json.loads(raw.strip())

                lbl = str(data.get("label", "")).capitalize()
                if lbl not in ("Real", "Fake"):
                    lbl = "Uncertain"  # last-resort marker; _finalize coerces it
                try:
                    conf = float(data.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5

                return {
                    "label": lbl,
                    "confidence": max(0.0, min(1.0, conf)),
                    "analysis": str(data.get("description") or data.get("analysis") or ""),
                }
            except Exception as e:
                last_error = e
                retry_delay = _extract_retry_delay(str(e))
                if attempt == 0 and retry_delay:
                    time.sleep(min(retry_delay, 15))
                    continue
                break  # give up on this model, try the next one

    raise last_error if last_error else RuntimeError("No Gemini model available")


def _finalize(gemini, web_results):
    """Turn the Gemini verdict into the final answer. Always ends in a
    definitive Real/Fake label (never leaves the user with 'Uncertain')."""
    label = gemini["label"]
    confidence = gemini["confidence"]
    analysis = gemini["analysis"] or ""

    trusted_count = sum(
        1 for w in web_results
        if (w.get("source") or "").lower().strip() in news_api.TRUSTED_SOURCES
    )
    if trusted_count:
        note = (
            f" This claim is being reported by {trusted_count} verified news "
            "source(s) on the web."
        )
    elif web_results:
        note = (
            " Related web coverage exists; the verdict weighs whether credible "
            "outlets are reporting it."
        )
    else:
        note = (
            " No related web coverage was found, which can mean the story is "
            "very new, obscure, or fabricated."
        )

    # Gemini refused to commit (shouldn't happen given the forced prompt) ->
    # coerce to a definitive label so the checker always gives an answer.
    if label == "Uncertain":
        label = "Fake"
        confidence = min(confidence, 0.5)
        analysis = (analysis or "") + (
            " Gemini could not separate the signal, so this claim is treated as "
            "unverified/fake rather than leaving the result unanswered."
        )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "analysis": (analysis + note).strip(),
        "classifier": "AI Cross-Validation",
    }


def gather_web_evidence(raw_text, max_results=5):
    """Collect current web coverage about the given text from Google News RSS
    and the NewsAPI archive, deduplicated by URL. Never raises."""
    results = fetch_related_news(raw_text, max_results=max_results)
    words = re.findall(r"[A-Za-z]{4,}", raw_text)[:6]
    if words and NEWS_API_KEY:
        try:
            api_news = news_api.search_everything(
                NEWS_API_KEY, " ".join(words), page_size=max_results
            )
            seen = {n["link"] for n in results if n["link"]}
            for n in api_news:
                if n["link"] and n["link"] not in seen:
                    results.append(n)
        except Exception:
            pass
    return results


def _coverage_fallback(web_results):
    """Deterministic verdict from web coverage alone -- used only when every
    Gemini model is unavailable (e.g. daily free-tier quota exhausted).
    Never returns 'Uncertain'."""
    trusted = []
    credible = []
    for w in web_results:
        src = (w.get("source") or "").lower().strip()
        if src in news_api.TRUSTED_SOURCES:
            trusted.append(w)
        elif src not in SHADY_SOURCES:
            credible.append(w)

    if trusted:
        return "Real", (
            f"AI cross-check was temporarily unavailable, but {len(trusted)} "
            "verified news source(s) are reporting this story, so it is treated "
            "as real."
        )
    if credible:
        return "Real", (
            "AI cross-check was temporarily unavailable, but real news outlets "
            "are reporting this story, so it is treated as real."
        )
    return "Fake", (
        "AI cross-check was temporarily unavailable and no credible news outlet "
        "was found reporting this claim, so it is treated as fake/unverified."
    )


def classify_fake_news(raw_text, web_results=None):
    """AI-powered classifier.

    Works on ANY descriptive input -- a news article, a headline, a claim, a
    rumor, or a short description. Web coverage is gathered (Google News RSS +
    NewsAPI archive) so the verdict can be checked against real, current
    reporting, then Gemini produces the final Real/Fake verdict with a short
    explanation.

    Resilience:
      - tries several Gemini models (free-tier quota is per model per day)
      - caches identical checks for 6h so repeat inputs don't burn quota
      - if every model is down, falls back to a web-coverage verdict so the
        checker still always returns Real or Fake.

    Returns a dict:
        label       – "Real" / "Fake" / "Uncertain"
        confidence  – 0.0–1.0  (how sure the engine is of *label*)
        analysis    – short explanation string
        classifier  – "AI Cross-Validation"
    """
    text = (raw_text or "").strip()
    cache_key = " ".join(text.lower().split())

    if len(text) < 3:
        return {
            "label": "Uncertain",
            "confidence": 0.0,
            "analysis": (
                "The input is too short to analyze. Please paste a news headline, "
                "an article, or a description of the claim."
            ),
            "classifier": "AI Cross-Validation",
        }

    cached = _VERDICT_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _VERDICT_TTL:
        return cached[1]

    # --- Gather web coverage (what's actually being reported) ---
    if web_results is None:
        web_results = gather_web_evidence(text)
    context = build_evidence_text(web_results)

    # --- Gemini produces the verdict (primary engine) ---
    if GEMINI_CLIENT:
        try:
            gemini = _call_gemini(text, context)
            if gemini:
                result = _finalize(gemini, web_results)
                _VERDICT_CACHE[cache_key] = (time.time(), result)
                return result
        except Exception:
            pass  # every Gemini model unavailable -> coverage fallback

    # --- Fallback: verdict from web coverage (never 'Uncertain') ---
    label, note = _coverage_fallback(web_results)
    result = {
        "label": label,
        "confidence": 0.65 if label == "Real" else 0.45,
        "analysis": note,
        "classifier": "AI Cross-Validation",
    }
    _VERDICT_CACHE[cache_key] = (time.time(), result)
    return result


def fetch_related_news(raw_text, max_results=5):
    """Fetch related news from Google News RSS (no API key needed).

    Returns a list of dicts with title, link, date, source.
    Never raises -- returns [] on any failure.
    """
    try:
        from urllib.request import urlopen, Request
        from urllib.parse import quote_plus
        import xml.etree.ElementTree as ET

        words = re.findall(r"[A-Za-z]{3,}", raw_text)
        if not words:
            return []
        query = " ".join(words[:10])

        url = (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query)}&hl=en&gl=US&ceid=US:en"
        )
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=6) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        items = root.findall(".//item")

        results = []
        for item in items[:max_results]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            source_el = item.find("source")
            source = source_el.text if source_el is not None and source_el.text else ""
            results.append({
                "title": title,
                "link": link,
                "date": pub_date,
                "source": source,
            })
        return results
    except Exception:
        return []


def translate_text(text, target_lang):
    """Real translation via deep-translator. Needs internet access --
    falls back to the original text with a note if the request fails
    (e.g. no internet connection).
    """
    if not text or target_lang == "en":
        return text, True
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target=target_lang).translate(text[:4900])
        return translated, True
    except Exception:
        return text, False


# ----------------------------------------------------------------------
# CONTEXT PROCESSOR -- makes these available in every template automatically
# ----------------------------------------------------------------------
@app.context_processor
def inject_globals():
    user_bookmark_keys = set()
    user_bookmark_count = 0
    if current_user.is_authenticated:
        try:
            saved = Interaction.query.filter_by(user_id=current_user.id, action="bookmark").all()
            for s in saved:
                if s.article_key:
                    user_bookmark_keys.add(s.article_key)
                if s.article_id:
                    user_bookmark_keys.add(f"ds:{s.article_id}")
                    user_bookmark_keys.add(str(s.article_id))
            user_bookmark_count = len(saved)
        except Exception:
            pass
    return {
        "today": datetime.now().strftime("%A, %B %d, %Y"),
        "nav_categories": NAV_CATEGORIES,
        "languages": LANGUAGES,
        "user_bookmark_keys": user_bookmark_keys,
        "user_bookmark_count": user_bookmark_count,
    }


# ----------------------------------------------------------------------
# ROUTES: HEALTH & MONITORING
# ----------------------------------------------------------------------
@app.route("/health")
@app.route("/api/health")
def health_check():
    """Health check endpoint for cloud platforms (Render, Railway, Fly.io, K8s)."""
    return jsonify({
        "status": "healthy",
        "app": "news-recommender",
        "timestamp": datetime.utcnow().isoformat(),
        "articles_loaded": len(ARTICLES_DF),
        "live_cache_size": len(news_api._live_article_index),
        "database": "connected",
    }), 200


# ----------------------------------------------------------------------
# ROUTES: HOME FEED
# ----------------------------------------------------------------------
@app.route("/")
def home():
    """Top News: live headlines from NewsAPI.org, processed through the
    recommendation pipeline.  Below it: a shuffled sample from the
    dataset that changes every refresh.

    Always uses cached data for speed. A background thread warms the
    cache on startup so live articles are available on first visit.
    """
    # --- Always use cached data (fast, never blocks on network) ---
    live_articles = get_cached_live_articles_only()

    if live_articles:
        top_news = live_articles[:TOP_NEWS_COUNT]
    else:
        top_news_ids = ALL_ARTICLE_IDS[:TOP_NEWS_COUNT]
        top_news = [get_article(aid) for aid in top_news_ids]

    remaining_ids = ALL_ARTICLE_IDS[TOP_NEWS_COUNT:]
    shuffled_ids = random.sample(remaining_ids, min(FEED_COUNT, len(remaining_ids)))
    feed_articles = [get_article(aid) for aid in shuffled_ids]

    return render_template("index.html", top_news=top_news, articles=feed_articles,
                            heading="Latest Articles")


# ----------------------------------------------------------------------
# ROUTES: LIVE ARTICLE DETAIL
# ----------------------------------------------------------------------
@app.route("/live-article/<int:live_id>")
def live_article_detail(live_id):
    """Detail page for a live news article, showing credibility analysis
    and similar articles from both the dataset and other live articles."""
    article = resolve_live_article(live_id)
    if article is None:
        # Not in memory and never persisted -- there is nothing to render.
        flash("This live article is no longer available. It may have expired from cache.")
        return redirect(url_for("home"))

    similar_articles = article.get("similar_from_dataset", [])
    similar_live = news_api.get_similar_live_articles(article, top_n=5)

    # Don't suggest a story the reader already has open.
    similar_live = [a for a in similar_live if a.get("live_id") != live_id]

    return render_template(
        "live_article.html",
        article=article,
        similar_articles=similar_articles,
        similar_live=similar_live,
    )


@app.route("/full-live-article/<int:live_id>")
def full_live_article(live_id):
    """Show the FULL text of a live article fetched from the publisher's site,
    rendered on our own page (instead of sending the user away). Includes a
    back link to the original article detail page."""
    article = resolve_live_article(live_id)
    if article is None:
        flash("This live article is no longer available. It may have expired from cache.")
        return redirect(url_for("home"))

    full_text = fetch_full_article(article.get("url"))
    return render_template(
        "full_article.html",
        article=article,
        full_title=article.get("title"),
        full_text=full_text,
        back_url=url_for("live_article_detail", live_id=live_id),
    )


# ----------------------------------------------------------------------
# ROUTES: SEARCH
# ----------------------------------------------------------------------
@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    results = []
    live_results = []
    web_results = []
    if query:
        # 1) Archive articles from the dataset (OR word matching)
        words = [w for w in query.split() if len(w) >= 2]
        if words:
            combined_mask = pd.Series([False] * len(ARTICLES_DF), index=ARTICLES_DF.index)
            for w in words:
                word_mask = (
                    ARTICLES_DF["title"].fillna("").str.contains(w, case=False, regex=False)
                    | ARTICLES_DF["category"].fillna("").str.contains(w, case=False, regex=False)
                    | ARTICLES_DF["description"].fillna("").str.contains(w, case=False, regex=False)
                )
                combined_mask = combined_mask | word_mask
            results = ARTICLES_DF[combined_mask].head(40).to_dict(orient="records")

        # 2) Live news already in the cache (fast, no network)
        live_results = filter_live_articles(query)

        # 3) Web search via multi-source aggregator (concurrent fetches;
        #    if NewsAPI quota is hit, Google RSS + GDELT still deliver)
        web_results = search_web_news(query, max_results=12)

    # Dedupe live results by live_id
    seen_live = set()
    unique_live = []
    for a in web_results + live_results:
        if a.get("is_live") and a.get("live_id") in seen_live:
            continue
        if a.get("is_live"):
            seen_live.add(a.get("live_id"))
        unique_live.append(a)

    # Interleave: 1 web/live, 2 dataset, repeat — so results feel mixed
    merged = []
    di = 0
    li = 0
    while li < len(unique_live) or di < len(results):
        if li < len(unique_live):
            merged.append(unique_live[li])
            li += 1
        if di < len(results):
            merged.append(results[di])
            di += 1
        if di < len(results):
            merged.append(results[di])
            di += 1

    return render_template("index.html", top_news=[], articles=merged,
                            heading=f'Search results for "{query}"',
                            is_search=True, query=query,
                            live_count=sum(1 for a in merged if a.get("is_live")))


# ----------------------------------------------------------------------
# ROUTES: CATEGORY
# ----------------------------------------------------------------------
@app.route("/category/<path:category_value>")
def category(category_value):
    # --- Dataset articles: exact category match ---
    matches = ARTICLES_DF[ARTICLES_DF["category"] == category_value].head(24)
    articles = matches.to_dict(orient="records")

    # --- Live articles: fetch using the mapped NewsAPI/RSS category key ---
    news_category = NEWS_CATEGORY_MAP.get(category_value)
    if news_category:
        live_matches = get_live_articles(category=news_category)
    else:
        live_matches = get_live_articles(category=category_value.lower())

    # --- Filter: only keep live articles that were actually fetched for this
    # category. process_articles() now stores the requested category on every
    # article it processes, so this prevents cross-category bleed-through.
    expected_cat = (news_category or category_value).lower()
    live_matches = [a for a in live_matches
                    if a.get("category", "").lower() == expected_cat]

    merged = live_matches + articles
    return render_template("index.html", top_news=[], articles=merged,
                            heading=f"{category_value} News",
                            live_count=len(live_matches))


# ----------------------------------------------------------------------
# ROUTES: ARTICLE DETAIL + RECOMMENDATIONS + LANGUAGE
# ----------------------------------------------------------------------
@app.route("/article/<int:article_id>")
def article_detail(article_id):
    article = get_article(article_id)
    if article is None:
        abort(404)

    lang = request.args.get("lang", "en")
    display_title, title_ok = translate_text(article.get("title", ""), lang)
    display_desc, desc_ok = translate_text(
        article.get("description") or article.get("content", ""), lang
    )
    translation_failed = lang != "en" and not (title_ok and desc_ok)

    recommendations = get_recommendations(article_id, top_n=5)
    live_recommendations = get_live_recommendations(article_id, top_n=3)

    user_reaction = None
    if current_user.is_authenticated:
        existing = Interaction.query.filter_by(
            user_id=current_user.id, article_id=article_id
        ).order_by(Interaction.timestamp.desc()).first()
        if existing:
            user_reaction = existing.action

    return render_template(
        "article.html",
        article=article,
        display_title=display_title,
        display_desc=display_desc,
        current_lang=lang,
        translation_failed=translation_failed,
        recommendations=recommendations,
        live_recommendations=live_recommendations,
        user_reaction=user_reaction,
    )


@app.route("/full-article/<int:article_id>")
def full_article(article_id):
    """Show the FULL text of a dataset article fetched from the publisher's
    site, rendered on our own page (instead of sending the user away).
    Includes a back link to the original article detail page."""
    article = get_article(article_id)
    if article is None:
        abort(404)

    full_text = fetch_full_article(article.get("url"))
    return render_template(
        "full_article.html",
        article=article,
        full_title=article.get("title"),
        full_text=full_text,
        back_url=url_for("article_detail", article_id=article_id),
    )


# ----------------------------------------------------------------------
# ROUTES: LIKE / DISLIKE
# ----------------------------------------------------------------------
@app.route("/react/<int:article_id>/<action>", methods=["POST"])
@login_required
def react(article_id, action):
    if action not in ("like", "dislike"):
        abort(400)
    interaction = Interaction(user_id=current_user.id, article_id=article_id, action=action)
    db.session.add(interaction)
    db.session.commit()
    return redirect(url_for("article_detail", article_id=article_id))


# ----------------------------------------------------------------------
# ROUTES: BOOKMARKS & READING LIST (QoL FEATURE)
# ----------------------------------------------------------------------
@app.route("/bookmark", methods=["POST"])
@login_required
def toggle_bookmark():
    """AJAX endpoint to toggle bookmark on an article (dataset or live).
    Expects JSON or form data: {article_id?, article_key?, title?, source_name?, url_to_image?, url?, category?}
    Returns: {bookmarked: bool, count: int, message: str}
    """
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    article_key = str(data.get("article_key", "")).strip()
    article_id_raw = data.get("article_id")

    try:
        article_id = int(article_id_raw) if article_id_raw is not None and str(article_id_raw).isdigit() else 0
    except (ValueError, TypeError):
        article_id = 0

    if not article_key and article_id:
        article_key = f"ds:{article_id}"
    elif not article_key:
        return jsonify({"error": "Missing article identifier"}), 400

    # Look for existing bookmark
    query = Interaction.query.filter_by(user_id=current_user.id, action="bookmark")
    if article_id and article_key:
        existing = query.filter(db.or_(Interaction.article_key == article_key, Interaction.article_id == article_id)).first()
    elif article_key:
        existing = query.filter_by(article_key=article_key).first()
    else:
        existing = query.filter_by(article_id=article_id).first()

    if existing:
        db.session.delete(existing)
        db.session.commit()
        total_saved = Interaction.query.filter_by(user_id=current_user.id, action="bookmark").count()
        return jsonify({"bookmarked": False, "count": total_saved, "message": "Removed from reading list"})

    title = data.get("title", "")
    source_name = data.get("source_name", "")
    url_to_image = data.get("url_to_image", "")
    url = data.get("url", "")
    category_val = data.get("category", "")

    # Auto-fill metadata if omitted
    if not title:
        if article_id and article_id in ARTICLES_DF.index:
            art = get_article(article_id)
            if art:
                title = art.get("title", "")
                source_name = art.get("source_name", "")
                url_to_image = art.get("url_to_image", "")
                url = art.get("url", "")
                category_val = art.get("category", "")
        elif article_key.startswith("live:"):
            try:
                lid = int(article_key.split(":")[1])
                live_art = resolve_live_article(lid)
                if live_art:
                    title = live_art.get("title", "")
                    source_name = live_art.get("source_name", "")
                    url_to_image = live_art.get("url_to_image", "")
                    url = live_art.get("url", "")
                    category_val = live_art.get("category", "")
            except Exception:
                pass

    interaction = Interaction(
        user_id=current_user.id,
        article_id=article_id,
        article_key=article_key,
        action="bookmark",
        title=title,
        source_name=source_name,
        url_to_image=url_to_image,
        url=url,
        category=category_val,
    )
    db.session.add(interaction)
    db.session.commit()
    total_saved = Interaction.query.filter_by(user_id=current_user.id, action="bookmark").count()
    return jsonify({"bookmarked": True, "count": total_saved, "message": "Saved to reading list! 🔖"})


@app.route("/bookmarks")
@login_required
def bookmarks():
    """View saved articles (reading list)."""
    saved_interactions = Interaction.query.filter_by(
        user_id=current_user.id, action="bookmark"
    ).order_by(Interaction.timestamp.desc()).all()

    bookmarks_list = []
    for item in saved_interactions:
        card = {
            "id": item.id,
            "article_id": item.article_id,
            "article_key": item.article_key or (f"ds:{item.article_id}" if item.article_id else ""),
            "title": item.title or "Saved Article",
            "source_name": item.source_name or "News",
            "url_to_image": item.url_to_image or "",
            "url": item.url or "",
            "category": item.category or "",
            "saved_at": item.timestamp.strftime("%b %d, %Y") if item.timestamp else "",
            "is_live": (item.article_key or "").startswith("live:"),
        }
        if card["is_live"]:
            try:
                lid = int(item.article_key.split(":")[1])
                card["detail_url"] = url_for("live_article_detail", live_id=lid)
            except Exception:
                card["detail_url"] = "#"
        elif item.article_id:
            card["detail_url"] = url_for("article_detail", article_id=item.article_id)
        else:
            card["detail_url"] = item.url or "#"
        bookmarks_list.append(card)

    return render_template("bookmarks.html", bookmarks=bookmarks_list)


@app.route("/bookmarks/remove/<int:interaction_id>", methods=["POST"])
@login_required
def remove_bookmark(interaction_id):
    """Remove an item from saved bookmarks."""
    item = Interaction.query.filter_by(id=interaction_id, user_id=current_user.id, action="bookmark").first()
    if item:
        db.session.delete(item)
        db.session.commit()
        flash("Article removed from reading list.")
    return redirect(url_for("bookmarks"))


# ----------------------------------------------------------------------
# ROUTES: AUTH
# ----------------------------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.")
            return redirect(url_for("signup"))

        user = User(username=username, email=email,
                    password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        if user.is_premium:
            return redirect(url_for("premium_dashboard"))
        return redirect(url_for("home"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user is None or not check_password_hash(user.password_hash, password):
            flash("Incorrect email or password.")
            return redirect(url_for("login"))

        login_user(user)
        if user.is_premium:
            return redirect(url_for("premium_dashboard"))
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


# ----------------------------------------------------------------------
# ROUTES: PREMIUM LANDING PAGE  (mock checkout -- no real payment gateway)
# ----------------------------------------------------------------------
@app.route("/premium")
def premium():
    """Public marketing landing page -- visible even to logged-out visitors.
    Premium users get a link to their personalised dashboard."""
    return render_template("premium.html")


@app.route("/premium/subscribe", methods=["POST"])
def premium_subscribe():
    if not current_user.is_authenticated:
        flash("Create an account first, then subscribe.")
        return redirect(url_for("signup"))

    current_user.is_premium = True
    db.session.add(RevenueLog(user_id=current_user.id, amount=249, plan="premium"))
    db.session.commit()
    flash("You're now a premium subscriber! Enjoy your personalised news experience.")
    return redirect(url_for("premium_dashboard"))


# ----------------------------------------------------------------------
# ROUTES: FAKE NEWS CHECKER
# ----------------------------------------------------------------------
@app.route("/fake-checker", methods=["GET", "POST"])
@login_required
def fake_checker():
    # Premium gate
    if not current_user.is_premium:
        flash("The Fake News Checker is a premium feature. Subscribe to unlock it!")
        return redirect(url_for("premium"))

    result = None
    related_news = []
    related_articles = []
    submitted_text = ""

    if request.method == "POST":
        submitted_text = request.form.get("article_text", "").strip()
        if submitted_text:
            try:
                # --- Web coverage for this story (Google News RSS + NewsAPI) ---
                related_news = gather_web_evidence(submitted_text)

                # The classifier checks the text against the gathered web
                # coverage and lets Gemini produce the final Real/Fake verdict.
                result = classify_fake_news(submitted_text, web_results=related_news)
                related_articles = find_related_articles(submitted_text, top_n=5)
            except Exception:
                result = {
                    "label": "Uncertain",
                    "confidence": 0.0,
                    "analysis": "An error occurred during analysis. Please try again.",
                    "classifier": "ML",
                }

    return render_template(
        "fake_checker.html",
        submitted_text=submitted_text,
        result=result,
        related_news=related_news,
        related_articles=related_articles,
    )


# ----------------------------------------------------------------------
# ROUTES: PUSH NOTIFICATIONS (real browser Notification API)
# ----------------------------------------------------------------------
@app.route("/api/top-headline")
def api_top_headline():
    """JSON endpoint the notification button fetches before showing a
    real browser notification.
    """
    top = get_article(ALL_ARTICLE_IDS[0])
    return jsonify({
        "title": top["title"],
        "url": url_for("article_detail", article_id=top["article_id"]),
    })


# ----------------------------------------------------------------------
# ROUTES: ADMIN DASHBOARD
# ----------------------------------------------------------------------
@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        abort(403)

    total_users = User.query.count()
    premium_users = User.query.filter_by(is_premium=True).count()
    total_revenue = db.session.query(db.func.sum(RevenueLog.amount)).scalar() or 0
    total_interactions = Interaction.query.count()
    likes = Interaction.query.filter_by(action="like").count()
    dislikes = Interaction.query.filter_by(action="dislike").count()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        premium_users=premium_users,
        total_revenue=total_revenue,
        total_interactions=total_interactions,
        likes=likes,
        dislikes=dislikes,
    )


# ----------------------------------------------------------------------
# ROUTES: ADMIN USER MANAGEMENT
# ----------------------------------------------------------------------
@app.route("/admin/users")
@login_required
def admin_users():
    if not current_user.is_admin:
        abort(403)

    query = request.args.get("q", "").strip()
    if query:
        users = User.query.filter(
            db.or_(
                User.username.ilike(f"%{query}%"),
                User.email.ilike(f"%{query}%")
            )
        ).order_by(User.created_at.desc()).all()
    else:
        users = User.query.order_by(User.created_at.desc()).all()

    return render_template("admin_users.html", users=users, query=query)


@app.route("/admin/users/<int:user_id>/update", methods=["POST"])
@login_required
def admin_update_user(user_id):
    if not current_user.is_admin:
        abort(403)

    user = db.session.get(User, user_id)
    if not user:
        flash("User not found.")
        return redirect(url_for("admin_users"))

    user.username = request.form.get("username", user.username).strip()
    user.email = request.form.get("email", user.email).strip()
    user.is_premium = "is_premium" in request.form
    user.is_admin = "is_admin" in request.form

    new_password = request.form.get("new_password", "").strip()
    if new_password:
        user.password_hash = generate_password_hash(new_password)

    db.session.commit()
    flash(f"User '{user.username}' updated successfully.")
    return redirect(url_for("admin_users"))


# ----------------------------------------------------------------------
# ROUTES: PREMIUM PERSONALISED NEWS DASHBOARD
# ----------------------------------------------------------------------
@app.route("/premium/dashboard")
@login_required
def premium_dashboard():
    """Premium personalised news experience -- only accessible to premium
    subscribers.  Uses ONLY cached articles to avoid blocking on network."""
    if not current_user.is_premium:
        flash("Upgrade to premium to access personalised news!")
        return redirect(url_for("premium"))

    # Gather ONLY cached live articles (never triggers network fetches)
    all_live = []
    for cat in ["general", "technology", "business", "sports", "entertainment"]:
        try:
            arts = get_cached_live_articles_only(category=cat)
            all_live.extend(arts)
        except Exception:
            pass

    # Fallback: if no cached live articles, use shuffled dataset articles
    if not all_live:
        sample_ids = random.sample(ALL_ARTICLE_IDS, min(30, len(ALL_ARTICLE_IDS)))
        for aid in sample_ids:
            art = get_article(aid)
            if art:
                art["article_key"] = "ds:{}".format(aid)
                art["is_live"] = False
                all_live.append(art)

    # Deduplicate by live_id
    seen = set()
    unique_live = []
    for a in all_live:
        lid = a.get("live_id")
        key = a.get("article_key", "")
        if lid:
            if lid in seen:
                continue
            seen.add(lid)
            a["article_key"] = "live:{}".format(lid)
        elif not key:
            continue
        unique_live.append(a)

    # Build personalised feed
    feed = premium_engine.get_premium_feed(
        current_user, unique_live, top_n=20, classify_fn=None
    )

    # Regional news section
    regional = premium_engine.get_regional_news(current_user, unique_live, top_n=6)

    # Trending news section
    trending = premium_engine.get_trending_news(unique_live, top_n=6)

    # Group by region
    region_groups = premium_engine.group_by_region(feed, current_user.city or current_user.state_region)

    # Regional trends (in-memory, no DB writes)
    trend_region = current_user.state_region or current_user.country or "India"
    trends = premium_engine.get_regional_trends(trend_region, level="state", top_n=10)

    return render_template(
        "premium_dashboard.html",
        feed=feed,
        regional=regional,
        trending=trending,
        region_groups=region_groups,
        trends=trends,
        trend_region=trend_region,
        is_cold_start=len(premium_engine.get_user_interests(current_user.id)) < 3,
    )


@app.route("/premium/feed", methods=["POST"])
@login_required
def premium_feed_api():
    """AJAX endpoint: returns the next page of personalised articles as JSON.
    Accepts JSON: {offset: int, limit: int, exclude: [article_key, ...]}
    Returns: {articles: [...], has_more: bool}
    """
    if not current_user.is_premium:
        return jsonify({"error": "Premium required"}), 403

    data = request.get_json(silent=True) or {}
    offset = int(data.get("offset", 0))
    limit = min(int(data.get("limit", 10)), 20)
    exclude = set(data.get("exclude", []))

    # Gather all cached live articles
    all_live = []
    for cat in ["general", "technology", "business", "sports", "entertainment",
                "health", "science", "education", "politics"]:
        try:
            arts = get_cached_live_articles_only(category=cat)
            all_live.extend(arts)
        except Exception:
            pass

    # Fallback: dataset articles
    if not all_live:
        sample_ids = random.sample(ALL_ARTICLE_IDS, min(60, len(ALL_ARTICLE_IDS)))
        for aid in sample_ids:
            art = get_article(aid)
            if art:
                art["article_key"] = "ds:{}".format(aid)
                art["is_live"] = False
                all_live.append(art)

    # Deduplicate + assign keys
    seen = set()
    unique = []
    for a in all_live:
        lid = a.get("live_id")
        key = a.get("article_key", "")
        if lid:
            if lid in seen:
                continue
            seen.add(lid)
            a["article_key"] = "live:{}".format(lid)
        elif not key:
            continue
        if a["article_key"] in exclude:
            continue
        unique.append(a)

    # Build full personalised feed
    feed = premium_engine.get_premium_feed(
        current_user, unique, top_n=len(unique), classify_fn=None
    )

    # Slice for pagination
    page = feed[offset:offset + limit]
    has_more = (offset + limit) < len(feed)

    # Strip internal fields, build JSON-safe list
    def _s(val, default=""):
        """Sanitize a value: NaN/None -> default, else str."""
        if val is None or (isinstance(val, float) and val != val):
            return default
        return str(val)

    articles = []
    for a in page:
        v = a.get("verification", {})
        card = {
            "article_key": _s(a.get("article_key")),
            "title": _s(a.get("title")),
            "description": _s(a.get("description"))[:200],
            "source_name": _s(a.get("source_name")),
            "published_at": _s(a.get("published_at"))[:10],
            "category": _s(a.get("category")),
            "url_to_image": _s(a.get("url_to_image")),
            "is_live": bool(a.get("is_live", False)),
            "recommendation_reason": _s(a.get("recommendation_reason")),
            "verification": {
                "status": _s(v.get("status"), "Checking"),
                "confidence": v.get("confidence", 0) if isinstance(v.get("confidence"), (int, float)) else 0,
            },
        }
        if a.get("is_live"):
            card["detail_url"] = url_for("live_article_detail", live_id=a.get("live_id"))
        elif a.get("article_id"):
            card["detail_url"] = url_for("article_detail", article_id=a.get("article_id"))
        else:
            card["detail_url"] = "#"
        articles.append(card)

    return jsonify({"articles": articles, "has_more": has_more})


# ----------------------------------------------------------------------
# ROUTES: PREMIUM PREFERENCES
# ----------------------------------------------------------------------
@app.route("/premium/preferences", methods=["GET", "POST"])
@login_required
def premium_preferences():
    """Let premium users set region, city, language, and topic interests."""
    if not current_user.is_premium:
        flash("Upgrade to premium to personalise your experience!")
        return redirect(url_for("premium"))

    if request.method == "POST":
        country = request.form.get("country", "").strip()
        state = request.form.get("state_region", "").strip()
        city = request.form.get("city", "").strip()
        language = request.form.get("language", "en").strip()
        topics_raw = request.form.get("topics", "").strip()

        premium_engine.update_user_preferences(
            current_user,
            country=country or "India",
            state_region=state,
            city=city,
            language=language or "en",
        )

        if topics_raw:
            topics = [t.strip() for t in re.split(r"[,;]+", topics_raw) if t.strip()]
            premium_engine.set_user_interests(current_user.id, topics)

        flash("Preferences saved!")
        return redirect(url_for("premium_dashboard"))

    interests = premium_engine.get_user_interests(current_user.id)
    return render_template(
        "premium_preferences.html",
        interests=interests,
    )


# ----------------------------------------------------------------------
# ROUTES: PREMIUM BEHAVIOUR TRACKING (AJAX)
# ----------------------------------------------------------------------
@app.route("/premium/track", methods=["POST"])
@login_required
def premium_track():
    """AJAX endpoint to track reading behaviour.  Expects JSON:
    {article_key, action, duration?, topics?, region?}
    """
    data = request.get_json(silent=True) or {}
    article_key = data.get("article_key", "")
    action = data.get("action", "click")
    duration = float(data.get("duration", 0))
    topics = data.get("topics", "")
    region = data.get("region", "")

    if article_key:
        premium_engine.track_behavior(
            current_user.id, article_key, action,
            duration=duration, topics=topics, region=region,
        )
    return jsonify({"ok": True})


# ----------------------------------------------------------------------
# ROUTES: VERIFY ARTICLE (AJAX, public — no login required)
# ----------------------------------------------------------------------
@app.route("/verify-article", methods=["POST"])
def verify_article():
    """AJAX endpoint for async fake/real verification of any article.
    Checks trusted source directory and database verification cache first before calling Gemini API.
    Expects JSON: {title, description, source_name?}
    Returns: {label, confidence, analysis, classifier}
    """
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    source_name = (data.get("source_name") or "").strip().lower()
    text = f"{title} {description}".strip()

    if len(text) < 10:
        return jsonify({"label": "Uncertain", "confidence": 0, "analysis": "Insufficient text for fact-checking."})

    # Fast path: Recognized reputable news agency
    if source_name and source_name in news_api.TRUSTED_SOURCES:
        return jsonify({
            "label": "Real",
            "confidence": 0.98,
            "analysis": f"Published by {source_name.title()}, a verified global news organization.",
            "classifier": "Verified Source",
        })

    result = classify_fake_news(text)
    return jsonify(result)


# ----------------------------------------------------------------------
# ROUTES: PREMIUM VERIFY ARTICLE (AJAX)
# ----------------------------------------------------------------------
@app.route("/premium/verify", methods=["POST"])
@login_required
def premium_verify():
    """AJAX endpoint to trigger/return verification for an article.
    Expects JSON: {article_key, title, description}
    """
    data = request.get_json(silent=True) or {}
    article_key = data.get("article_key", "")
    title = data.get("title", "")
    description = data.get("description", "")

    if not article_key:
        return jsonify({"error": "Missing article_key"}), 400

    result = premium_engine.verify_article(
        article_key, title=title, description=description,
        classify_fn=classify_fake_news,
    )
    return jsonify(result)


# ----------------------------------------------------------------------
# ERROR HANDLERS (QoL CUSTOM PAGES)
# ----------------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    try:
        db.session.rollback()
    except Exception:
        pass
    return render_template("500.html"), 500


# ----------------------------------------------------------------------
# DATABASE + STARTUP
# ----------------------------------------------------------------------
def init_database():
    """Create any missing tables and back-fill custom schema columns.

    Runs at import time, not under __main__, because render.yaml starts the
    app with `gunicorn app:app` -- which never executes the __main__ block.
    create_all() is idempotent, so calling it on every boot is safe and
    also picks up new tables (e.g. live_article, interaction updates).
    """
    with app.app_context():
        db.create_all()
        _ensure_schema_columns()


init_database()


def _warm_live_cache():
    """Prefetch every category feed so the first page visit is fast."""
    import time as _time
    with app.app_context():
        try:
            print("[Cache] Warming live-news cache in background...")
            for cat in ["general", "technology", "business", "sports", "entertainment",
                        "health", "science", "education", "politics", "travel", "food"]:
                try:
                    get_live_articles(category=cat)
                except Exception:
                    pass
            print("[Cache] Done. Live news is now cached.")
        except Exception as e:
            print(f"[Cache] Warm-up failed: {e}")


# ----------------------------------------------------------------------
# ROUTES: ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import threading, time as _time

    # Warm live-news cache in background so first page visit is fast
    def _warm_cache():
        _time.sleep(2)  # let Flask start serving before blocking on network
        threading.Thread(target=_warm_live_cache, daemon=True).start()

    threading.Thread(target=_warm_cache, daemon=True).start()

    # Host/port come from the environment so this works both on a laptop and on
    # Render, and so it never silently lands on a different port than expected.
    # 0.0.0.0 also makes the site reachable from another device on the same Wi-Fi,
    # which is usually what you want when you hand the project to a friend.
    _port = int(os.environ.get("PORT", 5000))

    # The interactive debugger is OFF unless you explicitly opt in. Leaving it on
    # is not just a nuisance: Werkzeug's debugger executes arbitrary Python from a
    # browser console, so anyone who can reach your machine gets a shell on it.
    # It also forks a second process, which makes "is the app running?" ambiguous.
    _debug = os.environ.get("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    print(f"Starting on http://127.0.0.1:{_port}  (debug={_debug})")

    app.run(host="0.0.0.0", port=_port, debug=_debug)
