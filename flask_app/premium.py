"""
premium.py -- Premium Personalisation Engine

Provides regional classification, language detection, user-interest
profiling, regional-trend computation, article verification (reusing
the existing Gemini pipeline), and a multi-signal ranking engine that
works *alongside* -- not instead of -- the existing TF-IDF recommender.

All public functions accept the Flask `db` session and the `User` object
so they integrate with the existing auth system without duplication.
"""

import re
import time
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict

from models import (
    db, User, UserInterest, UserBehavior, ArticleMetadata,
    ArticleVerification, RegionalTrend, Interaction,
)

# ======================================================================
# REGIONAL CLASSIFIER
# ======================================================================

INDIAN_STATES = {
    "andhra pradesh": "Andhra Pradesh", "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam", "bihar": "Bihar", "chhattisgarh": "Chhattisgarh",
    "goa": "Goa", "gujarat": "Gujarat", "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh", "jharkhand": "Jharkhand",
    "karnataka": "Karnataka", "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh", "maharashtra": "Maharashtra",
    "manipur": "Manipur", "meghalaya": "Meghalaya", "mizoram": "Mizoram",
    "nagaland": "Nagaland", "odisha": "Odisha", "punjab": "Punjab",
    "rajasthan": "Rajasthan", "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu", "telangana": "Telangana",
    "tripura": "Tripura", "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand", "west bengal": "West Bengal",
    "delhi": "Delhi", "new delhi": "Delhi",
    "jammu and kashmir": "Jammu & Kashmir", "ladakh": "Ladakh",
    "chandigarh": "Chandigarh", "puducherry": "Puducherry",
    "andaman and nicobar": "Andaman & Nicobar",
    "dadra and nagar haveli": "Dadra & Nagar Haveli",
    "lakshadweep": "Lakshadweep",
}

INDIAN_CITIES = {
    "ahmedabad": ("Ahmedabad", "Gujarat"), "surat": ("Surat", "Gujarat"),
    "vadodara": ("Vadodara", "Gujarat"), "rajkot": ("Rajkot", "Gujarat"),
    "mumbai": ("Mumbai", "Maharashtra"), "pune": ("Pune", "Maharashtra"),
    "nagpur": ("Nagpur", "Maharashtra"), "thane": ("Thane", "Maharashtra"),
    "bangalore": ("Bengaluru", "Karnataka"), "bengaluru": ("Bengaluru", "Karnataka"),
    "hyderabad": ("Hyderabad", "Telangana"), "chennai": ("Chennai", "Tamil Nadu"),
    "kolkata": ("Kolkata", "West Bengal"), "delhi": ("New Delhi", "Delhi"),
    "new delhi": ("New Delhi", "Delhi"),
    "jaipur": ("Jaipur", "Rajasthan"), "lucknow": ("Lucknow", "Uttar Pradesh"),
    "chandigarh": ("Chandigarh", "Chandigarh"), "bhopal": ("Bhopal", "Madhya Pradesh"),
    "patna": ("Patna", "Bihar"), "guwahati": ("Guwahati", "Assam"),
    "bhubaneswar": ("Bhubaneswar", "Odisha"), "thiruvananthapuram": ("Thiruvananthapuram", "Kerala"),
    "coimbatore": ("Coimbatore", "Tamil Nadu"), "indore": ("Indore", "Madhya Pradesh"),
    "nashik": ("Nashik", "Maharashtra"), "visakhapatnam": ("Visakhapatnam", "Andhra Pradesh"),
    "kanpur": ("Kanpur", "Uttar Pradesh"), "noida": ("Noida", "Uttar Pradesh"),
    "gurgaon": ("Gurugram", "Haryana"), "gurugram": ("Gurugram", "Haryana"),
    "faridabad": ("Faridabad", "Haryana"),
}

COUNTRY_KEYWORDS = {
    "united states": "USA", "american": "USA", "us president": "USA",
    "india": "India", "indian": "India",
    "isro": "India", "hindustan": "India", "bollywood": "India",
    "ipl": "India", "rupee": "India", "rupees": "India",
    "china": "China", "chinese": "China",
    "united kingdom": "UK", "britain": "UK", "british": "UK", "uk ": "UK",
    "russia": "Russia", "russian": "Russia",
    "japan": "Japan", "japanese": "Japan",
    "european": "Europe", "europe": "Europe",
    "australia": "Australia", "australian": "Australia",
    "african": "Africa", "africa": "Africa",
    "brazil": "Brazil", "brazilian": "Brazil",
    "france": "France", "french": "France",
    "germany": "Germany", "german": "Germany",
}


def classify_region(text):
    """Extract region info from article text. Returns dict with
    country, state_region, city."""
    if not text:
        return {"country": "", "state_region": "", "city": ""}
    low = text.lower()

    city_found = ""
    state_found = ""
    for city_key, (city_name, state_name) in INDIAN_CITIES.items():
        if re.search(r"\b" + re.escape(city_key) + r"\b", low):
            city_found = city_name
            state_found = state_name
            break

    if not state_found:
        for state_key, state_name in INDIAN_STATES.items():
            if re.search(r"\b" + re.escape(state_key) + r"\b", low):
                state_found = state_name
                break

    country_found = ""
    for kw, country in COUNTRY_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", low):
            country_found = country
            break
    if not country_found and state_found:
        country_found = "India"

    return {"country": country_found, "state_region": state_found, "city": city_found}


# ======================================================================
# LANGUAGE DETECTION (script-based, no external deps)
# ======================================================================

_SCRIPT_RANGES = [
    ("\u0900", "\u097F", "hi"),   # Devanagari (Hindi, Marathi)
    ("\u0980", "\u09FF", "bn"),   # Bengali
    ("\u0A00", "\u0A7F", "pa"),   # Gurmukhi (Punjabi)
    ("\u0A80", "\u0AFF", "gu"),   # Gujarati
    ("\u0B00", "\u0B7F", "or"),   # Odia
    ("\u0B80", "\u0BFF", "ta"),   # Tamil
    ("\u0C00", "\u0C7F", "te"),   # Telugu
    ("\u0C80", "\u0CFF", "kn"),   # Kannada
    ("\u0D00", "\u0D7F", "ml"),   # Malayalam
]


def detect_language(text):
    """Detect article language by character script analysis.
    Returns ISO 639-1 code (default 'en')."""
    if not text:
        return "en"
    sample = text[:2000]
    counts = Counter()
    for ch in sample:
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= ch <= hi:
                counts[lang] += 1
                break
    if not counts:
        return "en"
    best, cnt = counts.most_common(1)[0]
    if cnt > len(sample) * 0.05:
        return best
    return "en"


# ======================================================================
# TOPIC / CATEGORY EXTRACTION
# ======================================================================

_TOPIC_KEYWORDS = {
    "cricket": ["cricket", "ipl", "test match", "odi", "t20", "batsman", "bowler", "wicket"],
    "badminton": ["badminton", "shuttler", "pv sindhu", "saina nehwal", "bwf"],
    "football": ["football", "soccer", "premier league", "fifa", "champions league"],
    "tennis": ["tennis", "atp", "wta", "grand slam", "wimbledon", "roland garros"],
    "hockey": ["hockey", "hockey india", "fih"],
    "ai": [" ai ", "ai ", " ai", "artificial intelligence", "machine learning", "deep learning", "neural network", "llm", "chatgpt", "openai", "gemini"],
    "technology": ["technology", "tech", "software", "hardware", "startup", "app", "digital", "cyber"],
    "politics": ["politics", "election", "parliament", "minister", "government", "bjp", "congress", "mla", "mp"],
    "business": ["business", "stock", "market", "economy", "gdp", "inflation", "nifty", "sensex", "share"],
    "entertainment": ["bollywood", "movie", "film", "actor", "actress", "netflix", "series", "celebrity", "song"],
    "health": ["health", "medical", "hospital", "doctor", "vaccine", "disease", "covid", "mental health"],
    "science": ["science", "research", "study", "nasa", "isro", "space", "discovery", "physics"],
    "education": ["education", "university", "school", "exam", "student", "neet", "jee", "upsc"],
    "climate": ["climate", "environment", "pollution", "green", "carbon", "emission", "sustainability"],
    "defence": ["defence", "military", "army", "navy", "air force", "fighter jet", "missile", "border"],
}


def extract_topics(text):
    """Extract relevant topics from article text. Returns comma-separated string."""
    if not text:
        return ""
    low = text.lower()
    found = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                found.append(topic)
                break
    return ",".join(found)


# ======================================================================
# USER INTEREST ENGINE
# ======================================================================

# Behaviour signal weights
_ACTION_WEIGHTS = {
    "click": 0.10,
    "read": 0.20,
    "long_read": 0.35,
    "save": 0.50,
    "like": 0.40,
    "bookmark": 0.50,
    "dislike": -0.30,
    "skip": -0.05,
}

# Learning rate (how much a new interaction shifts the score)
_ALPHA = 0.15


def update_user_interests(user_id, topics_csv, action, duration=0.0):
    """Update interest scores based on a new behaviour event.

    Topics are comma-separated.  The score is updated via exponential
    moving average so single events cause gradual shifts.
    """
    if not topics_csv:
        return
    topics = [t.strip().lower() for t in topics_csv.split(",") if t.strip()]
    if not topics:
        return

    weight = _ACTION_WEIGHTS.get(action, 0.1)
    if action == "read" and duration > 60:
        weight = _ACTION_WEIGHTS["long_read"]

    for topic in topics:
        existing = UserInterest.query.filter_by(user_id=user_id, topic=topic).first()
        if existing:
            old = existing.score
            existing.score = max(0.0, min(1.0, old + _ALPHA * (weight - old * 0.5)))
            existing.last_updated = datetime.utcnow()
        else:
            initial = max(0.0, min(1.0, weight * 0.5))
            db.session.add(UserInterest(
                user_id=user_id, topic=topic,
                score=initial, last_updated=datetime.utcnow(),
            ))
    db.session.commit()


def get_user_interests(user_id):
    """Return dict of {topic: score} sorted by score descending."""
    rows = UserInterest.query.filter_by(user_id=user_id).order_by(
        UserInterest.score.desc()
    ).all()
    return {r.topic: r.score for r in rows}


def track_behavior(user_id, article_key, action, duration=0.0, topics="", region=""):
    """Record a behaviour event and update interest scores."""
    db.session.add(UserBehavior(
        user_id=user_id, article_key=article_key,
        action=action, duration_sec=duration,
        topics=topics, region=region,
    ))
    db.session.commit()
    if topics:
        update_user_interests(user_id, topics, action, duration)


def get_reading_history(user_id, limit=50):
    """Return recent behaviour events for a user."""
    return UserBehavior.query.filter_by(user_id=user_id).order_by(
        UserBehavior.timestamp.desc()
    ).limit(limit).all()


# ======================================================================
# REGIONAL TREND ENGINE
# ======================================================================

def compute_regional_trends(region, level="state", lookback_hours=24, min_articles=2):
    """Compute trending topics for a region from recently ingested articles.
    Stores results in RegionalTrend table."""
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    meta_rows = ArticleMetadata.query.filter(
        ArticleMetadata.extracted_at >= cutoff,
    ).all()

    region_key_map = {
        "country": "country", "state": "state_region", "city": "city",
    }
    col = region_key_map.get(level, "state_region")

    topic_counter = Counter()
    topic_articles = defaultdict(int)
    for m in meta_rows:
        region_val = getattr(m, col, "")
        if not region_val:
            continue
        if region_val.lower() != region.lower():
            continue
        for t in (m.topics or "").split(","):
            t = t.strip().lower()
            if t:
                topic_counter[t] += 1
                topic_articles[t] += 1

    RegionalTrend.query.filter_by(region=region, level=level).delete()
    for topic, count in topic_counter.most_common(30):
        db.session.add(RegionalTrend(
            region=region, level=level, topic=topic,
            score=float(count), article_count=topic_articles[topic],
        ))
    db.session.commit()


def get_regional_trends(region, level="state", top_n=15):
    """Return top trending topics for a region."""
    rows = RegionalTrend.query.filter_by(region=region, level=level).order_by(
        RegionalTrend.score.desc()
    ).limit(top_n).all()
    return [{"topic": r.topic, "score": r.score, "count": r.article_count} for r in rows]


# ======================================================================
# ARTICLE METADATA EXTRACTION
# ======================================================================

def extract_article_metadata(article_key, title="", description="", source="", category=""):
    """Extract and cache geographic / language / topic metadata for an article.
    Uses existing ArticleMetadata table to avoid re-processing."""
    existing = ArticleMetadata.query.filter_by(article_key=article_key).first()
    if existing:
        return existing

    text = "{} {} {}".format(title or "", description or "", source or "")
    region = classify_region(text)
    lang = detect_language(text)
    topics = extract_topics(text)

    domain = ""
    if source:
        domain = source.lower().replace(" ", "")

    meta = ArticleMetadata(
        article_key=article_key,
        country=region["country"],
        state_region=region["state_region"],
        city=region["city"],
        language=lang,
        topics=topics,
        category=category or "",
        source_domain=domain,
    )
    db.session.add(meta)
    db.session.commit()
    return meta


# ======================================================================
# VERIFICATION SERVICE (wraps existing classify_fake_news)
# ======================================================================

def verify_article(article_key, title="", description="", classify_fn=None):
    """Run fake-news verification on an article. Reuses the existing
    classify_fake_news() function and caches the result in ArticleVerification.

    classify_fn must be passed from app.py to avoid circular imports.
    Returns the verification dict: {status, confidence, analysis, latency_ms}.
    """
    existing = ArticleVerification.query.filter_by(article_key=article_key).first()
    if existing:
        return {
            "status": existing.status,
            "confidence": existing.confidence,
            "analysis": existing.analysis,
            "latency_ms": existing.latency_ms,
            "cached": True,
        }

    if not classify_fn:
        return {
            "status": "Unverified",
            "confidence": 0.0,
            "analysis": "Verification service not available.",
            "latency_ms": 0.0,
            "cached": False,
        }

    text = "{} {}".format(title or "", description or "").strip()
    if len(text) < 10:
        return {
            "status": "Unverified",
            "confidence": 0.0,
            "analysis": "Insufficient text to verify.",
            "latency_ms": 0.0,
            "cached": False,
        }

    t0 = time.time()
    try:
        result = classify_fn(text)
    except Exception:
        result = {"label": "Unverified", "confidence": 0.0, "analysis": "Verification failed."}
    latency_ms = (time.time() - t0) * 1000

    label = result.get("label", "Unverified")
    if label == "Real":
        status = "Real"
    elif label == "Fake":
        status = "Fake"
    else:
        status = "Unverified"

    db.session.add(ArticleVerification(
        article_key=article_key,
        status=status,
        confidence=result.get("confidence", 0.0),
        analysis=result.get("analysis", ""),
        latency_ms=round(latency_ms, 1),
    ))
    db.session.commit()

    return {
        "status": status,
        "confidence": result.get("confidence", 0.0),
        "analysis": result.get("analysis", ""),
        "latency_ms": round(latency_ms, 1),
        "cached": False,
    }


def batch_verify_articles(articles, classify_fn=None, max_batch=10):
    """Verify a list of articles, returning updated articles with verification.
    Only verifies articles not already cached. Returns verified count."""
    verified = 0
    for art in articles[:max_batch]:
        key = art.get("article_key", "")
        if not key:
            continue
        v = verify_article(
            key,
            title=art.get("title", ""),
            description=art.get("description", ""),
            classify_fn=classify_fn,
        )
        art["verification"] = v
        verified += 1
    return verified


# ======================================================================
# PREMIUM RANKING ENGINE
# ======================================================================

# Weights for the personalisation score
W_TOPIC = 0.30
W_REGIONAL = 0.25
W_LANGUAGE = 0.15
W_CATEGORY = 0.15
W_SOURCE = 0.10
W_FRESHNESS = 0.05


def _topic_interest_score(article_topics_csv, user_interests):
    """Score 0-1: how much the article matches user interests."""
    if not article_topics_csv or not user_interests:
        return 0.0
    article_topics = [t.strip().lower() for t in article_topics_csv.split(",") if t.strip()]
    if not article_topics:
        return 0.0
    scores = [user_interests.get(t, 0.0) for t in article_topics]
    return max(scores) if scores else 0.0


def _regional_score(article_meta, user):
    """Score 0-1: how relevant the article's region is to the user."""
    if not article_meta:
        return 0.0
    score = 0.0
    if article_meta.city and user.city and article_meta.city.lower() == user.city.lower():
        score = 1.0
    elif article_meta.state_region and user.state_region and article_meta.state_region.lower() == user.state_region.lower():
        score = 0.7
    elif article_meta.country and user.country and article_meta.country.lower() == user.country.lower():
        score = 0.4
    elif article_meta.country and article_meta.country.lower() == "india":
        score = 0.2
    return score


def _language_score(article_meta, user):
    """Score 0-1: language match."""
    if not article_meta or not user.preferred_language:
        return 0.5
    if article_meta.language == user.preferred_language:
        return 1.0
    if article_meta.language == "en" and user.preferred_language == "en":
        return 1.0
    return 0.2


def _category_score(article, user_interests):
    """Score 0-1: category interest."""
    cat = (article.get("category") or "").lower()
    if not cat:
        return 0.0
    return user_interests.get(cat, 0.0)


def _source_score(article):
    """Score 0-1: source trustworthiness."""
    src = (article.get("source_name") or "").lower().strip()
    # Import here to avoid circular imports at module level
    from news_api import TRUSTED_SOURCES
    if src in TRUSTED_SOURCES:
        return 1.0
    return 0.3


def _freshness_score(article):
    """Score 0-1: how fresh the article is."""
    pub = article.get("published_at") or ""
    if not pub:
        return 0.3
    try:
        if isinstance(pub, str):
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00").replace("+00:00", ""))
        else:
            dt = pub
        age_hours = (datetime.utcnow() - dt).total_seconds() / 3600
        if age_hours < 1:
            return 1.0
        if age_hours < 6:
            return 0.8
        if age_hours < 24:
            return 0.6
        if age_hours < 72:
            return 0.4
        return 0.2
    except Exception:
        return 0.3


def compute_premium_score(article, user, user_interests, article_meta=None):
    """Compute the personalisation score for an article given a user.

    Returns (personalisation_score, reason_text).
    The personalisation_score is 0.0 -- 1.0.
    reason_text is a human-readable explanation of why the article was recommended.
    """
    art_topics = ""
    if article_meta:
        art_topics = article_meta.topics or ""

    scores = {
        "topic": _topic_interest_score(art_topics, user_interests),
        "regional": _regional_score(article_meta, user),
        "language": _language_score(article_meta, user),
        "category": _category_score(article, user_interests),
        "source": _source_score(article),
        "freshness": _freshness_score(article),
    }

    total = (
        W_TOPIC * scores["topic"]
        + W_REGIONAL * scores["regional"]
        + W_LANGUAGE * scores["language"]
        + W_CATEGORY * scores["category"]
        + W_SOURCE * scores["source"]
        + W_FRESHNESS * scores["freshness"]
    )

    # Build recommendation reason
    reasons = []
    if scores["topic"] > 0.5:
        top_topics = [t for t in (art_topics or "").split(",") if t.strip()]
        if top_topics:
            reasons.append("Based on your interest: {}".format(top_topics[0].title()))
    if scores["regional"] >= 0.7:
        if article_meta and article_meta.city:
            reasons.append("Trending in {}".format(article_meta.city))
        elif article_meta and article_meta.state_region:
            reasons.append("Regional: {}".format(article_meta.state_region))
    elif scores["regional"] >= 0.4:
        reasons.append("National news")
    if not reasons:
        reasons.append("Recommended for you")

    return total, reasons[0]


# ======================================================================
# DIVERSITY FILTER
# ======================================================================

def diversify_results(scored_articles, top_n=20):
    """Apply diversity: limit same-topic repetitions and mix in
    regional, trending, and exploratory content.

    scored_articles: list of (score, article, reason) sorted by score desc.
    Returns top_n articles with diversity constraints.
    """
    if not scored_articles:
        return []

    # Target distribution
    MAX_SAME_TOPIC = 3
    result = []
    topic_count = Counter()
    seen_keys = set()

    for score, article, reason in scored_articles:
        key = article.get("article_key") or article.get("article_id") or article.get("live_id", "")
        if key in seen_keys:
            continue
        seen_keys.add(key)

        topics_str = ""
        meta = article.get("_meta")
        if meta:
            topics_str = meta.topics or ""
        elif article.get("category"):
            topics_str = article.get("category", "")

        primary_topic = topics_str.split(",")[0].strip() if topics_str else "general"

        if topic_count[primary_topic] >= MAX_SAME_TOPIC:
            continue

        topic_count[primary_topic] += 1
        result.append((score, article, reason))

        if len(result) >= top_n:
            break

    return result


# ======================================================================
# COLD START
# ======================================================================

def get_cold_start_recommendations(user, all_live_articles, top_n=20):
    """Generate recommendations for a new premium user with no reading history.
    Uses region, language, and fresh news as signals."""
    region = user.state_region or user.country or "India"
    lang = user.preferred_language or "en"

    scored = []
    for art in all_live_articles:
        meta = extract_article_metadata(
            art.get("article_key", ""),
            title=art.get("title", ""),
            description=art.get("description", ""),
            source=art.get("source_name", ""),
        )
        art["_meta"] = meta

        reg_score = _regional_score(meta, user)
        lang_score = _language_score(meta, user)
        fresh = _freshness_score(art)
        total = 0.20 * reg_score + 0.20 * lang_score + 0.60 * fresh
        reason = "Fresh news"
        if reg_score >= 0.7:
            reason = "Popular in your region"
        elif lang_score >= 0.9:
            reason = "In your preferred language"
        scored.append((total, art, reason))

    scored.sort(key=lambda x: x[0], reverse=True)
    return diversify_results(scored, top_n)


# ======================================================================
# FULL PREMIUM PIPELINE
# ======================================================================

def get_premium_feed(user, all_live_articles, top_n=20, classify_fn=None):
    """Build the full personalised premium feed for a user.

    1. Extract metadata for all articles
    2. Compute user interests
    3. Rank by personalisation score
    4. Apply diversity filter
    5. Verify articles for fake-news status
    6. Return list of articles with verification + reason
    """
    user_interests = get_user_interests(user.id)
    is_cold_start = len(user_interests) < 3

    if is_cold_start:
        scored = get_cold_start_recommendations(user, all_live_articles, top_n)
    else:
        scored = []
        for art in all_live_articles:
            key = art.get("article_key", "")
            if not key:
                continue
            meta = extract_article_metadata(
                key,
                title=art.get("title", ""),
                description=art.get("description", ""),
                source=art.get("source_name", ""),
                category=art.get("category", ""),
            )
            art["_meta"] = meta

            score, reason = compute_premium_score(art, user, user_interests, meta)
            scored.append((score, art, reason))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = diversify_results(scored, top_n)

    # Attach verification status
    for score, art, reason in scored:
        art["recommendation_reason"] = reason
        key = art.get("article_key", "")
        if key:
            v = ArticleVerification.query.filter_by(article_key=key).first()
            if v:
                art["verification"] = {
                    "status": v.status,
                    "confidence": v.confidence,
                    "analysis": v.analysis,
                    "cached": True,
                }
            else:
                art["verification"] = {
                    "status": "Checking",
                    "confidence": 0.0,
                    "analysis": "",
                    "cached": False,
                }

    return [art for _, art, _ in scored]


# ======================================================================
# REGIONAL NEWS GROUPING
# ======================================================================

def group_by_region(articles, user_region=None):
    """Group articles by their regional metadata.
    Returns dict of {region_level: [articles]}."""
    groups = {
        "local": [],
        "regional": [],
        "national": [],
        "international": [],
    }

    for art in articles:
        meta = art.get("_meta")
        if not meta:
            groups["national"].append(art)
            continue

        if meta.city and user_region and meta.city.lower() == user_region.lower():
            groups["local"].append(art)
        elif meta.state_region:
            groups["regional"].append(art)
        elif meta.country and meta.country.lower() == "india":
            groups["national"].append(art)
        else:
            groups["international"].append(art)

    return groups


def get_regional_news(user, all_live_articles, top_n=10):
    """Get news relevant to the user's region."""
    scored = []
    for art in all_live_articles:
        meta = extract_article_metadata(
            art.get("article_key", ""),
            title=art.get("title", ""),
            description=art.get("description", ""),
            source=art.get("source_name", ""),
        )
        art["_meta"] = meta
        reg = _regional_score(meta, user)
        if reg >= 0.2:
            scored.append((reg, art))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [art for _, art in scored[:top_n]]


def get_trending_news(all_live_articles, top_n=10):
    """Get fresh trending news (sorted by freshness)."""
    scored = []
    for art in all_live_articles:
        scored.append((_freshness_score(art), art))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [art for _, art in scored[:top_n]]


# ======================================================================
# PREFERENCES API HELPERS
# ======================================================================

def update_user_preferences(user, country=None, state_region=None, city=None, language=None):
    """Update user preference fields."""
    if country is not None:
        user.country = country
    if state_region is not None:
        user.state_region = state_region
    if city is not None:
        user.city = city
    if language is not None:
        user.preferred_language = language
    db.session.commit()


def set_user_interests(user_id, topics):
    """Set explicit user interests from a list of topic strings.
    These act as initial / override interests."""
    for topic in topics:
        topic = topic.strip().lower()
        if not topic:
            continue
        existing = UserInterest.query.filter_by(user_id=user_id, topic=topic).first()
        if existing:
            existing.score = max(existing.score, 0.6)
            existing.last_updated = datetime.utcnow()
        else:
            db.session.add(UserInterest(
                user_id=user_id, topic=topic,
                score=0.6, last_updated=datetime.utcnow(),
            ))
    db.session.commit()
