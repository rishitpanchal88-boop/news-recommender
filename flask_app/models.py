"""
Database models.

We're using SQLite (a database that's just a single file -- db.sqlite3 --
no server to install or configure) and Flask-SQLAlchemy, which lets us
define tables as Python classes instead of writing raw SQL.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """A registered user of the site."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_premium = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Premium personalization fields (added via migration / db.create_all)
    country = db.Column(db.String(60), default="")
    state_region = db.Column(db.String(80), default="")
    city = db.Column(db.String(80), default="")
    preferred_language = db.Column(db.String(10), default="en")


class Interaction(db.Model):
    """One user action on one article: a like, dislike, or bookmark.

    Supports both dataset articles (article_id) and live articles (article_key).
    Stores metadata so saved bookmarks can be listed quickly without external fetches.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    article_id = db.Column(db.Integer, nullable=True, default=0)
    article_key = db.Column(db.String(256), nullable=True, index=True)
    action = db.Column(db.String(20), nullable=False, index=True)  # 'like', 'dislike', 'bookmark'
    title = db.Column(db.String(500), default="")
    source_name = db.Column(db.String(200), default="")
    url_to_image = db.Column(db.String(1000), default="")
    url = db.Column(db.String(1000), default="")
    category = db.Column(db.String(100), default="")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class RevenueLog(db.Model):
    """A simulated subscription payment, for the admin revenue dashboard."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    plan = db.Column(db.String(20), nullable=False)  # 'premium'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


# ----------------------------------------------------------------------
# PREMIUM PERSONALIZATION MODELS
# ----------------------------------------------------------------------

class UserInterest(db.Model):
    """Per-topic interest score for a user, learned from behaviour."""
    __tablename__ = "user_interest"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    topic = db.Column(db.String(100), nullable=False)   # e.g. "badminton", "ai", "cricket"
    score = db.Column(db.Float, default=0.0)             # 0.0 -- 1.0
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (
        db.UniqueConstraint("user_id", "topic", name="uq_user_interest"),
    )


class UserBehavior(db.Model):
    """Fine-grained reading behaviour beyond simple like/dislike."""
    __tablename__ = "user_behavior"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    article_key = db.Column(db.String(256), nullable=False)   # "live:<id>" or "ds:<id>"
    action = db.Column(db.String(30), nullable=False)         # click, read, save, skip
    duration_sec = db.Column(db.Float, default=0.0)
    topics = db.Column(db.String(500), default="")            # comma-separated
    region = db.Column(db.String(80), default="")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class ArticleMetadata(db.Model):
    """Cached geographic / language / topic metadata for articles."""
    __tablename__ = "article_metadata"
    id = db.Column(db.Integer, primary_key=True)
    article_key = db.Column(db.String(256), unique=True, nullable=False)
    country = db.Column(db.String(60), default="")
    state_region = db.Column(db.String(80), default="")
    city = db.Column(db.String(80), default="")
    language = db.Column(db.String(10), default="en")
    topics = db.Column(db.String(500), default="")           # comma-separated
    category = db.Column(db.String(100), default="")
    source_domain = db.Column(db.String(200), default="")
    extracted_at = db.Column(db.DateTime, default=datetime.utcnow)


class ArticleVerification(db.Model):
    """Cached fake-news verification result per article."""
    __tablename__ = "article_verification"
    id = db.Column(db.Integer, primary_key=True)
    article_key = db.Column(db.String(256), unique=True, nullable=False)
    status = db.Column(db.String(20), nullable=False)   # Real, Fake, Unverified, Checking
    confidence = db.Column(db.Float, default=0.0)
    analysis = db.Column(db.Text, default="")
    latency_ms = db.Column(db.Float, default=0.0)
    verified_at = db.Column(db.DateTime, default=datetime.utcnow)


class RegionalTrend(db.Model):
    """Snap-shot of what topics are trending in a region."""
    __tablename__ = "regional_trend"
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(80), nullable=False)       # "India", "Gujarat", "Ahmedabad"
    level = db.Column(db.String(20), nullable=False)        # country, state, city, global
    topic = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Float, default=0.0)
    article_count = db.Column(db.Integer, default=0)
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)


class LiveArticle(db.Model):
    """Durable store for every live / searched article the site has shown.

    The in-memory caches in news_api.py are volatile: they are emptied on
    process restart and the oldest half is evicted once the index passes
    MAX_INDEXED_LIVE_ARTICLES. Before this table existed, any live-article
    link that outlived its cache entry 302'd back to the home page with
    "This live article is no longer available."

    Rows here are the durable fallback -- /live-article/<live_id> rebuilds the
    card from this row when the memory cache misses. The TF-IDF vector is not
    stored (too large); it is recomputed from cleaned_text on demand, which is
    cheap since the vectorizer is already loaded.
    """
    __tablename__ = "live_article"
    id = db.Column(db.Integer, primary_key=True)
    live_id = db.Column(db.Integer, unique=True, nullable=False, index=True)
    title = db.Column(db.String(500), default="")
    description = db.Column(db.Text, default="")
    content = db.Column(db.Text, default="")
    url = db.Column(db.String(1000), default="")
    url_to_image = db.Column(db.String(1000), default="")
    source_name = db.Column(db.String(200), default="")
    author = db.Column(db.String(200), default="")
    published_at = db.Column(db.String(40), default="")
    category = db.Column(db.String(100), default="")
    cleaned_text = db.Column(db.Text, default="")
    source_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
