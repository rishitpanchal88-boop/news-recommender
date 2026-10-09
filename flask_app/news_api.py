"""
news_api.py — Multi-Source Live News Integration Module

Fetches live headlines and search results from three independent sources,
merges them, and processes them through the existing recommendation pipeline:

  1. NewsAPI.org  — top-headlines (feeds) + everything (search); needs API key
  2. Google News RSS — free, no key; topic feeds (feeds) + search endpoint
  3. GDELT DOC 2.0   — free, no key; keyword queries for both modes

Results are normalised to a common dict shape, deduplicated by URL and
title fingerprint, and round-robin interleaved so no single source
dominates the feed. The merged stream is then processed through the
existing TF-IDF + fake-news detection pipeline exactly as before.

A live-to-live recommendation function (get_similar_live_articles) lets
users chain from any live article to other similar live articles, enabling
infinite browsing through breaking-news recommendations.

Public function signatures are unchanged so app.py and every template
continues to work without modification.
"""

import re
import time
import hashlib
import datetime
import email.utils
import threading
import concurrent.futures
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import quote_plus

import requests
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
NEWS_API_BASE = "https://newsapi.org/v2"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_RSS_BASE = "https://news.google.com/rss"

RSS_USER_AGENT = "Mozilla/5.0 (compatible; NewsRecommender/1.0)"
RSS_TIMEOUT = 5             # seconds
NEWSAPI_TIMEOUT = 10        # seconds
GDELT_TIMEOUT = 10          # seconds

CACHE_TTL_SECONDS = 30 * 60          # feed cache (30 min)
EVERYTHING_TTL_SECONDS = 15 * 60     # search cache (15 min)
FEED_TIMESPAN = "6h"

DEFAULT_COUNTRY = "us"
DEFAULT_CATEGORY = "general"
DEFAULT_PAGE_SIZE = 6

# ---------------------------------------------------------------------------
# TRUSTED SOURCES (display names, lowercased before comparison)
# ---------------------------------------------------------------------------
TRUSTED_SOURCES = {
    "reuters", "associated press", "bbc news", "cnn", "the new york times",
    "the washington post", "the guardian", "al jazeera english", "bloomberg",
    "the wall street journal", "nbc news", "abc news", "cbs news", "npr",
    "usa today", "time", "the economist", "politico", "axios", "ap news",
    "afp", "france 24", "deutsche welle", "sky news", "the hindu",
    "times of india", "ndtv", "hindustan times", "india today",
    "the indian express", "mint", "economic times",
}

# ---------------------------------------------------------------------------
# GDELT-SPECIFIC MAPPINGS
# ---------------------------------------------------------------------------
DOMAIN_TO_SOURCE = {
    "apnews.com": "AP News",
    "ap.org": "Associated Press",
    "afp.com": "AFP",
    "reuters.com": "Reuters",
    "bbc.com": "BBC News",
    "bbc.co.uk": "BBC News",
    "cnn.com": "CNN",
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "theguardian.com": "The Guardian",
    "aljazeera.com": "Al Jazeera English",
    "bloomberg.com": "Bloomberg",
    "wsj.com": "The Wall Street Journal",
    "nbcnews.com": "NBC News",
    "abcnews.go.com": "ABC News",
    "cbsnews.com": "CBS News",
    "npr.org": "NPR",
    "usatoday.com": "USA Today",
    "time.com": "Time",
    "economist.com": "The Economist",
    "politico.com": "Politico",
    "axios.com": "Axios",
    "cnbc.com": "CNBC",
    "france24.com": "France 24",
    "dw.com": "Deutsche Welle",
    "news.sky.com": "Sky News",
    "thehindu.com": "The Hindu",
    "timesofindia.indiatimes.com": "Times of India",
    "ndtv.com": "NDTV",
    "hindustantimes.com": "Hindustan Times",
    "indiatoday.in": "India Today",
    "indianexpress.com": "The Indian Express",
    "livemint.com": "Mint",
    "economictimes.indiatimes.com": "Economic Times",
}

GDELT_CATEGORY_QUERIES = {
    "sports": "(sports OR cricket OR soccer OR football OR tennis OR basketball OR Olympics) sourcelang:english",
    "business": "(economy OR finance OR markets OR business OR stocks OR GDP OR trade) sourcelang:english",
    "entertainment": "(entertainment OR movies OR cinema OR celebrity OR Bollywood OR Hollywood OR film) sourcelang:english",
    "technology": "(technology OR artificial intelligence OR AI OR gadgets OR software OR cybersecurity OR startup) sourcelang:english",
    "health": "(health OR medicine OR disease OR hospital OR vaccine OR WHO OR mental health) sourcelang:english",
    "science": "(science OR research OR space OR NASA OR climate OR environment OR study) sourcelang:english",
    "education": "(education OR school OR university OR college OR students OR learning OR curriculum) sourcelang:english",
    "politics": "(politics OR government OR election OR parliament OR president OR minister OR policy) sourcelang:english",
    "travel": "(travel OR tourism OR destination OR flights OR hotel OR visa OR vacation) sourcelang:english",
    "food": "(food OR cuisine OR recipe OR restaurant OR diet OR nutrition OR cooking) sourcelang:english",
    "general": "(top OR breaking OR world OR major) sourcelang:english",
}

# ---------------------------------------------------------------------------
# GOOGLE RSS TOPIC MAP
# ---------------------------------------------------------------------------
# Google News RSS topic feeds use uppercase identifiers; None = top feed.
_RSS_TOPIC_MAP = {
    "general": None,
    "sports": "SPORTS",
    "business": "BUSINESS",
    "entertainment": "ENTERTAINMENT",
    "technology": "TECHNOLOGY",
    "health": "HEALTH",
    "science": "SCIENCE",
    "education": None,
    "politics": None,
    "travel": None,
    "food": None,
    "world": "WORLD",
    "nation": "NATION",
}

_MEDIA_NS = {"media": "http://search.yahoo.com/mrss/"}

# ---------------------------------------------------------------------------
# IN-MEMORY CACHES
# ---------------------------------------------------------------------------
_feed_cache = {"general": {"articles": [], "timestamp": 0}}
_everything_cache = {}

MAX_INDEXED_LIVE_ARTICLES = 1000
_live_article_index = {}

# How many live articles to keep durably in the live_article table. Oldest
# rows are pruned past this so the SQLite file cannot grow without bound.
MAX_SAVED_LIVE_ARTICLES = 20000

# ---------------------------------------------------------------------------
# DURABLE PERSISTENCE (SQLite)
# ---------------------------------------------------------------------------
# news_api.py deliberately knows nothing about Flask, so these helpers import
# models lazily -- the same pattern premium.py::_source_score uses. Every
# function degrades to a no-op if there is no active database context, which
# keeps this module usable from plain scripts (test_recs.py, notebooks).


def save_article(article):
    """Upsert one processed live article into the live_article table.

    Called for every article the site displays (feeds and search hits), so a
    live-article link always has a durable row to fall back on. Never raises.
    """
    live_id = article.get("live_id")
    if live_id is None:
        return
    try:
        from models import db, LiveArticle

        row = LiveArticle.query.filter_by(live_id=live_id).first()
        if row is None:
            row = LiveArticle(live_id=live_id)
            db.session.add(row)
        row.title = (article.get("title") or "")[:500]
        row.description = article.get("description") or ""
        row.content = article.get("content") or ""
        row.url = (article.get("url") or "")[:1000]
        row.url_to_image = (article.get("url_to_image") or "")[:1000]
        row.source_name = (article.get("source_name") or "")[:200]
        row.author = (article.get("author") or "")[:200]
        row.published_at = (article.get("published_at") or "")[:40]
        row.category = (article.get("category") or "")[:100]
        row.cleaned_text = article.get("cleaned_text") or ""
        row.source_verified = bool(article.get("source_verified"))
        db.session.commit()
    except Exception:
        _rollback_quietly()


def _rollback_quietly():
    try:
        from models import db
        db.session.rollback()
    except Exception:
        pass


def load_article(live_id):
    """Rebuild a processed-article dict from the live_article table.

    Returns None if the id was never persisted. The returned dict is the
    display shape the templates expect; caller is responsible for adding
    _rec_vec / similar_from_dataset (app.py does that, since it owns the
    TF-IDF objects). Never raises.
    """
    try:
        from models import LiveArticle

        row = LiveArticle.query.filter_by(live_id=live_id).first()
        if row is None:
            return None
        return {
            "live_id": row.live_id,
            "article_id": "live_{}".format(row.live_id),
            "title": row.title or "",
            "description": row.description or "",
            "content": row.content or "",
            "url": row.url or "",
            "url_to_image": row.url_to_image or "",
            "source_name": row.source_name or "",
            "author": row.author or "",
            "published_at": row.published_at or "",
            "category": row.category or "Live News",
            "cleaned_text": row.cleaned_text or "",
            "source_verified": bool(row.source_verified),
            "is_live": True,
            "_rec_vec": None,
            "similar_from_dataset": [],
        }
    except Exception:
        _rollback_quietly()
        return None


def prune_saved_articles(max_rows=MAX_SAVED_LIVE_ARTICLES):
    """Drop the oldest durable rows once the table exceeds max_rows.

    Returns the number of rows deleted. Never raises.
    """
    try:
        from models import db, LiveArticle

        total = LiveArticle.query.count()
        if total <= max_rows:
            return 0
        # The id of the newest row we intend to KEEP; delete everything older.
        keep_from = (
            db.session.query(LiveArticle.id)
            .order_by(LiveArticle.id.desc())
            .offset(max_rows - 1)
            .first()
        )
        if keep_from is None:
            return 0
        deleted = LiveArticle.query.filter(LiveArticle.id < keep_from[0]).delete()
        db.session.commit()
        print("  [live_article] Pruned %d old rows (kept %d)" % (deleted, max_rows))
        return deleted
    except Exception:
        _rollback_quietly()
        return 0

# ---------------------------------------------------------------------------
# GDELT GLOBAL THROTTLE
# ---------------------------------------------------------------------------
_last_gdelt_call = 0.0
_gdelt_fail_count = 0
_gdelt_disabled = False
_GDELT_MIN_INTERVAL = 2.0


def _throttle_gdelt():
    global _last_gdelt_call
    wait = _GDELT_MIN_INTERVAL - (time.time() - _last_gdelt_call)
    if wait > 0:
        time.sleep(wait)


def _mark_gdelt_call():
    global _last_gdelt_call
    _last_gdelt_call = time.time()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _generate_live_id(url):
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _sanitize_query(query):
    """Lowercase + strip GDELT-reserved characters so user text is safe."""
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", query or "")]
    return " ".join(words[:12])[:180]


def _title_fingerprint(title):
    words = re.findall(r"[A-Za-z0-9]+", (title or "").lower())
    return " ".join(words[:6])


def _parse_seen_date(value):
    """GDELT seendate '20260824T191500Z' → ISO 8601."""
    try:
        dt = datetime.datetime.strptime((value or "").strip(), "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ""


def _parse_rfc822(date_str):
    """RFC 822 (e.g. 'Mon, 25 Aug 2025 12:00:00 GMT') → ISO 8601."""
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _strip_html(html):
    """Crude HTML tag stripper for RSS descriptions."""
    return re.sub(r"<[^>]+>", "", html or "")[:300].strip()


def _clean_domain(domain):
    d = (domain or "").lower().strip()
    if d.startswith("www."):
        d = d[4:]
    return d


def _source_name(domain):
    """Map a GDELT domain to the display name the rest of the app expects."""
    d = _clean_domain(domain)
    if not d:
        return "Unknown"
    return DOMAIN_TO_SOURCE.get(d, d)


# ---------------------------------------------------------------------------
# MERGE + DEDUPE + INTERLEAVE
# ---------------------------------------------------------------------------
def _merge_and_interleave(lists, limit):
    """Round-robin interleave several article lists, deduplicate by URL and
    title fingerprint, and return up to *limit* items."""
    deduped = []
    seen_urls = set()
    seen_fps = set()

    max_len = max((len(lst) for lst in lists), default=0)
    for i in range(max_len):
        for lst in lists:
            if i >= len(lst):
                continue
            item = lst[i]

            url = (item.get("url") or item.get("link") or "").strip()
            fp = _title_fingerprint(item.get("title") or "")

            # Dedupe by normalised URL (strip query/fragment)
            url_key = re.sub(r"[?#].*", "", url.lower()) if url else ""
            if url_key and url_key in seen_urls:
                continue
            # Dedupe by title fingerprint (skip short strings)
            if fp and len(fp) > 12 and fp in seen_fps:
                continue

            if url_key:
                seen_urls.add(url_key)
            if fp and len(fp) > 12:
                seen_fps.add(fp)
            deduped.append(item)

            if len(deduped) >= limit:
                return deduped

    return deduped


# ---------------------------------------------------------------------------
# NEWSAPI FETCHERS
# ---------------------------------------------------------------------------
def _fetch_newsapi_headlines(api_key, category, country, page_size):
    """NewsAPI /v2/top-headlines (key required). Returns legacy dicts."""
    if not api_key:
        return []
    try:
        params = {
            "apiKey": api_key,
            "country": country or DEFAULT_COUNTRY,
            "pageSize": min(page_size, 100),
        }
        cat = (category or "").lower()
        if cat and cat != "general":
            params["category"] = cat
        resp = requests.get(
            f"{NEWS_API_BASE}/top-headlines",
            params=params,
            timeout=(5, NEWSAPI_TIMEOUT),
        )
        if resp.status_code in (429, 402, 426):
            print(f"  [NewsAPI] Quota/rate limit ({resp.status_code}); skipping")
            return []
        resp.raise_for_status()
        return resp.json().get("articles") or []
    except Exception as e:
        print(f"  [NewsAPI] Headlines error: {e}")
        return []


def _fetch_newsapi_search(api_key, query, page_size):
    """NewsAPI /v2/everything (key required). Returns legacy dicts."""
    if not api_key or not query:
        return []
    try:
        resp = requests.get(
            f"{NEWS_API_BASE}/everything",
            params={
                "apiKey": api_key,
                "q": query,
                "pageSize": min(page_size, 100),
                "sortBy": "publishedAt",
                "language": "en",
            },
            timeout=(5, NEWSAPI_TIMEOUT),
        )
        if resp.status_code in (429, 402, 426):
            print(f"  [NewsAPI] Quota/rate limit ({resp.status_code}); skipping")
            return []
        resp.raise_for_status()
        return resp.json().get("articles") or []
    except Exception as e:
        print(f"  [NewsAPI] Search error: {e}")
        return []


# ---------------------------------------------------------------------------
# GOOGLE NEWS RSS FETCHERS
# ---------------------------------------------------------------------------
def _rss_to_legacy(item):
    """Convert one <item> XML element to a legacy-shaped dict."""
    title = (item.findtext("title") or "").strip()
    link = (item.findtext("link") or "").strip()
    pub_date = _parse_rfc822(item.findtext("pubDate", ""))
    source_el = item.find("source")
    source_name = ""
    if source_el is not None and source_el.text:
        source_name = source_el.text.strip()

    # Image: prefer media:thumbnail > media:content > enclosure
    image_url = ""
    media = (
        item.find("media:thumbnail", _MEDIA_NS)
        or item.find("media:content", _MEDIA_NS)
    )
    if media is not None:
        image_url = media.get("url", "")
    if not image_url:
        enc = item.find("enclosure")
        if enc is not None and (enc.get("type") or "").startswith("image"):
            image_url = enc.get("url", "")

    desc = _strip_html(item.findtext("description", ""))

    return {
        "title": title,
        "description": desc,
        "content": "",
        "url": link,
        "urlToImage": image_url,
        "publishedAt": pub_date,
        "author": "",
        "source": {"name": source_name or "Unknown"},
    }


def _rss_fetch_items(url):
    """Fetch an RSS feed and return the raw XML <item> elements."""
    req = Request(url, headers={"User-Agent": RSS_USER_AGENT})
    with urlopen(req, timeout=RSS_TIMEOUT) as resp:
        xml_data = resp.read()
    return ET.fromstring(xml_data).findall(".//item")


def _fetch_google_rss_headlines(category, page_size):
    """Google News RSS topic feed. Falls back to keyword search for
    categories without a direct RSS topic (education, politics, etc)."""
    try:
        cat = (category or "general").lower()
        topic = _RSS_TOPIC_MAP.get(cat)
        if topic:
            url = (
                f"{GOOGLE_RSS_BASE}/headlines/section/topic/{topic}"
                f"?hl=en-US&gl=US&ceid=US:en"
            )
        else:
            # No direct topic — use a precise keyword search instead
            _cat_queries = {
                "education": "education school university students learning",
                "politics": "politics government election parliament policy",
                "travel": "travel tourism destination vacation flights",
                "food": "food cuisine recipe restaurant nutrition cooking",
                "world": "world news international",
                "nation": "national news",
            }
            query = _cat_queries.get(cat, f"{cat} news")
            url = (
                f"{GOOGLE_RSS_BASE}/search"
                f"?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            )
        items = _rss_fetch_items(url)
        return [_rss_to_legacy(it) for it in items[:page_size]]
    except Exception as e:
        print(f"  [GoogleRSS] Headlines error: {e}")
        return []


def _fetch_google_rss_search(query, page_size):
    """Google News RSS search endpoint. Returns legacy dicts."""
    try:
        clean = _sanitize_query(query)
        if not clean:
            return []
        url = (
            f"{GOOGLE_RSS_BASE}/search"
            f"?q={quote_plus(clean)}&hl=en-US&gl=US&ceid=US:en"
        )
        items = _rss_fetch_items(url)
        return [_rss_to_legacy(it) for it in items[:page_size]]
    except Exception as e:
        print(f"  [GoogleRSS] Search error: {e}")
        return []


# ---------------------------------------------------------------------------
# GDELT FETCHER
# ---------------------------------------------------------------------------
def _gdelt_to_legacy(article):
    """Normalise one GDELT article dict to the legacy shape."""
    return {
        "title": (article.get("title") or "").strip(),
        "description": "",
        "content": "",
        "url": article.get("url") or "",
        "urlToImage": article.get("socialimage") or "",
        "publishedAt": _parse_seen_date(article.get("seendate")),
        "author": "",
        "source": {"name": _source_name(article.get("domain"))},
    }


def _gdelt_fetch(query, max_records, timespan=FEED_TIMESPAN):
    """GDELT DOC 2.0 ArtList call. Runs in a separate thread with
    a hard 5s deadline so it can never block the pipeline."""
    global _gdelt_fail_count, _gdelt_disabled
    if _gdelt_disabled:
        return []

    _throttle_gdelt()

    def _do_fetch():
        resp = requests.get(
            GDELT_DOC_URL,
            params={
                "query": query,
                "mode": "artlist",
                "maxrecords": min(max(max_records, 1), 250),
                "sort": "datedesc",
                "timespan": timespan,
                "format": "json",
            },
            timeout=(3, 5),
        )
        _mark_gdelt_call()
        if resp.status_code == 429:
            return None  # signal 429
        resp.raise_for_status()
        return resp.json()

    result = [None]
    error = [None]

    def _worker():
        try:
            result[0] = _do_fetch()
        except Exception as e:
            _mark_gdelt_call()
            error[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=5.0)

    if t.is_alive():
        _mark_gdelt_call()
        _gdelt_fail_count += 1
        if _gdelt_fail_count >= 2:
            _gdelt_disabled = True
            print("  [GDELT] Disabled after %d failures" % _gdelt_fail_count)
        else:
            print("  [GDELT] Timed out after 5s (attempt %d)" % _gdelt_fail_count)
        return []

    if error[0] is not None:
        _gdelt_fail_count += 1
        if _gdelt_fail_count >= 2:
            _gdelt_disabled = True
            print("  [GDELT] Disabled after %d failures" % _gdelt_fail_count)
        else:
            print("  [GDELT] Failed (attempt %d): %s" % (_gdelt_fail_count, str(error[0])[:60]))
        return []

    if result[0] is None:
        _gdelt_fail_count += 1
        if _gdelt_fail_count >= 2:
            _gdelt_disabled = True
        return []

    _gdelt_fail_count = 0
    data = result[0]
    normalized = []
    seen_urls = set()
    for a in data.get("articles") or []:
        legacy = _gdelt_to_legacy(a)
        if not legacy["title"] or not legacy["url"]:
            continue
        if legacy["url"] in seen_urls:
            continue
        seen_urls.add(legacy["url"])
        normalized.append(legacy)
    return normalized


# ---------------------------------------------------------------------------
# PUBLIC API — HEADLINES (with caching)
# ---------------------------------------------------------------------------
def fetch_live_headlines(api_key=None, country=DEFAULT_COUNTRY,
                         category=DEFAULT_CATEGORY,
                         page_size=DEFAULT_PAGE_SIZE):
    """Return normalised article dicts from all three sources, merged
    and interleaved. Fetches all sources CONCURRENTLY so one slow source
    doesn't block the others. Returns [] on total failure."""
    cat = (category or DEFAULT_CATEGORY).lower()

    def _safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_newsapi = pool.submit(_safe, _fetch_newsapi_headlines, api_key, cat, country, page_size)
        f_rss = pool.submit(_safe, _fetch_google_rss_headlines, cat, page_size)
        f_gdelt = pool.submit(_safe, _gdelt_fetch,
                              GDELT_CATEGORY_QUERIES.get(cat, f"{cat} sourcelang:english"),
                              page_size)
        lists = [f_newsapi.result(), f_rss.result(), f_gdelt.result()]

    return _merge_and_interleave(lists, limit=page_size)


# ---------------------------------------------------------------------------
# PUBLIC API — SEARCH (with caching)
# ---------------------------------------------------------------------------
def search_everything(api_key, query, page_size=5, raw=False):
    """Keyword search across all three sources.  When raw=True returns
    the normalised legacy dicts (for process_articles()); otherwise
    returns simplified dicts (title/link/date/source/description) for
    direct display."""
    if not query or not query.strip():
        return []

    clean_q = _sanitize_query(query)
    if not clean_q:
        return []

    cache_key = clean_q + ("|raw" if raw else "")
    now = time.time()
    cached = _everything_cache.get(cache_key)
    if cached and (now - cached[0]) < EVERYTHING_TTL_SECONDS:
        return cached[1]

    def _safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        f_newsapi = pool.submit(_safe, _fetch_newsapi_search, api_key, clean_q, page_size)
        f_rss = pool.submit(_safe, _fetch_google_rss_search, clean_q, page_size)
        f_rss2 = pool.submit(_safe, _fetch_google_rss_search, f"{clean_q} latest", page_size)
        f_gdelt = pool.submit(_safe, _gdelt_fetch, f"{clean_q} sourcelang:english", page_size)
        rss_lists = [f_newsapi.result(), f_rss.result(), f_rss2.result(), f_gdelt.result()]

    if raw:
        results = _merge_and_interleave(rss_lists, limit=page_size)
    else:
        results = [
            {
                "title": a["title"],
                "link": a["url"],
                "date": (a["publishedAt"] or "")[:10],
                "source": a["source"]["name"],
                "description": a["description"],
                "image": a.get("urlToImage") or "",
            }
            for a in _merge_and_interleave(rss_lists, limit=page_size)
        ]

    _everything_cache[cache_key] = (now, results)
    print(f"  [Multi-source] Cached {len(results)} search results for: {clean_q[:60]}")
    return results


# ---------------------------------------------------------------------------
# PIPELINE — process through TF-IDF + fake-news model (unchanged logic)
# ---------------------------------------------------------------------------
def process_articles(raw_articles, clean_text_fn, rec_tfidf, rec_tfidf_matrix,
                     rec_tfidf_matrix_t, all_article_ids, get_article_fn,
                     category=DEFAULT_CATEGORY):
    """Transform raw legacy dicts into processed dicts that match the
    existing dataset schema and include TF-IDF recommendations from the
    dataset.  Also stores the rec vector for live→live chaining later."""
    processed = []

    for raw in raw_articles:
        title = raw.get("title") or ""
        description = raw.get("description") or ""
        content = raw.get("content") or ""
        url = raw.get("url") or ""
        url_to_image = raw.get("urlToImage") or ""
        source_name = (raw.get("source") or {}).get("name", "Unknown")
        author = raw.get("author") or ""
        published_at = raw.get("publishedAt") or ""

        if title == "[Removed]" or not title.strip():
            continue

        # --- Build raw text ---
        raw_text = (content or description or title)[:3000]
        if len(raw_text.strip()) < 20:
            raw_text = f"{title} {description}"

        # --- Clean text ---
        cleaned = clean_text_fn(raw_text)
        if not cleaned or len(cleaned.split()) < 3:
            cleaned = clean_text_fn(f"{title} {description}")

        # --- Find similar dataset articles ---
        similar_from_dataset = []
        live_vec = None
        if cleaned:
            try:
                query_vec = rec_tfidf.transform([cleaned]).astype(np.float32)
                live_vec = query_vec
                sims = (query_vec @ rec_tfidf_matrix_t).toarray().flatten()
                top_idx = np.argsort(-sims)[:5]
                for i in top_idx:
                    if sims[i] <= 0:
                        continue
                    aid = all_article_ids[i]
                    article = get_article_fn(aid)
                    if article:
                        article["similarity_score"] = round(float(sims[i]), 4)
                        similar_from_dataset.append(article)
            except Exception:
                pass

        live_id = _generate_live_id(url)
        article = {
            "live_id": live_id,
            "article_id": f"live_{live_id}",
            "source_name": source_name,
            "author": author,
            "title": title,
            "description": description,
            "url": url,
            "url_to_image": url_to_image,
            "published_at": published_at,
            "content": content,
            "category": (category or DEFAULT_CATEGORY).lower(),
            "display_category": "Live News",
            "cleaned_text": cleaned,
            "source_verified": source_name.lower().strip() in TRUSTED_SOURCES,
            "is_live": True,
            "_rec_vec": live_vec,
            "similar_from_dataset": similar_from_dataset,
        }
        _live_article_index[live_id] = article
        processed.append(article)
        # Durable copy: the memory index below is volatile, so persist every
        # article we surface. This is what keeps a live-article link openable
        # after eviction or a process restart.
        save_article(article)

    if len(_live_article_index) >= MAX_INDEXED_LIVE_ARTICLES:
        # Evict oldest half instead of clearing everything
        sorted_ids = sorted(_live_article_index.keys(),
                            key=lambda k: _live_article_index[k].get("published_at", ""))
        for k in sorted_ids[:len(sorted_ids) // 2]:
            _live_article_index.pop(k, None)

    prune_saved_articles()
    return processed


# ---------------------------------------------------------------------------
# PUBLIC API — CACHED LIVE FEED
# ---------------------------------------------------------------------------
def _cache_for(category):
    key = (category or DEFAULT_CATEGORY).lower()
    if key not in _feed_cache:
        _feed_cache[key] = {"articles": [], "timestamp": 0}
    return _feed_cache[key]


def get_live_news(api_key, count, clean_text_fn, rec_tfidf, rec_tfidf_matrix,
                  rec_tfidf_matrix_t, all_article_ids, get_article_fn,
                  category=DEFAULT_CATEGORY, country=DEFAULT_COUNTRY):
    """Return cached or freshly-fetched live news.
    Each category caches independently for CACHE_TTL_SECONDS.
    Runs process_articles() to compute TF-IDF vectors (_rec_vec) and
    similar_from_dataset — needed for recommendation chaining."""
    bucket = _cache_for(category)
    now = time.time()

    if bucket["articles"] and (now - bucket["timestamp"]) < CACHE_TTL_SECONDS:
        return bucket["articles"][:count]

    raw = fetch_live_headlines(api_key, category=category, country=country,
                               page_size=max(count, 10))
    if not raw:
        return bucket["articles"][:count]

    processed = process_articles(
        raw, clean_text_fn, rec_tfidf, rec_tfidf_matrix,
        rec_tfidf_matrix_t, all_article_ids, get_article_fn,
        category=category,
    )

    bucket["articles"] = processed
    bucket["timestamp"] = now
    print("  [Multi-source] Cached %d live articles (category: %s)" % (len(processed), category))

    return processed[:count]


def get_cached_live_article(live_id):
    """Look up a single live article by its live_id.

    Resolution order:
      1. every category feed cache
      2. the all-articles search index
      3. the durable live_article table (SQLite)

    Step 3 is what stops /live-article/<live_id> from bouncing to the home
    page with "This live article is no longer available." once the volatile
    caches are gone (process restart, or eviction past
    MAX_INDEXED_LIVE_ARTICLES). A row rebuilt from the DB has no _rec_vec and
    no similar_from_dataset; app.py rehydrates those on the way out.
    """
    for bucket in _feed_cache.values():
        for article in bucket["articles"]:
            if article["live_id"] == live_id:
                return article
    hit = _live_article_index.get(live_id)
    if hit is not None:
        return hit
    # Durable fallback.
    return load_article(live_id)


def get_all_live_articles(category=None):
    """Return every live article currently cached for a category."""
    return _cache_for(category)["articles"]


# ---------------------------------------------------------------------------
# LIVE → LIVE RECOMMENDATIONS (chaining)
# ---------------------------------------------------------------------------
def get_similar_live_articles(article, top_n=5):
    """Find other cached live articles most similar to *article* using the
    pre-computed TF-IDF recommendation vectors stored by process_articles().

    This enables live-to-live recommendation chaining: clicking a live
    article shows similar live stories, and clicking through those shows
    *their* similar stories, and so on — infinite browsable depth."""
    vec = article.get("_rec_vec")
    if vec is None:
        return []

    seen_id = article.get("live_id")
    scored = []

    # Collect every processed live article (category caches + search index)
    candidates = {}
    for bucket in _feed_cache.values():
        for a in bucket["articles"]:
            candidates[a["live_id"]] = a
    for lid, a in _live_article_index.items():
        candidates[lid] = a

    for lid, other in candidates.items():
        if lid == seen_id:
            continue
        other_vec = other.get("_rec_vec")
        if other_vec is None:
            continue
        try:
            score = float((vec @ other_vec.T).toarray()[0, 0])
        except Exception:
            continue
        if score <= 0:
            continue
        other["similarity_score"] = round(score, 4)
        scored.append(other)

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_n]
