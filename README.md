# 📰 News Recommendation & Fact-Checking System

A full-stack, production-ready News Recommendation and AI Fact-Checking Web Application built with **Flask**, **scikit-learn**, **NewsAPI / Google News RSS / GDELT**, and **Google Gemini AI**.

---

## ✨ Features & Quality-of-Life (QoL) Highlights

- **🧠 Content-Based & Collaborative Recommendation Engine**:
  - TF-IDF vectorization with cosine similarity and fast sparse matrix dot-products.
  - Live-to-Live news chaining and Dataset-to-Live cross-recommendations.
- **🛡️ AI Fact & Fake News Checker**:
  - Multi-tier fact-checking: Semantic consistency, source credibility verification, and Google Gemini AI cross-validation against live global web reporting.
  - 1-Click sample claims to test immediately.
  - Interactive Confidence Gauge, step-by-step verification breakdown, and 1-click clipboard report sharing.
- **🔖 Personal Reading List & 1-Click Bookmarks**:
  - Bookmark any live or dataset article instantly with AJAX and animated toast notifications.
  - Dedicated `/bookmarks` reading list page with instant title/source search filter.
- **📖 Enhanced Reader Experience**:
  - Adjustable reading font size (`A-`, `A`, `A+`).
  - Estimated reading time (`⏱️ X min read`).
  - Native Web Share API + 1-click copy article link.
  - Built-in Text-to-Speech (TTS) audio reader with voice selection and playback speed controls (`0.75x` – `1.5x`).
- **🌐 Multi-Language Translation**:
  - Live on-the-fly article translation across English, Hindi, Gujarati, Marathi, Bengali, Tamil, Telugu, and Malayalam.
- **⭐ Premium Personalised Experience**:
  - Hyper-local regional news feed tailored to user country, state, and city.
  - Dynamic user topic interest learning from reading behaviour.
- **📊 Admin Analytics Dashboard**:
  - Overview of users, interactions, sentiment ratios (likes vs. dislikes), and premium conversion rates.
- **🚀 Deployable Anywhere**:
  - Memory-optimized dataset loading (<60MB RAM footprint).
  - Built-in graceful starter dataset fallback for zero-downtime deployment.
  - Ready-to-go `Dockerfile`, `Procfile`, `render.yaml`, and `railway.json`.
  - Health check endpoint `/health` for cloud uptime monitors.

---

## 🛠️ Quick Local Setup

### 1. Clone & Navigate
```bash
git clone <your-repo-url>
cd DATA-SCI
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r flask_app/requirements.txt
```

### 4. Configure Environment (Optional)
```bash
# Windows
copy flask_app\.env.example flask_app\.env

# macOS / Linux
cp flask_app/.env.example flask_app/.env
```
*Fill in `NEWS_API_KEY` or `GEMINI_API_KEY` if available. If left empty, the app automatically uses Google News RSS and deterministic web coverage verification.*

### 5. Run the Application
You can run from the root directory or inside `flask_app/`:
```bash
python app.py
# Or:
# python flask_app/app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## ☁️ 1-Click Cloud Deployment

### Option A: Render (Recommended)
1. In the **Render Dashboard**, click **New +** -> **Blueprint**.
2. Connect your Git repository. Render will detect `render.yaml`.
3. Click **Apply**. Render will automatically provision the PostgreSQL database and web service with health checks configured at `/health`.

### Option B: Railway
1. Click **New Project** -> **Deploy from GitHub repo**.
2. Railway detects `railway.json` and `Procfile` automatically.
3. Set optional environment variables (`SECRET_KEY`, `GEMINI_API_KEY`, `NEWS_API_KEY`).

### Option C: Docker / Fly.io / GCP Cloud Run
```bash
# Build Docker image
docker build -t news-recommender .

# Run container
docker run -p 5000:5000 -e SECRET_KEY="your-secret-key" news-recommender
```

---

## 🔍 API & Health Endpoints

- `GET /health` or `GET /api/health` -> System health check for load balancers.
- `POST /bookmark` -> AJAX toggle bookmark on any story (`ds:<id>` or `live:<id>`).
- `POST /verify-article` -> Async claim verification with fast-path trusted source validation.
- `POST /premium/track` -> Background reading behaviour tracking.

---

## 📄 License
MIT License. Built for news exploration and fact-checking research.
