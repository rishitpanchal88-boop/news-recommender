"""
Root entrypoint for News Recommendation System.
Delegates to flask_app.app so the project can be run from the workspace root
(e.g., `python app.py` or `gunicorn app:app`) as well as from `flask_app/`.
"""

import os
import sys

# Ensure flask_app is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLASK_APP_DIR = os.path.join(BASE_DIR, "flask_app")

if FLASK_APP_DIR not in sys.path:
    sys.path.insert(0, FLASK_APP_DIR)

# Import the initialized Flask application
from flask_app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes", "on")
    app.run(host="0.0.0.0", port=port, debug=debug)
