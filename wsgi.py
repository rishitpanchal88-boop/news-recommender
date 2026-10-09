import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLASK_APP_DIR = os.path.join(BASE_DIR, "flask_app")

if FLASK_APP_DIR not in sys.path:
    sys.path.insert(0, FLASK_APP_DIR)

from flask_app.app import app

if __name__ == "__main__":
    app.run()
