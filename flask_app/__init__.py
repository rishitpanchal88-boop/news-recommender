"""News Recommender Flask application package.

The empty __init__.py makes `flask_app` a regular package so
`gunicorn flask_app.app:app` (Procfile / railway.json) imports
reliably on every Python 3 version, not just via namespace
packages.
"""
