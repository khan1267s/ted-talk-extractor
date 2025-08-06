#!/usr/bin/env python3
"""
Main entry point for Google App Engine deployment.
"""

from app import app

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app.
    app.run(host='127.0.0.1', port=8080, debug=False)