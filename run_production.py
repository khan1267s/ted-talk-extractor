from waitress import serve
from app import app
import os

if __name__ == "__main__":
    # Get port from environment variable or default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting production server on http://0.0.0.0:{port} ...")
    serve(app, host="0.0.0.0", port=port)
