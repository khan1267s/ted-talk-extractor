import os
import subprocess

# Get the port from the environment variable, default to 5000 if not found
port = os.environ.get("PORT", "5000")

# Construct the Gunicorn command
command = [
    "gunicorn",
    "--bind",
    f"0.0.0.0:{port}",
    "app:app",
]

# Run the Gunicorn command
try:
    subprocess.run(command, check=True)
except FileNotFoundError:
    print("Error: 'gunicorn' command not found.")
    print("Please make sure gunicorn is installed in your virtual environment.")
except subprocess.CalledProcessError as e:
    print(f"Gunicorn failed to start with error: {e}")

