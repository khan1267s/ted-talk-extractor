#!/usr/bin/env python3
"""
Startup script for TED Talk Speaker Extractor Web Application
Handles dependency installation and launches the Flask app.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_dependency(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None

def install_dependencies():
    """Install required dependencies."""
    print("🔍 Checking dependencies...")
    
    required_modules = [
        'flask', 'yt_dlp', 'moviepy', 'cv2', 'mediapipe', 
        'numpy', 'PIL', 'tqdm', 'requests'
    ]
    
    missing_modules = []
    for module in required_modules:
        if not check_dependency(module):
            missing_modules.append(module)
    
    if missing_modules:
        print(f"📦 Installing missing dependencies: {', '.join(missing_modules)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements_web.txt'
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("✅ All dependencies are already installed!")
    
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'static/clips',
        'downloads',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Directories created/verified")

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                     capture_output=True, check=True)
        print("✅ FFmpeg is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg not found. Please install FFmpeg:")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
        print("   Windows: Download from https://ffmpeg.org/")
        return False

def start_application():
    """Start the Flask application."""
    print("🚀 Starting TED Talk Speaker Extractor Web Application...")
    print("🌐 The application will be available at: http://localhost:5000")
    print("📱 Open your browser and navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main startup function."""
    print("🎬 TED Talk Speaker Extractor - Web Application")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: app.py not found in current directory")
        print("   Please run this script from the ted-talk-processor directory")
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Create directories
    create_directories()
    
    # Check FFmpeg
    if not check_ffmpeg():
        print("⚠️  Continuing without FFmpeg (some features may not work)")
    
    # Start the application
    start_application()

if __name__ == '__main__':
    main() 