#!/usr/bin/env python3
"""
TED Talk Speaker Extractor - Web Interface
"""

import os
import sys
from pathlib import Path
import logging
import time
import threading

# Web framework
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Custom processor
from ted_processor import TEDTalkProcessor, MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Define absolute paths
APP_ROOT = Path(__file__).parent.resolve()
DOWNLOADS_DIR = APP_ROOT / "downloads"
CLIPS_DIR = APP_ROOT / "static" / "clips"

# Create directories
DOWNLOADS_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True, parents=True)

# Global variables
jobs = {}
job_counter = 0

# Initialize processor
try:
    processor = TEDTalkProcessor(output_dir=str(CLIPS_DIR), model_path=str(MODEL_PATH))
except FileNotFoundError as e:
    logger.error(f"Could not initialize TEDTalkProcessor: {e}")
    # Exit if the model is essential for the app to run
    sys.exit(1)

def process_job(job_id, url, max_clips):
    """Process a job in a separate thread with progress updates."""
    global jobs
    
    def progress_callback(progress):
        # Update progress for downloading and processing
        # Let's say downloading is 20%, analysis is 80%
        base_progress = 20
        analysis_progress = int(progress * 0.7) # 70% of the total time
        jobs[job_id]['progress'] = base_progress + analysis_progress

    try:
        jobs[job_id].update({'status': 'processing', 'progress': 5, 'message': 'Downloading video...'})
        
        # This is a simplified progress update. For real-world apps, yt-dlp provides hooks.
        # For now, we'll just jump to a certain percentage after download.
        actual_video_path = processor.download_video(url)
        if not actual_video_path:
            jobs[job_id].update({'status': 'failed', 'message': 'Failed to download video.'})
            return

        jobs[job_id].update({'status': 'processing', 'progress': 20, 'message': 'Analyzing video for speaker segments...'})
        
        segments = processor.find_speaker_segments(actual_video_path, progress_callback=progress_callback)
        
        if not segments:
            jobs[job_id].update({'status': 'failed', 'message': 'No speaker segments found.'})
            os.remove(actual_video_path)
            return

        jobs[job_id].update({'status': 'processing', 'progress': 90, 'message': 'Extracting clips...'})
        clip_paths = processor.extract_clips(actual_video_path, segments, max_clips)
        os.remove(actual_video_path)

        if not clip_paths:
            jobs[job_id].update({'status': 'failed', 'message': 'Could not extract any clips.'})
            return
            
        result_data = [{'name': Path(p).name, 'url': f'/static/clips/{Path(p).name}'} for p in clip_paths]
        jobs[job_id].update({
            'status': 'completed', 
            'progress': 100, 
            'message': 'Processing complete!', 
            'result': result_data
        })
    except Exception as e:
        logger.error(f"Error in job {job_id}: {e}")
        jobs[job_id].update({'status': 'failed', 'message': f'An unexpected error occurred: {e}'})

def process_uploaded_job(job_id, file_path, max_clips):
    """Process an uploaded video job with progress updates."""
    global jobs
    
    def progress_callback(progress):
        # Update progress for analysis, which is the main part here
        jobs[job_id]['progress'] = int(progress * 0.9) # 90% of the time

    try:
        jobs[job_id].update({'status': 'processing', 'progress': 5, 'message': 'Analyzing video...'})
        
        segments = processor.find_speaker_segments(file_path, progress_callback=progress_callback)
        
        if not segments:
            jobs[job_id].update({'status': 'failed', 'message': 'No speaker segments found.'})
            os.remove(file_path)
            return

        jobs[job_id].update({'status': 'processing', 'progress': 95, 'message': 'Extracting clips...'})
        clip_paths = processor.extract_clips(file_path, segments, max_clips)
        os.remove(file_path)

        if not clip_paths:
            jobs[job_id].update({'status': 'failed', 'message': 'Could not extract any clips.'})
            return

        result_data = [{'name': Path(p).name, 'url': f'/static/clips/{Path(p).name}'} for p in clip_paths]
        jobs[job_id].update({
            'status': 'completed', 
            'progress': 100, 
            'message': 'Processing complete!', 
            'result': result_data
        })
    except Exception as e:
        logger.error(f"Error in uploaded job {job_id}: {e}")
        jobs[job_id].update({'status': 'failed', 'message': f'An unexpected error occurred: {e}'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_video_route():
    global job_counter
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    job_counter += 1
    job_id = f"job_{job_counter}"
    jobs[job_id] = {'id': job_id, 'status': 'queued', 'progress': 0, 'message': 'Waiting to start...', 'result': [], 'start_time': time.time()}
    
    thread = threading.Thread(target=process_job, args=(job_id, url, 5))
    thread.daemon = True
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    global job_counter
    if 'videoFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['videoFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        filename = secure_filename(file.filename)
        file_path = DOWNLOADS_DIR / filename
        file.save(str(file_path))
        
        job_counter += 1
        job_id = f"job_{job_counter}"
        jobs[job_id] = {'id': job_id, 'status': 'queued', 'progress': 0, 'message': 'File uploaded, waiting...', 'result': [], 'start_time': time.time()}

        thread = threading.Thread(target=process_uploaded_job, args=(job_id, str(file_path), 5))
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id})

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    job = jobs.get(job_id)
    if job:
        elapsed = time.time() - job.get('start_time', time.time())
        job['time_elapsed'] = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        return jsonify(job)
    return jsonify({'error': 'Job not found'}), 404

@app.route('/download/<filename>')
def download_clip(filename):
    file_path = CLIPS_DIR / filename
    if file_path.exists():
        return send_file(str(file_path), as_attachment=True, mimetype='video/mp4')
    return "File not found.", 404

@app.route('/api/jobs')
def list_jobs():
    return jsonify(list(jobs.values()))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
