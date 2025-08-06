#!/usr/bin/env python3
"""
TED Talk Speaker Extractor - Web Interface
A Flask web application for extracting speaker-only clips from TED Talks.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import json
import time
from datetime import datetime

# Web framework
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

# Video processing
import yt_dlp
from moviepy.editor import VideoFileClip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Define absolute paths
APP_ROOT = Path(__file__).parent.resolve()
DOWNLOADS_DIR = APP_ROOT / "downloads"
CLIPS_DIR = APP_ROOT / "static" / "clips"
CASCADE_FILE = APP_ROOT / "haarcascade_frontalface_default.xml"

# Create directories
DOWNLOADS_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True, parents=True)

# Global variables for job tracking
jobs = {}
job_counter = 0

class WebSpeakerExtractor:
    def __init__(self):
        """
        Initialize the web speaker extractor.
        """
        self.output_dir = CLIPS_DIR
        self.face_cascade = cv2.CascadeClassifier(str(CASCADE_FILE))
        if self.face_cascade.empty():
            logger.error("Could not load Haar Cascade classifier. Make sure the XML file is present.")

    def download_video(self, url: str) -> Optional[str]:
        """Download video from YouTube URL using yt-dlp."""
        try:
            logger.info(f"Downloading video from: {url}")
            ydl_opts = {
                'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(DOWNLOADS_DIR / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                file_path = list(DOWNLOADS_DIR.glob(f'{video_id}.*'))[0]
                
                if file_path.exists():
                    logger.info(f"Video downloaded successfully: {file_path}")
                    return str(file_path)
                else:
                    logger.error("Video file not found after download.")
                    return None
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None

    def check_speaker_frame(self, frame: np.ndarray) -> bool:
        """
        Check for a single person using OpenCV Haar Cascade.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) != 1:
                return False

            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return edge_density < 0.20

        except Exception as e:
            logger.debug(f"Frame check failed: {e}")
            return False

    def find_speaker_segments(self, video_path: str) -> List[Tuple[float, float]]:
        """Find speaker-only segments."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        good_frames = []
        
        frame_interval = int(fps * 2)
        if frame_interval == 0: frame_interval = 1

        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            if self.check_speaker_frame(frame):
                good_frames.append(frame_idx)
        
        cap.release()
        return self.frames_to_segments(good_frames, fps)

    def frames_to_segments(self, good_frames: List[int], fps: float) -> List[Tuple[float, float]]:
        """Convert good frame numbers to time segments, tolerating gaps."""
        if not good_frames: return []

        segments, start_frame = [], good_frames[0]
        max_gap_frames = int(fps * 10)
        min_segment_duration = 20

        for i in range(1, len(good_frames)):
            if good_frames[i] - good_frames[i-1] > max_gap_frames:
                end_frame = good_frames[i-1]
                if (end_frame - start_frame) / fps >= min_segment_duration:
                    segments.append((start_frame / fps, end_frame / fps))
                start_frame = good_frames[i]
        
        end_frame = good_frames[-1]
        if (end_frame - start_frame) / fps >= min_segment_duration:
            segments.append((start_frame / fps, end_frame / fps))

        return segments

    def extract_clips(self, video_path: str, segments: List[Tuple[float, float]], max_clips: int) -> List[str]:
        """Extract non-overlapping clips."""
        clip_paths, video = [], VideoFileClip(video_path)
        segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)
        used_times, clip_duration = [], 25

        for start, end in segments:
            if len(clip_paths) >= max_clips: break
            if end - start < clip_duration: continue
            
            is_overlapping = any(max(start, s) < min(start + clip_duration, e) for s, e in used_times)
            if not is_overlapping:
                try:
                    clip_path = self.output_dir / f"{Path(video_path).stem}_speaker_{len(clip_paths)+1:03d}.mp4"
                    video.subclip(start, start + clip_duration).write_videofile(str(clip_path), codec='libx264', audio_codec='aac', preset='fast', threads=2, logger=None)
                    clip_paths.append(str(clip_path))
                    used_times.append((start, start + clip_duration))
                    logger.info(f"Extracted clip: {clip_path.name}")
                except Exception as e:
                    logger.error(f"Error extracting clip: {e}")

        video.close()
        return clip_paths

extractor = WebSpeakerExtractor()

def run_processing(job_id, video_path, max_clips):
    """
    This function runs the main video processing logic sequentially.
    """
    global jobs
    try:
        jobs[job_id].update({'status': 'processing', 'progress': 30, 'message': 'Analyzing video...'})
        segments = extractor.find_speaker_segments(video_path)
        if not segments:
            raise ValueError("No valid speaker segments found.")

        jobs[job_id].update({'progress': 60, 'message': 'Extracting clips...'})
        clip_paths = extractor.extract_clips(video_path, segments, max_clips)
        if not clip_paths:
            raise ValueError("Could not extract any clips.")

        result_data = [{'name': Path(p).name, 'url': f'/static/clips/{Path(p).name}'} for p in clip_paths]
        jobs[job_id].update({'status': 'completed', 'progress': 100, 'message': 'Processing complete!', 'result': result_data})

    except Exception as e:
        logger.error(f"Error in job {job_id}: {e}")
        jobs[job_id].update({'status': 'failed', 'message': str(e)})
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Cleaned up source file: {video_path}")
            except OSError as e:
                logger.error(f"Error deleting file {video_path}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_video_route():
    global job_counter
    url = request.json.get('url')
    max_clips = request.json.get('max_clips', 5)
    if not url: return jsonify({'error': 'URL is required'}), 400

    job_counter += 1
    job_id = f"job_{job_counter}_{int(time.time())}"
    jobs[job_id] = {'id': job_id, 'status': 'queued', 'progress': 0, 'message': 'Starting job...', 'result': [], 'start_time': time.time()}
    
    jobs[job_id].update({'status': 'processing', 'progress': 10, 'message': 'Downloading video...'})
    video_path = extractor.download_video(url)
    if not video_path:
        jobs[job_id].update({'status': 'failed', 'message': 'Failed to download video.'})
        return jsonify({'job_id': job_id})
    
    # Run processing directly and wait for it to finish
    run_processing(job_id, video_path, max_clips)
    return jsonify({'job_id': job_id})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    global job_counter
    if 'videoFile' not in request.files: return jsonify({'error': 'No file part in request.'}), 400
    file = request.files['videoFile']
    if file.filename == '': return jsonify({'error': 'No selected file.'}), 400
    
    max_clips = int(request.form.get('max_clips', 5))
    
    if file and file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        filename = secure_filename(file.filename)
        file_path = DOWNLOADS_DIR / filename
        file.save(str(file_path))
        
        job_counter += 1
        job_id = f"job_{job_counter}_{int(time.time())}"
        jobs[job_id] = {'id': job_id, 'status': 'queued', 'progress': 0, 'message': 'File uploaded, starting job...', 'result': [], 'start_time': time.time()}

        # Run processing directly and wait for it to finish
        run_processing(job_id, str(file_path), max_clips)
        
        return jsonify({'job_id': job_id})
    
    return jsonify({'error': 'Invalid file type.'}), 400

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    job = jobs.get(job_id)
    if job:
        return jsonify(job)
    return jsonify({'error': 'Job not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
