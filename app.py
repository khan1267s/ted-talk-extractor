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
import threading
from datetime import datetime

# Web framework
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

# Video processing
import yt_dlp
from moviepy.editor import VideoFileClip

# Person detection
import face_recognition

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

class WebSpeakerExtractor:
    def __init__(self):
        """
        Initialize the web speaker extractor.
        """
        self.output_dir = CLIPS_DIR

    def download_video(self, url: str) -> Optional[str]:
        """Download video from YouTube URL using yt-dlp."""
        try:
            logger.info(f"Downloading video from: {url}")
            ydl_opts = {
                'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(DOWNLOADS_DIR / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            proxy_url = os.environ.get('PROXY_URL')
            if proxy_url:
                logger.info(f"Using proxy: {proxy_url}")
                ydl_opts['proxy'] = proxy_url

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
        Check for a single person using face_recognition.
        """
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Check if exactly one face was found
            if len(face_locations) != 1:
                return False

            # Check for low edge density (no slides/text)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return edge_density < 0.15

        except Exception as e:
            logger.debug(f"Frame check failed: {e}")
            return False

    def find_speaker_segments(self, video_path: str) -> List[Tuple[float, float]]:
        """Find speaker-only segments."""
        logger.info(f"Finding speaker segments in: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        logger.info(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}")

        sample_interval = 2
        frame_interval = int(fps * sample_interval)
        good_frames = []

        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            if self.check_speaker_frame(frame):
                good_frames.append(frame_idx)
        
        cap.release()
        
        segments = self.frames_to_segments(good_frames, fps, duration)
        logger.info(f"Found {len(segments)} speaker segments")
        return segments

    def frames_to_segments(self, good_frames: List[int], fps: float, duration: float) -> List[Tuple[float, float]]:
        """Convert good frame numbers to time segments, tolerating gaps."""
        if not good_frames:
            return []

        segments = []
        start_frame = good_frames[0]
        max_gap_seconds = 10
        max_gap_frames = int(fps * max_gap_seconds)
        min_segment_duration = 25

        for i in range(1, len(good_frames)):
            gap = good_frames[i] - good_frames[i - 1]
            if gap > max_gap_frames:
                end_frame = good_frames[i - 1]
                if (end_frame - start_frame) / fps >= min_segment_duration:
                    segments.append((start_frame / fps, end_frame / fps))
                start_frame = good_frames[i]
        
        end_frame = good_frames[-1]
        if (end_frame - start_frame) / fps >= min_segment_duration:
            segments.append((start_frame / fps, end_frame / fps))

        return segments

    def extract_clips(self, video_path: str, segments: List[Tuple[float, float]], max_clips: int = 5) -> List[str]:
        """Extract non-overlapping clips."""
        logger.info(f"Extracting clips from {len(segments)} segments...")
        clip_paths = []
        try:
            video = VideoFileClip(video_path)
        except Exception as e:
            logger.error(f"Failed to load video file with MoviePy: {e}")
            return []
        
        segments = sorted(segments, key=lambda x: x[1] - x[0], reverse=True)
        used_time_ranges = []
        clip_duration = 25

        for i, (start_time, end_time) in enumerate(segments):
            if len(clip_paths) >= max_clips:
                break
            if end_time - start_time < clip_duration:
                continue

            clip_start = start_time
            is_overlapping = any(max(clip_start, s) < min(clip_start + clip_duration, e) for s, e in used_time_ranges)
            
            if is_overlapping:
                continue

            try:
                clip = video.subclip(clip_start, clip_start + clip_duration)
                video_name = Path(video_path).stem
                clip_filename = f"{video_name}_speaker_{len(clip_paths)+1:03d}.mp4"
                clip_path = self.output_dir / clip_filename
                
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    preset='medium',
                    threads=4,
                    logger=None
                )
                
                clip_paths.append(str(clip_path))
                used_time_ranges.append((clip_start, clip_start + clip_duration))
                logger.info(f"Extracted clip {len(clip_paths)}: {clip_filename}")

            except Exception as e:
                logger.error(f"Error extracting clip {i+1}: {e}")

        video.close()
        return clip_paths

    def process_video(self, url: str, max_clips: int = 5) -> List[str]:
        """Full processing pipeline for a URL."""
        logger.info(f"Processing video: {url}")
        video_path = self.download_video(url)
        if not video_path:
            logger.error("Failed to download video")
            return []
        
        segments = self.find_speaker_segments(video_path)
        if not segments:
            logger.warning("No valid speaker segments found")
            os.remove(video_path)
            return []
        
        clip_paths = self.extract_clips(video_path, segments, max_clips)
        
        try:
            os.remove(video_path)
            logger.info("Cleaned up downloaded video")
        except Exception as e:
            logger.warning(f"Could not clean up video file: {e}")
        
        return clip_paths

    def process_uploaded_video(self, video_path: str, max_clips: int = 5) -> List[str]:
        """Full processing pipeline for an uploaded file."""
        logger.info(f"Processing uploaded video: {video_path}")
        segments = self.find_speaker_segments(video_path)
        if not segments:
            logger.warning("No valid speaker segments found")
            os.remove(video_path)
            return []
        
        clip_paths = self.extract_clips(video_path, segments, max_clips)
        
        try:
            os.remove(video_path)
            logger.info("Cleaned up uploaded video")
        except Exception as e:
            logger.warning(f"Could not clean up uploaded video file: {e}")
        
        return clip_paths

# Initialize extractor
extractor = WebSpeakerExtractor()

def process_job(job_id, url, max_clips):
    """Process a job in a separate thread."""
    global jobs
    try:
        jobs[job_id].update({'status': 'processing', 'progress': 10, 'message': 'Downloading video...'})
        video_path = extractor.download_video(url)
        if not video_path:
            jobs[job_id].update({'status': 'failed', 'message': 'Failed to download video.'})
            return

        jobs[job_id].update({'progress': 30, 'message': 'Analyzing video...'})
        segments = extractor.find_speaker_segments(video_path)
        if not segments:
            jobs[job_id].update({'status': 'failed', 'message': 'No speaker segments found.'})
            os.remove(video_path)
            return

        jobs[job_id].update({'progress': 60, 'message': 'Extracting clips...'})
        clip_paths = extractor.extract_clips(video_path, segments, max_clips)
        os.remove(video_path)

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
    """Process an uploaded video job in a separate thread."""
    global jobs
    try:
        jobs[job_id].update({'status': 'processing', 'progress': 30, 'message': 'Analyzing video...'})
        segments = extractor.find_speaker_segments(file_path)
        if not segments:
            jobs[job_id].update({'status': 'failed', 'message': 'No speaker segments found.'})
            os.remove(file_path)
            return

        jobs[job_id].update({'progress': 60, 'message': 'Extracting clips...'})
        clip_paths = extractor.extract_clips(file_path, segments, max_clips)
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
