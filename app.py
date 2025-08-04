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
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size for video uploads

# --- FIX for FileNotFoundError on Windows ---
# Define absolute paths based on the location of this file (app.py)
APP_ROOT = Path(__file__).parent.resolve()
DOWNLOADS_DIR = APP_ROOT / "downloads"
# Clips are served from 'static', so we need the static folder in the path.
CLIPS_DIR = APP_ROOT / "static" / "clips"

# Create directories if they don't exist to avoid errors
DOWNLOADS_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True, parents=True)
# --- End of fix ---


# Global variables for job tracking
jobs = {}
job_counter = 0

class WebSpeakerExtractor:
    def __init__(self):
        """
        Initialize the web speaker extractor.
        """
        self.output_dir = CLIPS_DIR # Use absolute path
        
        # Initialize MediaPipe once
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
    
    def download_video(self, url: str) -> Optional[str]:
        """Download video from YouTube URL using yt-dlp."""
        try:
            logger.info(f"Downloading video from: {url}")
            
            downloads_dir = DOWNLOADS_DIR # Use absolute path
            
            # --- QUALITY IMPROVEMENT ---
            # Download up to 1080p for better quality
            ydl_opts = {
                'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(downloads_dir / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }

            # Check for a proxy URL from environment variables
            proxy_url = os.environ.get('PROXY_URL')
            if proxy_url:
                logger.info(f"Using proxy: {proxy_url}")
                ydl_opts['proxy'] = proxy_url

            # --- End of quality improvement ---
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)  # Get info first
                video_id = info['id']
                file_path = downloads_dir / f"{video_id}.mp4"

                # Get expected file size
                expected_filesize = 0
                for f in info['formats']:
                    if f['format_id'] == info['format_id']:
                        expected_filesize = f.get('filesize') or f.get('filesize_approx')
                        break
                
                logger.info(f"Expected file size: {expected_filesize} bytes")

                # Start download
                ydl.download([url])

                # --- FILE INTEGRITY CHECK ---
                if not file_path.exists():
                    logger.error("Video file not found after download.")
                    return None
                
                actual_filesize = file_path.stat().st_size
                logger.info(f"Actual file size: {actual_filesize} bytes")
                
                if expected_filesize and actual_filesize < expected_filesize * 0.9:
                    logger.error(f"Incomplete download. Expected ~{expected_filesize}, got {actual_filesize}")
                    os.remove(file_path) # Clean up partial file
                    return None
                
                logger.info(f"Video downloaded successfully: {file_path}")
                return str(file_path)
                # --- END FILE INTEGRITY CHECK ---

        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None

    def check_speaker_frame(self, frame: np.ndarray) -> bool:
        """
        Fast check for speaker-only frames.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame contains only speaker
        """
        try:
            # Resize for speed - analysis is still fast, but on original downloaded frames
            frame = cv2.resize(frame, (640, 360)) # Increased size for better analysis
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Check for exactly one face
            face_results = self.mp_face.process(rgb_frame)
            if len(face_results.detections or []) != 1:
                return False
            
            # Check for low edge density (no slides/text)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # If edge density is low (no slides/text), consider it speaker-only
            if edge_density < 0.12: # Slightly higher threshold for higher-res
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Frame check failed: {e}")
            return False
    
    def find_speaker_segments(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Find speaker-only segments in the video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (start_time, end_time) tuples for valid segments
        """
        logger.info(f"Finding speaker segments in: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video duration: {duration:.2f} seconds, FPS: {fps:.2f}")
        
        # Sample frames every 2 seconds for more granular analysis
        sample_interval = 2
        frame_interval = int(fps * sample_interval)
        
        good_frames = []
        
        # Scan frames
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if self.check_speaker_frame(frame):
                good_frames.append(frame_idx)
        
        cap.release()
        
        # Convert good frames to segments
        segments = self.frames_to_segments(good_frames, fps, duration)
        logger.info(f"Found {len(segments)} speaker segments")
        return segments
    
    def frames_to_segments(self, good_frames: List[int], fps: float, duration: float) -> List[Tuple[float, float]]:
        """
        Convert list of good frame numbers to time segments.
        This version is more tolerant of small gaps (e.g., brief audience shots).
        
        Args:
            good_frames: List of frame numbers that are speaker-only
            fps: Frames per second
            duration: Total video duration
            
        Returns:
            List of (start_time, end_time) tuples
        """
        if not good_frames:
            return []

        segments = []
        if not good_frames:
            return segments

        start_frame = good_frames[0]
        max_gap_seconds = 10  # Allow up to a 10-second gap
        max_gap_frames = int(fps * max_gap_seconds)
        min_segment_duration = 20 # Minimum length of a valid segment

        for i in range(1, len(good_frames)):
            gap = good_frames[i] - good_frames[i - 1]
            if gap > max_gap_frames:
                end_frame = good_frames[i - 1]
                if (end_frame - start_frame) / fps >= min_segment_duration:
                    segments.append((start_frame / fps, end_frame / fps))
                start_frame = good_frames[i]

        # Add the last segment
        end_frame = good_frames[-1]
        if (end_frame - start_frame) / fps >= min_segment_duration:
            segments.append((start_frame / fps, end_frame / fps))

        return segments
    
    def extract_clips(self, video_path: str, segments: List[Tuple[float, float]], max_clips: int = 5) -> List[str]:
        """
        Extract clips from video based on valid segments.
        Ensures clips don't overlap by maintaining minimum time gaps.
        
        Args:
            video_path: Path to video file
            segments: List of (start_time, end_time) tuples
            max_clips: Maximum number of clips to extract (default 5)
            
        Returns:
            List of paths to extracted clip files
        """
        logger.info(f"Extracting clips from {len(segments)} segments...")
        
        clip_paths = []
        try:
            video = VideoFileClip(video_path)
        except Exception as e:
            logger.error(f"Failed to load video file with MoviePy: {e}")
            return []
        
        # Sort segments by duration (longest first)
        segments = sorted(segments, key=lambda x: x[1] - x[0], reverse=True)
        
        # Track used time ranges to prevent overlap
        used_time_ranges = []
        clip_duration = 30  # seconds
        
        for i, (start_time, end_time) in enumerate(segments):
            if len(clip_paths) >= max_clips:
                break

            segment_duration = end_time - start_time
            if segment_duration < clip_duration:
                continue

            # Find a 30-second slot within this segment that doesn't overlap with used clips
            clip_start = start_time
            
            # Check for overlap
            is_overlapping = False
            for used_start, used_end in used_time_ranges:
                if max(clip_start, used_start) < min(clip_start + clip_duration, used_end):
                    is_overlapping = True
                    break
            
            if is_overlapping:
                continue

            try:
                # Extract the 30-second clip
                clip = video.subclip(clip_start, clip_start + clip_duration)
                
                # Generate output filename
                video_name = Path(video_path).stem
                clip_filename = f"{video_name}_speaker_{len(clip_paths)+1:03d}.mp4"
                clip_path = self.output_dir / clip_filename
                
                # Use a better preset for higher quality encoding
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    preset='medium', # Faster preset
                    threads=4,
                    logger=None
                )
                
                clip_paths.append(str(clip_path))
                used_time_ranges.append((clip_start, clip_start + clip_duration))
                logger.info(f"Extracted clip {len(clip_paths)}: {clip_filename} (time: {clip_start:.1f}s - {clip_start + clip_duration:.1f}s)")

            except Exception as e:
                logger.error(f"Error extracting clip {i+1}: {e}")

        video.close()
        return clip_paths

    def process_video(self, url: str, max_clips: int = 5) -> List[str]:
        """
        Process a single video: download, analyze, and extract clips.
        
        Args:
            url: YouTube video URL
            max_clips: Maximum number of clips to extract (default 5)
            
        Returns:
            List of paths to extracted clip files
        """
        logger.info(f"Processing video: {url}")
        
        # Download video
        video_path = self.download_video(url)
        if not video_path:
            logger.error("Failed to download video")
            return []
        
        # Find speaker segments
        segments = self.find_speaker_segments(video_path)
        
        if not segments:
            logger.warning("No valid speaker segments found")
            return []
        
        # Extract clips
        clip_paths = self.extract_clips(video_path, segments, max_clips)
        
        # Clean up downloaded video
        try:
            os.remove(video_path)
            logger.info("Cleaned up downloaded video")
        except Exception as e:
            logger.warning(f"Could not clean up video file: {e}")
        
        return clip_paths

    def process_uploaded_video(self, video_path: str, max_clips: int = 5) -> List[str]:
        """
        Process an uploaded video file: analyze and extract clips.
        
        Args:
            video_path: Path to the uploaded video file
            max_clips: Maximum number of clips to extract (default 5)
            
        Returns:
            List of paths to extracted clip files
        """
        logger.info(f"Processing uploaded video: {video_path}")
        
        # Find speaker segments
        segments = self.find_speaker_segments(video_path)
        
        if not segments:
            logger.warning("No valid speaker segments found")
            return []
        
        # Extract clips
        clip_paths = self.extract_clips(video_path, segments, max_clips)
        
        # Clean up uploaded video file after processing
        try:
            os.remove(video_path)
            logger.info("Cleaned up uploaded video")
        except Exception as e:
            logger.warning(f"Could not clean up uploaded video file: {e}")
        
        return clip_paths

# Initialize the extractor
extractor = WebSpeakerExtractor()

def process_job(job_id, url, max_clips):
    """Process a job in a separate thread."""
    global jobs
    
    try:
        # Step 1: Initialize
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0
        jobs[job_id]['message'] = 'Initializing download...'
        time.sleep(1)  # Brief pause for UI update
        
        # Step 2: Download video
        jobs[job_id]['progress'] = 10
        jobs[job_id]['message'] = 'Downloading video from YouTube...'
        video_path = extractor.download_video(url)
        
        if not video_path:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['message'] = 'Failed to download video. It may be incomplete or blocked.'
            return
        
        # Step 3: Analyze video
        jobs[job_id]['progress'] = 30
        jobs[job_id]['message'] = 'Analyzing video structure...'
        segments = extractor.find_speaker_segments(video_path)
        
        if not segments:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['message'] = 'No speaker segments found. The video may not be suitable.'
            try:
                os.remove(video_path)
            except:
                pass
            return

        # Step 4: Extract clips
        jobs[job_id]['progress'] = 60
        jobs[job_id]['message'] = 'Extracting speaker clips...'
        clip_paths = extractor.extract_clips(video_path, segments, max_clips)

        # Final step: Clean up
        try:
            os.remove(video_path)
            logger.info(f"Cleaned up source video: {video_path}")
        except Exception as e:
            logger.warning(f"Could not clean up source video: {e}")
            
        if not clip_paths:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['message'] = 'Could not extract any clips after processing.'
            return
            
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Processing complete!'
        jobs[job_id]['result'] = [str(Path(p).name) for p in clip_paths]

    except Exception as e:
        logger.error(f"Error in job {job_id}: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = f'An unexpected error occurred: {e}'


def process_uploaded_job(job_id, file_path, max_clips):
    """Process an uploaded video job in a separate thread."""
    global jobs

    try:
        # Step 1: Initialize
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 5
        jobs[job_id]['message'] = 'Validating uploaded file...'
        time.sleep(1)

        # Step 2: Analyze video
        jobs[job_id]['progress'] = 30
        jobs[job_id]['message'] = 'Analyzing video structure...'
        segments = extractor.find_speaker_segments(file_path)

        if not segments:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['message'] = 'No speaker segments found in the uploaded video.'
            try:
                os.remove(file_path)
            except:
                pass
            return

        # Step 3: Extract clips
        jobs[job_id]['progress'] = 60
        jobs[job_id]['message'] = 'Extracting speaker clips...'
        clip_paths = extractor.extract_clips(file_path, segments, max_clips)

        # Final step: Clean up uploaded file
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up uploaded file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up uploaded file: {e}")

        if not clip_paths:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['message'] = 'Could not extract any clips from the uploaded video.'
            return

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['message'] = 'Processing complete!'
        jobs[job_id]['result'] = [str(Path(p).name) for p in clip_paths]

    except Exception as e:
        logger.error(f"Error in uploaded job {job_id}: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = f'An unexpected error occurred: {e}'


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
    
    jobs[job_id] = {
        'id': job_id,
        'status': 'queued',
        'progress': 0,
        'message': 'Waiting to start...',
        'result': [],
        'start_time': time.time()
    }

    # Start the processing in a background thread
    thread = threading.Thread(target=process_job, args=(job_id, url, 5))
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

    # Validate file type
    if file and file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        filename = secure_filename(file.filename)
        # Save to the absolute path
        file_path = DOWNLOADS_DIR / filename
        file.save(str(file_path))
        
        job_counter += 1
        job_id = f"job_{job_counter}"
        
        jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'progress': 0,
            'message': 'File uploaded, waiting to process...',
            'result': [],
            'start_time': time.time()
        }

        # Start processing in a background thread
        thread = threading.Thread(target=process_uploaded_job, args=(job_id, str(file_path), 5))
        thread.start()
        
        return jsonify({'job_id': job_id})

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    job = jobs.get(job_id)
    if job:
        # Calculate time elapsed
        elapsed = time.time() - job.get('start_time', time.time())
        job['time_elapsed'] = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        return jsonify(job)
    return jsonify({'error': 'Job not found'}), 404

@app.route('/download/<filename>')
def download_clip(filename):
    """
    Serve a file for download.
    The filename is relative to the CLIPS_DIR.
    """
    # Use the absolute path for sending the file
    file_path = CLIPS_DIR / filename
    if file_path.exists():
        return send_file(
            str(file_path),
            as_attachment=True,
            mimetype='video/mp4'
        )
    return "File not found.", 404

@app.route('/api/jobs')
def list_jobs():
    return jsonify(list(jobs.values()))

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on the network
    # Debug mode is off for production-like environment
    app.run(host='0.0.0.0', port=5000, debug=False)
