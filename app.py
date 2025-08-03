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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
                info = ydl.extract_info(url, download=True)
                video_id = info['id']
                video_path = downloads_dir / f"{video_id}.mp4"
                
                if video_path.exists():
                    logger.info(f"Video downloaded successfully: {video_path}")
                    return str(video_path)
                else:
                    logger.error("Video file not found after download")
                    return None
            
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
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video duration: {duration:.2f} seconds, FPS: {fps:.2f}")
        
        # Sample frames every 4 seconds for efficiency
        sample_interval = 4
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
        Ensures segments don't overlap by requiring minimum gap between them.
        
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
        if good_frames: # Check if list is not empty
            start_frame = good_frames[0]
            sample_interval = 4  # seconds
            frame_interval = int(fps * sample_interval)
            
            # Minimum gap between segments (in frames) to prevent overlap
            min_gap_frames = int(fps * 5)  # 5 second minimum gap
            
            for i in range(1, len(good_frames)):
                # If gap is too large, end current segment
                if good_frames[i] - good_frames[i-1] > frame_interval + 1:
                    end_frame = good_frames[i-1]
                    start_time = start_frame / fps
                    end_time = end_frame / fps
                    
                    # Only keep segments that are at least 20 seconds
                    if end_time - start_time >= 20:
                        segments.append((start_time, end_time))
                    
                    start_frame = good_frames[i]
            
            # Handle the last segment
            end_frame = good_frames[-1]
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            if end_time - start_time >= 20:
                segments.append((start_time, end_time))
        
        # Filter out overlapping segments
        non_overlapping_segments = []
        for i, (start_time, end_time) in enumerate(segments):
            # Check if this segment overlaps with any previous segment
            overlaps = False
            for prev_start, prev_end in non_overlapping_segments:
                # Check if segments overlap (with 5 second buffer)
                if (start_time < prev_end + 5 and end_time > prev_start - 5):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping_segments.append((start_time, end_time))
        
        return non_overlapping_segments
    
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
        video = VideoFileClip(video_path)
        
        # Sort segments by duration (longest first) and limit to max_clips
        segments = sorted(segments, key=lambda x: x[1] - x[0], reverse=True)[:max_clips]
        
        # Track used time ranges to prevent overlap
        used_ranges = []
        clip_duration = 30  # seconds
        
        for i, (start_time, end_time) in enumerate(segments):
            try:
                # Check if this segment overlaps with any previously used range
                overlaps = False
                for used_start, used_end in used_ranges:
                    # Check if segments overlap (with 2 second buffer)
                    if (start_time < used_end + 2 and start_time + clip_duration > used_start - 2):
                        overlaps = True
                        break
                
                if overlaps:
                    logger.info(f"Skipping segment {i+1} due to overlap")
                    continue
                
                # Extract clip (30 seconds max)
                actual_duration = min(clip_duration, end_time - start_time)
                clip = video.subclip(start_time, start_time + actual_duration)
                
                # Generate output filename
                video_name = Path(video_path).stem
                clip_filename = f"{video_name}_speaker_{len(clip_paths)+1:03d}.mp4"
                clip_path = self.output_dir / clip_filename
                
                # --- QUALITY IMPROVEMENT ---
                # Use a better preset for higher quality encoding
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    preset='slow', # Better quality preset
                    ffmpeg_params=['-crf', '18'] # Lower CRF means better quality
                )
                # --- End of quality improvement ---
                
                clip_paths.append(str(clip_path))
                used_ranges.append((start_time, start_time + actual_duration))
                logger.info(f"Extracted clip {len(clip_paths)}: {clip_filename} (time: {start_time:.1f}s - {start_time + actual_duration:.1f}s)")
                
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

# Initialize the extractor
extractor = WebSpeakerExtractor()

def process_job(job_id, url, max_clips):
    """Process a job in a separate thread."""
    global jobs
    
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0
        jobs[job_id]['message'] = 'Downloading video...'
        
        # Process the video
        clip_paths = extractor.process_video(url, max_clips)
        
        if clip_paths:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['message'] = f'Successfully extracted {len(clip_paths)} clips'
            jobs[job_id]['clips'] = clip_paths
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
        else:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['message'] = 'No speaker segments found or processing failed'
            
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = f'Error: {str(e)}'
        logger.error(f"Job {job_id} failed: {e}")

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_video():
    """API endpoint to process a video."""
    global job_counter
    
    data = request.get_json()
    url = data.get('url')
    max_clips = data.get('max_clips', 5)
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    # Create new job
    job_counter += 1
    job_id = f"job_{job_counter}"
    
    jobs[job_id] = {
        'id': job_id,
        'url': url,
        'max_clips': max_clips,
        'status': 'queued',
        'progress': 0,
        'message': 'Job queued',
        'created_at': datetime.now().isoformat(),
        'clips': []
    }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_job, args=(job_id, url, max_clips))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Job started'
    })

@app.route('/api/job/<job_id>')
def get_job_status(job_id):
    """Get job status."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    # Convert clip paths to URLs for download
    clips = []
    for clip_path in job.get('clips', []):
        clip_name = Path(clip_path).name
        clips.append({
            'name': clip_name,
            'url': url_for('download_clip', filename=clip_name)
        })
    
    return jsonify({
        'id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'clips': clips,
        'created_at': job['created_at'],
        'completed_at': job.get('completed_at')
    })

@app.route('/download/<filename>')
def download_clip(filename):
    """Download a clip file."""
    clip_path = CLIPS_DIR / filename # Use absolute path
    if clip_path.exists():
        return send_file(clip_path, as_attachment=True)
    else:
        logger.error(f"File not found for download: {clip_path}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/jobs')
def list_jobs():
    """List all jobs."""
    job_list = []
    for job_id, job in jobs.items():
        job_list.append({
            'id': job['id'],
            'url': job['url'],
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message'],
            'created_at': job['created_at'],
            'completed_at': job.get('completed_at')
        })
    
    return jsonify(job_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
