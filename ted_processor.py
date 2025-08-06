#!/usr/bin/env python3
"""
TED Talk Processor
Downloads TED Talks from YouTube and extracts speaker-only clips.
"""

import os
import sys
import cv2
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from tqdm import tqdm
import logging

# Set environment variable to handle PyTorch 2.6+ compatibility
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

import yt_dlp

# Video processing
# from pytube import YouTube # No longer using pytube

# Person detection
import mediapipe as mp
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Import our OpenAI detector
try:
    from openai_detector import OpenAIPersonDetector
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TEDTalkProcessor:
    def __init__(self, output_dir: str = "output_clips"):
        """
        Initialize the TED Talk processor.
        
        Args:
            output_dir: Directory to save extracted clips
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe for person detection
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Use OpenAI Detector if available
        if OPENAI_AVAILABLE:
            self.detector = OpenAIPersonDetector()
            logger.info("Using OpenAI CLIP detector.")
        # Fallback to YOLO if available
        elif YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.detector = self
                logger.info("Using YOLO detector.")
            except Exception as e:
                logger.warning(f"Could not load YOLO model: {e}")
                self.detector = None
        else:
            self.detector = None
            logger.error("No person detection model available.")
    
    def get_speaker_name(self, video_title: str) -> str:
        """
        Extracts the speaker's name from a TED Talk video title.
        
        Args:
            video_title: The title of the YouTube video.
            
        Returns:
            The extracted speaker name, or a default name if not found.
        """
        # Common TED patterns: "Speaker: Title" or "Title | Speaker"
        match = re.match(r"([^:|]+)[:|]", video_title)
        if match:
            speaker = match.group(1).strip()
        else:
            # Fallback for "Title by Speaker"
            match_by = re.search(r"by (.+)", video_title, re.IGNORECASE)
            if match_by:
                speaker = match_by.group(1).strip()
            else:
                speaker = "Unknown_Speaker"

        # Sanitize name for filesystem
        speaker = re.sub(r'[\\/*?:"<>|]', "", speaker)
        speaker = speaker.replace(" ", "_")
        return speaker

    def download_video(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Download video from YouTube URL using yt-dlp.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (path_to_video, video_title) or None if failed
        """
        downloads_dir = Path("downloads")
        downloads_dir.mkdir(exist_ok=True)
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': str(downloads_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extractaudio': False,
            'audioformat': 'mp3',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'referer': 'https://www.youtube.com/',
            'sleep_interval': 1,
            'max_sleep_interval': 5,
            'retries': 3,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                video_id = info_dict.get('id', None)
                video_title = info_dict.get('title', None)
                video_path = str(downloads_dir / f"{video_id}.mp4")
                
                if not os.path.exists(video_path):
                     # Try with webm if mp4 not available
                    video_path = str(downloads_dir / f"{video_id}.webm")
                    if not os.path.exists(video_path):
                        logger.error("Downloaded video file not found.")
                        return None

                logger.info(f"Video downloaded successfully: {video_path}")
                return video_path, video_title

        except Exception as e:
            logger.error(f"Error downloading video with yt-dlp: {e}")
            return None

    def detect_persons_in_frame(self, frame: np.ndarray) -> int:
        """
        Detect persons in a frame using multiple methods.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            The number of persons detected.
        """
        person_count = 0
        
        # Method 1: YOLO detection
        if self.yolo_model:
            try:
                results = self.yolo_model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if int(box.cls) == 0:  # person class in COCO
                                person_count += 1
            except Exception as e:
                logger.debug(f"YOLO detection failed: {e}")
        
        return person_count

    def is_speaker_only_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains only the main speaker.
        """
        if not self.detector:
            return False
        return self.detector.is_speaker_only_frame(frame)

    def analyze_video_segments(self, video_path: str, segment_duration: int = 30, overlap: int = 15) -> List[Tuple[float, float]]:
        """
        Analyze video to find speaker-only segments with overlap.
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each clip in seconds
            overlap: Overlap between clips in seconds
            
        Returns:
            List of (start_time, end_time) tuples for valid segments
        """
        logger.info(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        
        valid_segments = []
        step = segment_duration - overlap
        
        for start_time in tqdm(np.arange(0, duration - segment_duration, step), desc="Analyzing segments"):
            end_time = start_time + segment_duration
            
            # Sample 10 frames from this segment
            sample_times = np.linspace(start_time, end_time, 10)
            
            speaker_only_count = 0
            for sample_time in sample_times:
                cap.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
                ret, frame = cap.read()
                if ret and self.is_speaker_only_frame(frame):
                    speaker_only_count += 1
            
            if speaker_only_count / 10 >= 0.7: # At least 70% of frames are good
                valid_segments.append((start_time, end_time))
                logger.info(f"Valid segment found: {start_time:.1f}s - {end_time:.1f}s")

        cap.release()
        logger.info(f"Found {len(valid_segments)} valid speaker-only segments")
        return valid_segments

    def extract_clips(self, video_path: str, segments: List[Tuple[float, float]], speaker_name: str, speaker_dir: Path, max_clips: int = 5) -> List[str]:
        """
        Extract clips and save to a speaker-specific directory.
        """
        logger.info(f"Extracting up to {max_clips} clips for {speaker_name}...")
        
        clip_paths = []
        video = VideoFileClip(video_path)
        
        for i, (start_time, end_time) in enumerate(segments[:max_clips]):
            clip_filename = f"{speaker_name}_{i+1}.mp4"
            clip_path = speaker_dir / clip_filename
            
            try:
                clip = video.subclip(start_time, end_time)
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                clip_paths.append(str(clip_path))
                logger.info(f"Saved clip: {clip_path.name}")
            except Exception as e:
                logger.error(f"Error extracting clip {clip_filename}: {e}")

        video.close()
        return clip_paths

    def process_video(self, url: str, max_clips: int = 5, overlap: int = 15) -> List[str]:
        """
        Process a single video: download, analyze, and extract clips.
        """
        logger.info(f"Processing video: {url}")

        # Download video and get metadata
        download_result = self.download_video(url)
        if not download_result:
            return []
        video_path, video_title = download_result
        
        speaker_name = self.get_speaker_name(video_title)
        speaker_dir = self.output_dir / speaker_name
        
        # Skip if already processed
        if speaker_dir.exists() and any(speaker_dir.iterdir()):
            logger.info(f"Speaker '{speaker_name}' already processed. Skipping.")
            return [str(p) for p in speaker_dir.iterdir()]
        
        speaker_dir.mkdir(exist_ok=True)
        
        # Analyze and extract
        segments = self.analyze_video_segments(video_path, overlap=overlap)
        
        if not segments:
            logger.warning("No valid speaker-only segments found.")
            clip_paths = []
        else:
            clip_paths = self.extract_clips(video_path, segments, speaker_name, speaker_dir, max_clips)
        
        # Clean up
        try:
            os.remove(video_path)
            logger.info("Cleaned up downloaded video.")
        except OSError as e:
            logger.warning(f"Could not remove video file {video_path}: {e}")
            
        return clip_paths

def main():
    """Main function to run the TED Talk processor from command line."""
    parser = argparse.ArgumentParser(description="TED Talk Processor - Extract speaker-only clips.")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs to process")
    parser.add_argument("--output-dir", default="output_clips", help="Root output directory for clips")
    parser.add_argument("--max-clips", type=int, default=5, help="Maximum number of clips per video")
    parser.add_argument("--overlap", type=int, default=15, help="Overlap in seconds between clips")
    
    args = parser.parse_args()
    
    processor = TEDTalkProcessor(args.output_dir)
    
    for url in args.urls:
        processor.process_video(url, args.max_clips, args.overlap)

if __name__ == "__main__":
    main()
