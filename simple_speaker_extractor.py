#!/usr/bin/env python3
"""
Simple Speaker Extractor - Fast and Reliable
Extracts speaker-only clips from TED Talks quickly.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from tqdm import tqdm
import logging

# Video processing
import yt_dlp
from moviepy.editor import VideoFileClip

# Person detection
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSpeakerExtractor:
    def __init__(self, output_dir: str = "output_clips"):
        """
        Initialize the simple speaker extractor.
        
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
    
    def download_video(self, url: str) -> Optional[str]:
        """Download video from YouTube URL using yt-dlp."""
        try:
            logger.info(f"Downloading video from: {url}")
            
            downloads_dir = Path("downloads")
            downloads_dir.mkdir(exist_ok=True)
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': str(downloads_dir / '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_files = list(downloads_dir.glob("*.mp4"))
                if downloaded_files:
                    video_path = downloaded_files[-1]
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
        Simple check for speaker-only frames.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame contains only speaker
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Check for person detection
            pose_results = self.pose.process(rgb_frame)
            face_results = self.face_detection.process(rgb_frame)
            
            person_count = 0
            if pose_results.pose_landmarks:
                person_count += 1
            if face_results.detections:
                person_count += len(face_results.detections)
            
            # If exactly one person detected
            if person_count == 1:
                # Simple check for slides/text using edge density
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # If edge density is low (no slides/text), consider it speaker-only
                if edge_density < 0.1:
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
        
        valid_segments = []
        current_start = None
        segment_duration = 30  # 30 seconds per segment
        
        # Sample frames every 2 seconds for efficiency
        sample_interval = 2
        frame_interval = int(fps * sample_interval)
        
        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            current_time = frame_idx / fps
            
            # Check if this frame is speaker-only
            is_speaker = self.check_speaker_frame(frame)
            
            if is_speaker:
                if current_start is None:
                    current_start = current_time
            else:
                # If we have a valid segment, add it
                if current_start is not None:
                    segment_end = current_time
                    segment_duration_actual = segment_end - current_start
                    
                    if segment_duration_actual >= 20:  # At least 20 seconds
                        valid_segments.append((current_start, segment_end))
                        logger.info(f"Found speaker segment: {current_start:.1f}s - {segment_end:.1f}s")
                    
                    current_start = None
        
        # Handle the last segment
        if current_start is not None:
            segment_end = duration
            segment_duration_actual = segment_end - current_start
            if segment_duration_actual >= 20:
                valid_segments.append((current_start, segment_end))
                logger.info(f"Found final speaker segment: {current_start:.1f}s - {segment_end:.1f}s")
        
        cap.release()
        logger.info(f"Found {len(valid_segments)} speaker segments")
        return valid_segments
    
    def extract_clips(self, video_path: str, segments: List[Tuple[float, float]], max_clips: int = 5) -> List[str]:
        """
        Extract clips from video based on valid segments.
        
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
        
        # Limit to max_clips
        segments = segments[:max_clips]
        
        for i, (start_time, end_time) in enumerate(tqdm(segments, desc="Extracting clips")):
            try:
                # Extract clip
                clip = video.subclip(start_time, end_time)
                
                # Generate output filename
                video_name = Path(video_path).stem
                clip_filename = f"{video_name}_speaker_{i+1:03d}.mp4"
                clip_path = self.output_dir / clip_filename
                
                # Write clip
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                
                clip_paths.append(str(clip_path))
                logger.info(f"Extracted clip {i+1}: {clip_filename}")
                
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

def main():
    """Main function to run the simple speaker extractor."""
    parser = argparse.ArgumentParser(description="Simple Speaker Extractor - Extract speaker-only clips")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs to process")
    parser.add_argument("--output-dir", default="output_clips", help="Output directory for clips")
    parser.add_argument("--max-clips", type=int, default=5, help="Maximum number of clips per video (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SimpleSpeakerExtractor(args.output_dir)
    
    # Process videos
    for i, url in enumerate(args.urls, 1):
        logger.info(f"Processing video {i}/{len(args.urls)}: {url}")
        
        try:
            clip_paths = extractor.process_video(url, args.max_clips)
            
            print(f"\nVideo {i}: {url}")
            print(f"Speaker clips extracted: {len(clip_paths)}")
            
            for clip_path in clip_paths:
                print(f"  - {Path(clip_path).name}")
            
        except Exception as e:
            logger.error(f"Error processing video {i}: {e}")
    
    print(f"\nOutput directory: {args.output_dir}")

if __name__ == "__main__":
    main() 