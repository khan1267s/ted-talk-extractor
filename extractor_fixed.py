#!/usr/bin/env python3
"""
Fixed TED-Talk speaker extractor - Fast and Reliable
Downloads TED Talks and extracts speaker-only clips quickly.
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

class FixedSpeakerExtractor:
    def __init__(self, output_dir: str = "output_clips"):
        """
        Initialize the fixed speaker extractor.
        
        Args:
            output_dir: Directory to save extracted clips
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe once
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
    
    def download_video(self, url: str) -> Optional[str]:
        """Download video from YouTube URL using yt-dlp."""
        try:
            logger.info(f"Downloading video from: {url}")
            
            downloads_dir = Path("downloads")
            downloads_dir.mkdir(exist_ok=True)
            
            # Use fixed filename to avoid re-downloading
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': str(downloads_dir / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
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
            # Resize for speed
            frame = cv2.resize(frame, (320, 180))
            
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
        
        # Sample frames every 4 seconds for efficiency
        sample_interval = 4
        frame_interval = int(fps * sample_interval)
        
        good_frames = []
        
        # Scan frames with progress bar
        for frame_idx in tqdm(range(0, total_frames, frame_interval), desc="Scanning frames"):
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
        if good_frames:
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
        
        for i, (start_time, end_time) in enumerate(tqdm(segments, desc="Extracting clips")):
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
                
                # Write clip
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    verbose=False,
                    logger=None
                )
                
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

def main():
    """Main function to run the fixed speaker extractor."""
    parser = argparse.ArgumentParser(description="Fixed Speaker Extractor - Extract speaker-only clips")
    parser.add_argument("url", help="YouTube video URL to process")
    parser.add_argument("--output-dir", default="output_clips", help="Output directory for clips")
    parser.add_argument("--max-clips", type=int, default=5, help="Maximum number of clips per video (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = FixedSpeakerExtractor(args.output_dir)
    
    # Process video
    try:
        clip_paths = extractor.process_video(args.url, args.max_clips)
        
        print(f"\nVideo: {args.url}")
        print(f"Speaker clips extracted: {len(clip_paths)}")
        
        for clip_path in clip_paths:
            print(f"  - {Path(clip_path).name}")
        
        print(f"\nOutput directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")

if __name__ == "__main__":
    main() 