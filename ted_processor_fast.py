#!/usr/bin/env python3
"""
Fast TED Talk Processor - Speaker Only (No Slides/Images)
Quick and efficient extraction of speaker-only clips.
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

class FastTEDTalkProcessor:
    def __init__(self, output_dir: str = "output_clips"):
        """
        Initialize the fast TED Talk processor.
        
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
    
    def quick_speaker_check(self, frame: np.ndarray) -> bool:
        """
        Fast check for speaker-only frames.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame contains only speaker
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Quick person detection
            pose_results = self.pose.process(rgb_frame)
            face_results = self.face_detection.process(rgb_frame)
            
            person_count = 0
            if pose_results.pose_landmarks:
                person_count += 1
            if face_results.detections:
                person_count += len(face_results.detections)
            
            # If exactly one person detected
            if person_count == 1:
                # Quick slide/text detection using edge density
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # If edge density is low (no slides/text), consider it speaker-only
                if edge_density < 0.08:  # Lower threshold for faster processing
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Quick check failed: {e}")
            return False
    
    def find_speaker_segments(self, video_path: str, segment_duration: int = 30) -> List[Tuple[float, float]]:
        """
        Quickly find speaker-only segments.
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of (start_time, end_time) tuples for valid segments
        """
        logger.info(f"Quickly analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video duration: {duration:.2f} seconds, FPS: {fps:.2f}")
        
        valid_segments = []
        
        # Sample fewer frames for faster processing
        for start_time in range(0, int(duration), segment_duration):
            end_time = min(start_time + segment_duration, duration)
            
            # Sample only 5 frames per segment for speed
            sample_times = np.linspace(start_time, end_time, 5)
            speaker_frames = 0
            
            for sample_time in sample_times:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_time * fps))
                ret, frame = cap.read()
                
                if ret and self.quick_speaker_check(frame):
                    speaker_frames += 1
            
            # If 80% of sampled frames are speaker-only, consider segment valid
            if speaker_frames >= 4:  # 4 out of 5 frames
                valid_segments.append((start_time, end_time))
                logger.info(f"Valid speaker segment found: {start_time:.1f}s - {end_time:.1f}s")
        
        cap.release()
        logger.info(f"Found {len(valid_segments)} valid speaker segments")
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
                clip_filename = f"{video_name}_speaker_only_{i+1:03d}.mp4"
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
        
        # Find speaker-only segments
        segments = self.find_speaker_segments(video_path)
        
        if not segments:
            logger.warning("No valid speaker-only segments found")
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
    
    def process_multiple_videos(self, urls: List[str], max_clips_per_video: int = 5) -> dict:
        """
        Process multiple videos.
        
        Args:
            urls: List of YouTube video URLs
            max_clips_per_video: Maximum clips per video (default 5)
            
        Returns:
            Dictionary mapping URLs to lists of clip paths
        """
        results = {}
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing video {i}/{len(urls)}: {url}")
            
            try:
                clip_paths = self.process_video(url, max_clips_per_video)
                results[url] = clip_paths
                
                logger.info(f"Successfully processed video {i}: {len(clip_paths)} clips extracted")
                
            except Exception as e:
                logger.error(f"Error processing video {i}: {e}")
                results[url] = []
        
        return results

def main():
    """Main function to run the fast TED Talk processor."""
    parser = argparse.ArgumentParser(description="Fast TED Talk Processor - Extract speaker-only clips")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs to process")
    parser.add_argument("--output-dir", default="output_clips", help="Output directory for clips")
    parser.add_argument("--max-clips", type=int, default=5, help="Maximum number of clips per video (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FastTEDTalkProcessor(args.output_dir)
    
    # Process videos
    results = processor.process_multiple_videos(args.urls, args.max_clips)
    
    # Print summary
    print("\n" + "="*50)
    print("FAST PROCESSING SUMMARY")
    print("="*50)
    
    total_clips = 0
    for url, clip_paths in results.items():
        print(f"\nVideo: {url}")
        print(f"Speaker-only clips extracted: {len(clip_paths)}")
        total_clips += len(clip_paths)
        
        for clip_path in clip_paths:
            print(f"  - {Path(clip_path).name}")
    
    print(f"\nTotal speaker-only clips extracted: {total_clips}")
    print(f"Output directory: {args.output_dir}")
    print("\nNote: These clips contain the speaker without slides/images")

if __name__ == "__main__":
    main() 