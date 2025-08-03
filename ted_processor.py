#!/usr/bin/env python3
"""
TED Talk Processor
Downloads TED Talks from YouTube and extracts speaker-only clips.
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
from pytube import YouTube
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Person detection
import mediapipe as mp
from ultralytics import YOLO

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
        
        # Initialize YOLO for person detection
        try:
            # Fix for PyTorch 2.6+ compatibility
            import torch
            torch.hub.set_dir('./models')  # Set model directory
            # Use weights_only=False for compatibility
            self.yolo_model = YOLO('yolov8n.pt')
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}")
            logger.info("Continuing with MediaPipe detection only")
            self.yolo_model = None
    
    def download_video(self, url: str) -> Optional[str]:
        """
        Download video from YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Path to downloaded video file or None if failed
        """
        try:
            logger.info(f"Downloading video from: {url}")
            yt = YouTube(url)
            
            # Get the best quality stream
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not stream:
                logger.error("No suitable video stream found")
                return None
            
            # Create downloads directory
            downloads_dir = Path("downloads")
            downloads_dir.mkdir(exist_ok=True)
            
            # Download video
            video_path = downloads_dir / f"{yt.title.replace(' ', '_')}.mp4"
            stream.download(output_path=downloads_dir, filename=video_path.name)
            
            logger.info(f"Video downloaded successfully: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    def detect_persons_in_frame(self, frame: np.ndarray) -> Tuple[int, List[dict]]:
        """
        Detect persons in a frame using multiple methods.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (person_count, detection_info)
        """
        person_count = 0
        detection_info = []
        
        # Method 1: YOLO detection
        if self.yolo_model:
            try:
                results = self.yolo_model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if box.cls == 0:  # person class in COCO
                                person_count += 1
                                detection_info.append({
                                    'method': 'yolo',
                                    'confidence': float(box.conf),
                                    'bbox': box.xyxy[0].cpu().numpy()
                                })
            except Exception as e:
                logger.debug(f"YOLO detection failed: {e}")
        
        # Method 2: MediaPipe pose detection
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                person_count += 1
                detection_info.append({
                    'method': 'mediapipe_pose',
                    'confidence': 0.8,  # MediaPipe doesn't provide confidence
                    'landmarks': pose_results.pose_landmarks
                })
        except Exception as e:
            logger.debug(f"MediaPipe pose detection failed: {e}")
        
        # Method 3: MediaPipe face detection
        try:
            face_results = self.face_detection.process(rgb_frame)
            if face_results.detections:
                for detection in face_results.detections:
                    person_count += 1
                    detection_info.append({
                        'method': 'mediapipe_face',
                        'confidence': detection.score[0],
                        'bbox': detection.location_data.relative_bounding_box
                    })
        except Exception as e:
            logger.debug(f"MediaPipe face detection failed: {e}")
        
        return person_count, detection_info
    
    def is_speaker_only_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame contains only the main speaker.
        
        Args:
            frame: Input frame
            
        Returns:
            True if frame contains only one person (speaker)
        """
        person_count, detection_info = self.detect_persons_in_frame(frame)
        
        # We want exactly one person in the frame
        if person_count == 1:
            # Additional checks for speaker quality
            for info in detection_info:
                if info['confidence'] > 0.5:  # Good confidence
                    return True
        
        return False
    
    def analyze_video_segments(self, video_path: str, segment_duration: int = 30) -> List[Tuple[float, float]]:
        """
        Analyze video to find speaker-only segments.
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of (start_time, end_time) tuples for valid segments
        """
        logger.info(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video duration: {duration:.2f} seconds, FPS: {fps:.2f}")
        
        valid_segments = []
        frames_per_segment = int(fps * segment_duration)
        segments_checked = 0
        
        # Sample frames from each segment
        for start_time in range(0, int(duration), segment_duration):
            end_time = min(start_time + segment_duration, duration)
            segment_frames = []
            
            # Sample frames from this segment
            sample_times = np.linspace(start_time, end_time, 10)  # Sample 10 frames per segment
            
            for sample_time in sample_times:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_time * fps))
                ret, frame = cap.read()
                
                if ret:
                    segment_frames.append(frame)
            
            # Check if segment is speaker-only
            speaker_only_count = 0
            for frame in segment_frames:
                if self.is_speaker_only_frame(frame):
                    speaker_only_count += 1
            
            # If more than 70% of sampled frames are speaker-only, consider it valid
            if len(segment_frames) > 0 and speaker_only_count / len(segment_frames) > 0.7:
                valid_segments.append((start_time, end_time))
                logger.info(f"Valid segment found: {start_time:.1f}s - {end_time:.1f}s")
            
            segments_checked += 1
            if segments_checked % 10 == 0:
                logger.info(f"Processed {segments_checked} segments...")
        
        cap.release()
        logger.info(f"Found {len(valid_segments)} valid speaker-only segments")
        return valid_segments
    
    def extract_clips(self, video_path: str, segments: List[Tuple[float, float]], max_clips: int = 30) -> List[str]:
        """
        Extract clips from video based on valid segments.
        
        Args:
            video_path: Path to video file
            segments: List of (start_time, end_time) tuples
            max_clips: Maximum number of clips to extract
            
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
                clip_filename = f"{video_name}_clip_{i+1:03d}.mp4"
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
    
    def process_video(self, url: str, max_clips: int = 30) -> List[str]:
        """
        Process a single video: download, analyze, and extract clips.
        
        Args:
            url: YouTube video URL
            max_clips: Maximum number of clips to extract
            
        Returns:
            List of paths to extracted clip files
        """
        logger.info(f"Processing video: {url}")
        
        # Download video
        video_path = self.download_video(url)
        if not video_path:
            logger.error("Failed to download video")
            return []
        
        # Analyze video for speaker-only segments
        segments = self.analyze_video_segments(video_path)
        
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
    
    def process_multiple_videos(self, urls: List[str], max_clips_per_video: int = 30) -> dict:
        """
        Process multiple videos.
        
        Args:
            urls: List of YouTube video URLs
            max_clips_per_video: Maximum clips per video
            
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
    """Main function to run the TED Talk processor."""
    parser = argparse.ArgumentParser(description="TED Talk Processor - Extract speaker-only clips from YouTube videos")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs to process")
    parser.add_argument("--output-dir", default="output_clips", help="Output directory for clips")
    parser.add_argument("--max-clips", type=int, default=30, help="Maximum number of clips per video")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = TEDTalkProcessor(args.output_dir)
    
    # Process videos
    results = processor.process_multiple_videos(args.urls, args.max_clips)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    total_clips = 0
    for url, clip_paths in results.items():
        print(f"\nVideo: {url}")
        print(f"Clips extracted: {len(clip_paths)}")
        total_clips += len(clip_paths)
        
        for clip_path in clip_paths:
            print(f"  - {Path(clip_path).name}")
    
    print(f"\nTotal clips extracted: {total_clips}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 