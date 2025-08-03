#!/usr/bin/env python3
"""
Final TED Talk Processor - Speaker Only (No Slides/Images)
Downloads TED Talks and extracts ONLY speaker-on-stage clips.
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

class FinalTEDTalkProcessor:
    def __init__(self, output_dir: str = "output_clips"):
        """
        Initialize the final TED Talk processor.
        
        Args:
            output_dir: Directory to save extracted clips
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe for person detection
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
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
    
    def detect_speaker_only(self, frame: np.ndarray) -> Tuple[bool, dict]:
        """
        Advanced detection for speaker-only frames (no slides, images, text).
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (is_speaker_only, detection_info)
        """
        detection_info = {
            'pose_detected': False,
            'face_detected': False,
            'person_count': 0,
            'confidence': 0.0,
            'has_slides': False,
            'has_text': False,
            'has_images': False
        }
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Method 1: MediaPipe pose detection
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                detection_info['pose_detected'] = True
                detection_info['person_count'] += 1
                detection_info['confidence'] += 0.3
            
            # Method 2: MediaPipe face detection
            face_results = self.face_detection.process(rgb_frame)
            if face_results.detections:
                for detection in face_results.detections:
                    detection_info['face_detected'] = True
                    detection_info['person_count'] += 1
                    detection_info['confidence'] += 0.2
            
            # Advanced slide/image/text detection
            if detection_info['person_count'] == 1:
                # Convert to different color spaces for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # 1. Edge detection for text/slides
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # 2. Contour analysis for rectangular elements (slides)
                _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                large_rectangles = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Large enough to be a slide
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio for slides
                            large_rectangles += 1
                
                # 3. Text detection using morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilated = cv2.dilate(edges, kernel, iterations=1)
                text_regions = cv2.erode(dilated, kernel, iterations=1)
                text_density = np.sum(text_regions > 0) / (text_regions.shape[0] * text_regions.shape[1])
                
                # 4. Color analysis for images/slides
                saturation = hsv[:, :, 1]
                saturation_std = np.std(saturation)
                brightness = hsv[:, :, 2]
                brightness_std = np.std(brightness)
                
                # 5. Contrast analysis
                contrast = np.std(gray)
                
                # Determine if frame has unwanted elements
                has_slides = (edge_density > 0.12 or  # Too many edges
                             large_rectangles > 1 or   # Too many rectangular elements
                             text_density > 0.08)     # Too much text
                
                has_text = text_density > 0.05
                has_images = (saturation_std > 40 or brightness_std > 50)  # Too much color variation
                
                detection_info['has_slides'] = has_slides
                detection_info['has_text'] = has_text
                detection_info['has_images'] = has_images
                
                # Bonus for clean speaker shots
                if not has_slides and not has_text and not has_images:
                    detection_info['confidence'] += 0.4
                    
                    # Additional check for speaker position (should be in center)
                    if pose_results.pose_landmarks:
                        landmarks = pose_results.pose_landmarks.landmark
                        if landmarks:
                            nose = landmarks[0]  # Nose landmark
                            if 0.3 < nose.x < 0.7 and 0.2 < nose.y < 0.8:  # Center area
                                detection_info['confidence'] += 0.2
            
        except Exception as e:
            logger.debug(f"Detection failed: {e}")
        
        # Final criteria: exactly 1 person, no slides/text/images, good confidence
        is_speaker_only = (detection_info['person_count'] == 1 and 
                          not detection_info['has_slides'] and
                          not detection_info['has_text'] and
                          not detection_info['has_images'] and
                          detection_info['confidence'] >= 0.7)
        
        return is_speaker_only, detection_info
    
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
            total_confidence = 0.0
            
            for frame in segment_frames:
                is_speaker, detection_info = self.detect_speaker_only(frame)
                if is_speaker:
                    speaker_only_count += 1
                    total_confidence += detection_info['confidence']
            
            # Criteria: 80% of frames must be speaker-only
            if (len(segment_frames) > 0 and 
                speaker_only_count / len(segment_frames) > 0.8):
                valid_segments.append((start_time, end_time))
                logger.info(f"Valid speaker-only segment found: {start_time:.1f}s - {end_time:.1f}s")
            
            segments_checked += 1
            if segments_checked % 5 == 0:
                logger.info(f"Processed {segments_checked} segments...")
        
        cap.release()
        logger.info(f"Found {len(valid_segments)} valid speaker-only segments")
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
    """Main function to run the final TED Talk processor."""
    parser = argparse.ArgumentParser(description="Final TED Talk Processor - Extract speaker-only clips (no slides/images/text)")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs to process")
    parser.add_argument("--output-dir", default="output_clips", help="Output directory for clips")
    parser.add_argument("--max-clips", type=int, default=5, help="Maximum number of clips per video (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FinalTEDTalkProcessor(args.output_dir)
    
    # Process videos
    results = processor.process_multiple_videos(args.urls, args.max_clips)
    
    # Print summary
    print("\n" + "="*50)
    print("FINAL PROCESSING SUMMARY")
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
    print("\nNote: These clips contain ONLY the speaker on stage (no slides/images/text)")

if __name__ == "__main__":
    main() 