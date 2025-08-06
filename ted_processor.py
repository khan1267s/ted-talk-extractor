#!/usr/bin/env python3
"""
TED Talk Processor Core Logic
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import yt_dlp
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
from progress_tracker import ProgressTracker, MultiStageProgressTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define absolute paths
APP_ROOT = Path(__file__).parent.resolve()
DOWNLOADS_DIR = APP_ROOT / "downloads"
DEFAULT_OUTPUT_DIR = APP_ROOT / "output_clips"
MODEL_PATH = APP_ROOT / "yolov8n.pt"

# Create directories
DOWNLOADS_DIR.mkdir(exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

class TEDTalkProcessor:
    def __init__(self, output_dir: str = str(DEFAULT_OUTPUT_DIR), model_path: str = str(MODEL_PATH)):
        """
        Initialize the TED Talk Processor.
        Args:
            output_dir (str): Directory to save the output clips.
            model_path (str): Path to the YOLOv8 model file.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the YOLOv8 model."""
        if not Path(model_path).exists():
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        try:
            return YOLO(model_path)
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise

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
        Check for a single person using YOLOv8.
        A frame is considered a 'speaker frame' if it contains exactly one person
        and the person occupies a significant portion of the frame.
        """
        try:
            results = self.model(frame, classes=[0], verbose=False) # Class 0 is 'person' in COCO
            
            # We are looking for exactly one person
            if len(results) == 0 or len(results[0].boxes) != 1:
                return False

            box = results[0].boxes[0].xywh[0]
            _, _, w, h = box
            
            frame_height, frame_width, _ = frame.shape
            box_area = w * h
            frame_area = frame_width * frame_height

            # The person should occupy a certain percentage of the frame to be considered the speaker
            # This helps filter out audience members or distant figures
            area_ratio = box_area / frame_area
            if area_ratio < 0.1 or area_ratio > 0.9:
                return False

            return True

        except Exception as e:
            logger.debug(f"Frame check failed: {e}")
            return False

    def find_speaker_segments(self, video_path: str, progress_callback=None, verbose_progress: bool = True) -> List[Tuple[float, float]]:
        """Find speaker-only segments."""
        logger.info(f"Finding speaker segments in: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        logger.info(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}, Total frames: {total_frames}")

        sample_interval_seconds = 1
        frame_interval = int(fps * sample_interval_seconds)
        good_frames = []
        
        # Create progress tracker for frame analysis
        sampled_frames = list(range(0, total_frames, frame_interval))
        frame_tracker = ProgressTracker(
            len(sampled_frames), 
            "Analyzing frames for speaker detection",
            verbose=verbose_progress
        )
        frame_tracker.start()

        for i, frame_idx in enumerate(sampled_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.check_speaker_frame(frame):
                good_frames.append(frame_idx)
            
            # Update progress tracker
            time_position = frame_idx / fps
            frame_tracker.update(i, f"Frame {frame_idx} ({time_position:.1f}s)")
            
            if progress_callback:
                progress = int((frame_idx / total_frames) * 100)
                progress_callback(progress)
        
        cap.release()
        frame_tracker.complete(f"Found {len(good_frames)} frames with speaker")
        
        if progress_callback:
            progress_callback(100)

        segments = self.frames_to_segments(good_frames, fps, duration)
        logger.info(f"Found {len(segments)} speaker segments")
        return segments

    def frames_to_segments(self, good_frames: List[int], fps: float, duration: float) -> List[Tuple[float, float]]:
        """Convert good frame numbers to time segments, tolerating gaps."""
        if not good_frames:
            return []

        segments = []
        start_frame = good_frames[0]
        max_gap_seconds = 8
        max_gap_frames = int(fps * max_gap_seconds)
        min_segment_duration = 20

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

    def extract_clips(self, video_path: str, segments: List[Tuple[float, float]], max_clips: int = 5, verbose_progress: bool = True) -> List[str]:
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
        
        # Create progress tracker for clip extraction
        clip_tracker = ProgressTracker(
            max_clips,
            "Extracting speaker clips",
            verbose=verbose_progress
        )
        clip_tracker.start()

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
                clip_filename = f"{video_name}_speaker_clip_{len(clip_paths)+1:03d}.mp4"
                clip_path = self.output_dir / clip_filename
                
                # Update progress before extraction
                clip_tracker.update(len(clip_paths), f"Extracting {clip_filename}")
                
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    preset='fast',
                    threads=4,
                    logger=None
                )
                
                clip_paths.append(str(clip_path))
                used_time_ranges.append((clip_start, clip_start + clip_duration))
                logger.info(f"Extracted clip {len(clip_paths)}: {clip_filename}")

            except Exception as e:
                logger.error(f"Error extracting clip {i+1}: {e}")

        video.close()
        clip_tracker.complete(f"Extracted {len(clip_paths)} clips successfully")
        return clip_paths

    def process_video(self, video_path: str, max_clips: int = 5, from_url: bool = False, progress_callback=None, verbose_progress: bool = True) -> List[str]:
        """
        Full processing pipeline for a single video file.
        """
        # Setup multi-stage progress tracking
        stages = [
            {"name": "Download Video" if from_url else "Load Video"},
            {"name": "Analyze Frames"},
            {"name": "Extract Clips"}
        ]
        
        multi_tracker = MultiStageProgressTracker(stages, verbose=verbose_progress)
        multi_tracker.start()
        
        # Stage 1: Download/Load video
        multi_tracker.next_stage()
        if from_url:
            actual_video_path = self.download_video(video_path)
            if not actual_video_path:
                logger.error("Failed to download video")
                return []
        else:
            actual_video_path = video_path

        # Stage 2: Find speaker segments
        multi_tracker.next_stage()
        logger.info(f"Processing video: {actual_video_path}")
        segments = self.find_speaker_segments(actual_video_path, progress_callback, verbose_progress)
        if not segments:
            logger.warning("No valid speaker segments found")
            if from_url:
                os.remove(actual_video_path)
            return []
        
        # Stage 3: Extract clips
        multi_tracker.next_stage()
        clip_paths = self.extract_clips(actual_video_path, segments, max_clips, verbose_progress)
        
        if from_url:
            try:
                os.remove(actual_video_path)
                logger.info(f"Cleaned up downloaded video: {actual_video_path}")
            except Exception as e:
                logger.warning(f"Could not clean up video file: {e}")
        
        multi_tracker.complete()
        return clip_paths

    def process_multiple_videos(self, urls: List[str], max_clips_per_video: int = 5, verbose_progress: bool = True) -> Dict[str, List[str]]:
        """
        Process a batch of videos from a list of URLs.
        """
        results = {}
        batch_tracker = ProgressTracker(len(urls), "Processing videos", verbose=verbose_progress)
        batch_tracker.start()
        
        for i, url in enumerate(urls):
            logger.info(f"\nProcessing URL {i+1}/{len(urls)}: {url}")
            batch_tracker.update(i, f"Processing: {url}")
            try:
                clip_paths = self.process_video(url, max_clips=max_clips_per_video, from_url=True, verbose_progress=verbose_progress)
                results[url] = clip_paths
            except Exception as e:
                logger.error(f"Failed to process video {url}: {e}")
                results[url] = []
        
        batch_tracker.complete(f"Processed {len(urls)} videos")
        return results
