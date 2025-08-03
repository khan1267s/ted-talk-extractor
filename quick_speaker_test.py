#!/usr/bin/env python3
"""
Quick test to find speaker-only segments
"""

import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

def test_speaker_detection():
    """Test speaker detection on the existing video."""
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_face = mp.solutions.face_detection
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    face_detection = mp_face.FaceDetection(
        model_selection=1, min_detection_confidence=0.6
    )
    
    # Check if we have the downloaded video
    video_path = Path("downloads/The next outbreak？ We're not ready ｜ Bill Gates ｜ TED.mp4")
    
    if not video_path.exists():
        print("Video not found. Please run the download first.")
        return
    
    print(f"Testing speaker detection on: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video duration: {duration:.2f} seconds")
    
    # Test frames at different timestamps
    test_times = [30, 60, 120, 180, 240, 300, 360, 420, 480]  # Test at 30s intervals
    
    for test_time in test_times:
        if test_time > duration:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(test_time * fps))
        ret, frame = cap.read()
        
        if ret:
            # Detect person
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pose_results = pose.process(rgb_frame)
            face_results = face_detection.process(rgb_frame)
            
            person_count = 0
            if pose_results.pose_landmarks:
                person_count += 1
            if face_results.detections:
                person_count += len(face_results.detections)
            
            # Check for slides (simple edge detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            has_slides = edge_density > 0.1
            
            print(f"Time {test_time}s: {person_count} person(s), slides: {has_slides}, edge_density: {edge_density:.3f}")
    
    cap.release()

if __name__ == "__main__":
    test_speaker_detection() 