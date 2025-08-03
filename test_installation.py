#!/usr/bin/env python3
"""
Test script to verify TED Talk Processor installation
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'cv2',
        'numpy',
        'pytube',
        'moviepy',
        'mediapipe',
        'ultralytics',
        'tqdm',
        'requests'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_yolo_model():
    """Test if YOLO model can be loaded."""
    try:
        from ultralytics import YOLO
        print("\nTesting YOLO model loading...")
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        print("This might be due to network issues or missing dependencies.")
        return False

def test_mediapipe():
    """Test MediaPipe initialization."""
    try:
        import mediapipe as mp
        print("\nTesting MediaPipe...")
        
        # Test pose detection
        pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Test face detection
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        print("✅ MediaPipe initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize MediaPipe: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality."""
    try:
        import cv2
        import numpy as np
        print("\nTesting OpenCV...")
        
        # Test basic OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        print("✅ OpenCV working correctly!")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("TED TALK PROCESSOR - INSTALLATION TEST")
    print("="*50)
    
    # Import numpy for tests
    try:
        import numpy as np
    except ImportError:
        print("❌ NumPy is required but not installed!")
        return False
    
    # Run tests
    tests = [
        test_imports,
        test_yolo_model,
        test_mediapipe,
        test_opencv
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Installation is complete and ready to use!")
        print("\nYou can now run:")
        print("  python ted_processor.py \"https://www.youtube.com/watch?v=VIDEO_ID\"")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTry running: pip install -r requirements.txt")
    
    print("="*50)

if __name__ == "__main__":
    main() 