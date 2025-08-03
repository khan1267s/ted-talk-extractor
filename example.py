#!/usr/bin/env python3
"""
Example usage of the TED Talk Processor
"""

from ted_processor import TEDTalkProcessor

def main():
    """Example of how to use the TED Talk processor programmatically."""
    
    # Initialize the processor
    processor = TEDTalkProcessor(output_dir="example_clips")
    
    # Example TED Talk URLs (popular TED Talks)
    ted_talks = [
        "https://www.youtube.com/watch?v=u4ZoJKF_VuA",  # Simon Sinek - Start With Why
        "https://www.youtube.com/watch?v=Ks-_Mh1QhMc",  # Amy Cuddy - Power Poses
    ]
    
    print("Starting TED Talk processing...")
    print(f"Processing {len(ted_talks)} videos...")
    
    # Process the videos
    results = processor.process_multiple_videos(ted_talks, max_clips_per_video=10)
    
    # Print results
    print("\n" + "="*50)
    print("PROCESSING RESULTS")
    print("="*50)
    
    total_clips = 0
    for url, clip_paths in results.items():
        print(f"\nVideo: {url}")
        print(f"Clips extracted: {len(clip_paths)}")
        total_clips += len(clip_paths)
        
        for clip_path in clip_paths:
            print(f"  - {clip_path}")
    
    print(f"\nTotal clips extracted: {total_clips}")
    print("Check the 'example_clips' directory for the extracted clips!")

if __name__ == "__main__":
    main() 