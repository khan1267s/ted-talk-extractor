#!/usr/bin/env python3
"""
Command-line interface for TED Talk Processor with enhanced progress tracking
Process single TED Talk videos to extract speaker-only clips.
Supports both uploaded video files and YouTube URLs.
"""

import sys
import os
import argparse
from pathlib import Path
from ted_processor import TEDTalkProcessor


def main():
    """Main function for single video processing."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process a TED Talk video to extract speaker-only clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process an uploaded video file
  python process_ted_talk.py video.mp4
  python process_ted_talk.py /path/to/uploaded_video.mp4 --max-clips 20
  
  # Process from YouTube URL
  python process_ted_talk.py "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Disable progress tracking
  python process_ted_talk.py video.mp4 --no-progress
        """
    )
    
    parser.add_argument('video', help='Local video file path or YouTube URL')
    parser.add_argument('--output-dir', default='output_clips', help='Directory to save clips (default: output_clips)')
    parser.add_argument('--max-clips', type=int, default=30, help='Maximum number of clips to extract (default: 30)')
    parser.add_argument('--no-progress', action='store_true', help='Disable verbose progress tracking')
    
    args = parser.parse_args()
    
    # Determine if input is URL or file
    is_url = args.video.startswith(('http://', 'https://', 'www.'))
    
    # Validate local file exists if not URL
    if not is_url:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        if not video_path.is_file():
            print(f"Error: Path is not a file: {args.video}")
            sys.exit(1)
        # Use absolute path for consistency
        args.video = str(video_path.absolute())
    
    # Initialize processor
    processor = TEDTalkProcessor(output_dir=args.output_dir)
    
    # Print processing info
    if is_url:
        print(f"Processing YouTube video: {args.video}")
    else:
        print(f"Processing uploaded video: {Path(args.video).name}")
        print(f"File size: {os.path.getsize(args.video) / (1024*1024):.1f} MB")
    print(f"Output directory: {args.output_dir}")
    print(f"Max clips: {args.max_clips}")
    print(f"Progress tracking: {'Enabled' if not args.no_progress else 'Disabled'}")
    print()
    
    try:
        # Process video
        clip_paths = processor.process_video(
            args.video, 
            max_clips=args.max_clips, 
            from_url=is_url,
            verbose_progress=not args.no_progress
        )
        
        # Print results
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        
        if clip_paths:
            print(f"\nSuccessfully extracted {len(clip_paths)} clips:")
            for clip_path in clip_paths:
                print(f"  - {Path(clip_path).name}")
        else:
            print("\nNo clips were extracted. This could be due to:")
            print("  - No clear speaker-only segments found")
            print("  - Too many audience shots or multiple people")
            print("  - Video quality or detection issues")
        
        print(f"\nOutput directory: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()