#!/usr/bin/env python3
"""
Batch processing script for TED Talk Processor
Reads URLs from a file and processes them in batches.
"""

import sys
import argparse
from pathlib import Path
from ted_processor import TEDTalkProcessor

def read_urls_from_file(filename: str) -> list:
    """Read URLs from a text file."""
    urls = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    urls.append(line)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    
    return urls

def main():
    """Main function for batch processing."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Batch process TED Talk videos to extract speaker-only clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process.py urls.txt
  python batch_process.py urls.txt --output-dir my_clips --max-clips 20
  python batch_process.py urls.txt --no-progress
        """
    )
    
    parser.add_argument('urls_file', help='Text file containing YouTube URLs (one per line)')
    parser.add_argument('--output-dir', default='batch_output', help='Directory to save clips (default: batch_output)')
    parser.add_argument('--max-clips', type=int, default=30, help='Maximum clips per video (default: 30)')
    parser.add_argument('--no-progress', action='store_true', help='Disable verbose progress tracking')
    
    args = parser.parse_args()
    
    urls_file = args.urls_file
    output_dir = args.output_dir
    max_clips = args.max_clips
    verbose_progress = not args.no_progress
    
    # Read URLs from file
    urls = read_urls_from_file(urls_file)
    
    if not urls:
        print("No URLs found in file. Please check the file format.")
        sys.exit(1)
    
    print(f"Found {len(urls)} URLs to process")
    print(f"Output directory: {output_dir}")
    print(f"Max clips per video: {max_clips}")
    print(f"Progress tracking: {'Enabled' if verbose_progress else 'Disabled'}")
    
    # Initialize processor
    processor = TEDTalkProcessor(output_dir=output_dir)
    
    # Process videos
    print("\nStarting batch processing...")
    results = processor.process_multiple_videos(urls, max_clips_per_video=max_clips, verbose_progress=verbose_progress)
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    
    total_clips = 0
    successful_videos = 0
    
    for url, clip_paths in results.items():
        print(f"\nVideo: {url}")
        print(f"Clips extracted: {len(clip_paths)}")
        
        if clip_paths:
            successful_videos += 1
            total_clips += len(clip_paths)
        
        for clip_path in clip_paths:
            print(f"  - {Path(clip_path).name}")
    
    print(f"\nProcessing complete!")
    print(f"Successful videos: {successful_videos}/{len(urls)}")
    print(f"Total clips extracted: {total_clips}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
