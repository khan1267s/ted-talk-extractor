#!/usr/bin/env python3
"""
TED Talk Processing Pipeline
Processes a list of TED Talk URLs from a file to extract speaker-only clips.
"""

import logging
from ted_processor import TEDTalkProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os

# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_urls_from_file(filename="sample_urls.txt"):
    """
    Loads a list of URLs from a text file located in the same directory as the script.
    """
    filepath = os.path.join(script_dir, filename)
    logger.info(f"Loading video URLs from {filepath}...")
    try:
        with open(filepath, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        logger.info(f"Found {len(urls)} URLs to process.")
        return urls
    except FileNotFoundError:
        logger.error(f"Could not find the URL file: {filepath}")
        return []

def run_pipeline():
    """
    Main pipeline to process TED talks from a file.
    """
    # 1. Load video URLs
    video_urls = load_urls_from_file()
    
    if not video_urls:
        logger.error("No videos to process. Exiting.")
        return

    # 2. Initialize the processor
    processor = TEDTalkProcessor(output_dir="output")
    
    # 3. Process videos one by one
    processed_talks_count = 0
    for i, url in enumerate(video_urls):
        logger.info(f"--- Processing video {i+1}/{len(video_urls)}: {url} ---")
        
        try:
            clips = processor.process_video(url, max_clips=5, overlap=15)
            if clips:
                processed_talks_count += 1
                logger.info(f"Finished processing {url}. Total talks processed so far: {processed_talks_count}")
            else:
                logger.warning(f"No clips were extracted for {url}. It might have been skipped or failed.")

        except Exception as e:
            logger.error(f"An error occurred while processing {url}: {e}")

    logger.info(f"Pipeline finished. Total talks processed: {processed_talks_count}")

if __name__ == "__main__":
    run_pipeline()
