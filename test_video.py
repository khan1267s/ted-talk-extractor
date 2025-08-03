#!/usr/bin/env python3
"""
Test script to check YouTube video accessibility
"""

from pytube import YouTube
import sys

def test_video_download(url):
    """Test if a YouTube video can be downloaded."""
    try:
        print(f"Testing URL: {url}")
        yt = YouTube(url)
        
        print(f"Title: {yt.title}")
        print(f"Length: {yt.length} seconds")
        print(f"Views: {yt.views}")
        
        # Get available streams
        streams = yt.streams.filter(progressive=True, file_extension='mp4')
        print(f"Available streams: {len(streams)}")
        
        for i, stream in enumerate(streams):
            print(f"  {i+1}. {stream.resolution} - {stream.filesize_mb:.1f}MB")
        
        if streams:
            print("✅ Video is accessible and downloadable!")
            return True
        else:
            print("❌ No suitable streams found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://www.youtube.com/watch?v=6Af6b_wyiwI"
    
    test_video_download(url) 