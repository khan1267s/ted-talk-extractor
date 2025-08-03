# TED Talk Processor

A Python tool that downloads TED Talks from YouTube and extracts speaker-only clips using advanced computer vision techniques.

## Features

- **YouTube Download**: Downloads TED Talks using pytube
- **Person Detection**: Uses multiple AI models (YOLOv8, MediaPipe) to detect people in video frames
- **Speaker-Only Filtering**: Extracts clips containing only the main speaker
- **Quality Control**: Excludes clips with audience members, multiple people, or distracting objects
- **Batch Processing**: Process multiple videos at once
- **Configurable Output**: Customizable clip duration and maximum number of clips

## Requirements

- Python 3.8+
- Windows 10/11 (tested on Windows)
- Sufficient disk space for video downloads and processing

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install FFmpeg** (for better video processing):
   - Download from: https://ffmpeg.org/download.html
   - Add to system PATH

## Usage

### Basic Usage

Process a single TED Talk:
```bash
python ted_processor.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Process multiple TED Talks:
```bash
python ted_processor.py "https://www.youtube.com/watch?v=VIDEO_ID1" "https://www.youtube.com/watch?v=VIDEO_ID2"
```

### Advanced Usage

```bash
python ted_processor.py "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output-dir "my_clips" \
    --max-clips 20
```

### Command Line Options

- `urls`: One or more YouTube video URLs (required)
- `--output-dir`: Output directory for clips (default: "output_clips")
- `--max-clips`: Maximum number of clips per video (default: 30)

## How It Works

### 1. Video Download
- Uses pytube to download the highest quality available stream
- Saves temporarily in `downloads/` directory

### 2. Person Detection
The tool uses three complementary methods to detect people:

1. **YOLOv8**: Fast and accurate person detection
2. **MediaPipe Pose**: Detects human pose landmarks
3. **MediaPipe Face**: Detects faces for additional validation

### 3. Speaker-Only Analysis
For each 30-second segment:
- Samples 10 frames throughout the segment
- Analyzes each frame for person detection
- Requires exactly 1 person with high confidence
- Excludes segments with multiple people or audience shots

### 4. Clip Extraction
- Extracts valid 30-second segments as MP4 files
- Uses H.264 codec for compatibility
- Preserves audio quality

## Output

The tool creates:
- `output_clips/`: Directory containing extracted clips
- `downloads/`: Temporary directory for downloaded videos (auto-cleaned)
- Log files with processing details

### Clip Naming Convention
```
{original_video_name}_clip_001.mp4
{original_video_name}_clip_002.mp4
...
```

## Example TED Talk URLs

Here are some popular TED Talks you can test with:

- **Simon Sinek - Start With Why**: `https://www.youtube.com/watch?v=u4ZoJKF_VuA`
- **Amy Cuddy - Power Poses**: `https://www.youtube.com/watch?v=Ks-_Mh1QhMc`
- **Brene Brown - Vulnerability**: `https://www.youtube.com/watch?v=iCvmsMzlF7o`

## Performance Tips

1. **GPU Acceleration**: Install CUDA for faster YOLO processing
2. **SSD Storage**: Use SSD for faster video I/O
3. **Sufficient RAM**: 8GB+ recommended for large videos
4. **Internet Speed**: Faster download speeds for better quality videos

## Troubleshooting

### Common Issues

1. **"No suitable video stream found"**
   - Video might be private or region-restricted
   - Try a different TED Talk URL

2. **"Could not load YOLO model"**
   - First run will download the model automatically
   - Check internet connection

3. **"Error downloading video"**
   - Check YouTube URL validity
   - Ensure stable internet connection

4. **Low clip count**
   - TED Talk might have many audience shots
   - Try adjusting confidence thresholds in code

### Performance Issues

- **Slow processing**: Consider reducing `max_clips` parameter
- **Memory issues**: Process one video at a time
- **Disk space**: Monitor `downloads/` and `output_clips/` directories

## Technical Details

### Detection Methods

1. **YOLOv8 (You Only Look Once)**
   - Real-time object detection
   - Person class detection with confidence scores
   - Fast and accurate

2. **MediaPipe Pose**
   - 33 body landmarks detection
   - Good for full-body speaker detection
   - Works well with TED Talk stage setups

3. **MediaPipe Face**
   - Face detection with confidence scores
   - Additional validation for speaker presence
   - Handles close-up shots

### Quality Thresholds

- **Person Count**: Exactly 1 person per frame
- **Confidence**: > 0.5 for YOLO detections
- **Segment Quality**: > 70% of sampled frames must be speaker-only
- **Clip Duration**: 30 seconds per clip

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 