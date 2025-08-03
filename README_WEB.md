# TED Talk Speaker Extractor - Web Application

A modern web interface for extracting speaker-only clips from TED Talks using AI-powered detection.

## Features

- 🎯 **AI-Powered Detection**: Uses MediaPipe for accurate speaker identification
- ✂️ **Smart Clipping**: Extracts 30-second optimized clips
- 🚫 **No Overlap**: Ensures non-overlapping segments
- ⚡ **Fast Processing**: Optimized for speed and efficiency
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔄 **Real-time Progress**: Live status updates during processing
- 📊 **Job History**: Track all processed videos

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_web.txt
```

### 2. Run the Web Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## Usage

1. **Enter YouTube URL**: Paste a TED Talk YouTube URL
2. **Select Clips**: Choose how many clips to extract (3, 5, or 10)
3. **Start Processing**: Click "Start Processing" to begin
4. **Monitor Progress**: Watch real-time progress updates
5. **Download Clips**: Download extracted speaker-only clips

## Deployment Options

### Option 1: Local Development

```bash
# Install dependencies
pip install -r requirements_web.txt

# Run the application
python app.py
```

### Option 2: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p static/clips downloads

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t ted-extractor .
docker run -p 5000:5000 ted-extractor
```

### Option 3: Cloud Deployment

#### Heroku

1. Create `Procfile`:
```
web: python app.py
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

#### Railway

1. Connect your GitHub repository
2. Railway will automatically detect and deploy the Flask app

#### Render

1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements_web.txt`
4. Set start command: `python app.py`

### Option 4: VPS Deployment

1. **Install dependencies on your server**:
```bash
sudo apt update
sudo apt install python3 python3-pip ffmpeg
pip3 install -r requirements_web.txt
```

2. **Run with Gunicorn** (for production):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Set up Nginx** (optional):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Configuration

### Environment Variables

You can set these environment variables:

- `SECRET_KEY`: Flask secret key (default: auto-generated)
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)
- `FLASK_ENV`: Set to 'production' for production deployment

### Customization

- **Output Directory**: Change `output_dir` in `WebSpeakerExtractor` class
- **Clip Duration**: Modify `clip_duration` in `extract_clips` method
- **Detection Confidence**: Adjust `min_detection_confidence` in MediaPipe initialization

## API Endpoints

- `GET /`: Main web interface
- `POST /api/process`: Start video processing
- `GET /api/job/<job_id>`: Get job status
- `GET /api/jobs`: List all jobs
- `GET /download/<filename>`: Download clip file

## File Structure

```
ted-talk-processor/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface template
├── static/
│   └── clips/            # Extracted clips storage
├── downloads/            # Temporary video downloads
├── requirements_web.txt  # Python dependencies
└── README_WEB.md        # This file
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg on your system
2. **MediaPipe errors**: Ensure you have the latest version
3. **Memory issues**: Reduce `max_clips` or video quality
4. **Download failures**: Check internet connection and URL validity

### Performance Tips

- Use SSD storage for faster video processing
- Increase server RAM for better performance
- Consider using a CDN for clip downloads
- Implement video caching for repeated requests

## Security Considerations

- Change the default `SECRET_KEY`
- Use HTTPS in production
- Implement rate limiting for API endpoints
- Add user authentication if needed
- Regularly clean up temporary files

## Monitoring

The application includes:
- Real-time job status tracking
- Progress indicators
- Error logging
- Job history

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs in the console
3. Ensure all dependencies are installed correctly

## License

This project is open source and available under the MIT License. 