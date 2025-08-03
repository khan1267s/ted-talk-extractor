# Quick Start Guide - TED Talk Processor

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Test Installation
```bash
python test_installation.py
```

### Step 3: Run Your First Processing
```bash
python ted_processor.py "https://www.youtube.com/watch?v=u4ZoJKF_VuA"
```

## 📁 What You Get

After processing, you'll find:
- `output_clips/` - Your extracted speaker-only clips
- `downloads/` - Temporary video files (auto-cleaned)

## 🎯 Key Features

✅ **Downloads TED Talks** from YouTube  
✅ **Detects speakers** using AI (YOLOv8 + MediaPipe)  
✅ **Extracts clean clips** (30 seconds each)  
✅ **Excludes audience shots** and multiple people  
✅ **Batch processing** for multiple videos  
✅ **High-quality output** with preserved audio  

## 📋 Usage Examples

### Single Video
```bash
python ted_processor.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Multiple Videos
```bash
python ted_processor.py "URL1" "URL2" "URL3"
```

### Custom Settings
```bash
python ted_processor.py "URL" --output-dir "my_clips" --max-clips 20
```

### Batch Processing
```bash
python batch_process.py sample_urls.txt my_output 15
```

## 🔧 Troubleshooting

**"No suitable video stream found"**
- Check if the video is public and accessible
- Try a different TED Talk URL

**"Could not load YOLO model"**
- First run downloads the model automatically
- Ensure stable internet connection

**Low clip count**
- TED Talk might have many audience shots
- Try a different video or adjust settings

## 📊 Expected Results

- **Processing time**: 5-15 minutes per video (depending on length)
- **Clip count**: 5-30 clips per video (depending on speaker visibility)
- **Clip duration**: 30 seconds each
- **Quality**: High-definition with preserved audio

## 🎬 Sample TED Talks to Try

1. **Simon Sinek - Start With Why**
   ```
   https://www.youtube.com/watch?v=u4ZoJKF_VuA
   ```

2. **Amy Cuddy - Power Poses**
   ```
   https://www.youtube.com/watch?v=Ks-_Mh1QhMc
   ```

3. **Brene Brown - Vulnerability**
   ```
   https://www.youtube.com/watch?v=iCvmsMzlF7o
   ```

## 💡 Tips for Best Results

1. **Choose videos** with clear speaker shots
2. **Avoid videos** with frequent audience cuts
3. **Use good internet** for faster downloads
4. **Have sufficient disk space** (2-5GB per video)
5. **Be patient** - AI processing takes time

## 🆘 Need Help?

1. Run the test script: `python test_installation.py`
2. Check the full README.md for detailed documentation
3. Ensure all dependencies are installed correctly

---

**Happy Processing! 🎉** 