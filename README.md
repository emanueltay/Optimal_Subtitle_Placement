# Optimal Subtitle Positioning through Deep Learning

## 📌 Project Description

Optimal Subtitle Placement is a project designed to **intelligently overlay subtitles on videos** while avoiding key visual elements. It ensures **readable, non-obtrusive subtitles** by utilizing **YOLO-based object detection** to determine safe zones for text placement.

## ✨ Key Features

- 🚀 **YOLO-based object detection** to avoid occluding important visual elements (e.g., faces, news tickers, graphs).
- 🌍 **Multi-language support** including Arabic, CJK (Chinese, Japanese, Korean), Thai, and Latin scripts.
- 🔄 **Dynamic subtitle rendering**, adjusting text placement and font selection based on frame characteristics.
- ⚡ **Batch frame processing** for improved efficiency.
- 🎯 **Ensures subtitle clarity** by avoiding cluttered regions in video frames.

---

## 🛠 Installation Instructions

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/emanueltay/Optimal_Subtitle_Placement.git
cd optimal-subtitle-placement
```

### 2️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

Ensure you have **FFmpeg** installed:

```bash
sudo apt update && sudo apt install ffmpeg
```

### 3️⃣ **Download Required Model Files**

Download the YOLO model weights and place them in the appropriate directory:

```bash
mkdir models
cd models
wget https://your-model-download-link/yolo_model.pt
```

---

## 🚀 Usage Instructions

### Run the main program:

```bash
python main.py --video_input "path/to/video.mp4" \  
               --srt_file "path/to/subtitles.srt" \  
               --output_video "path/to/output.mp4" \  
               --tmp_audio "path/to/temp_audio.aac" \  
               --tmp_video "path/to/temp_video.mp4" \  
               --pre_position "path/to/predefined_positions.json" \  
               --max_chars_per_line 40 \  
               --opacity 0.8 \  
               --batch_size 4
```

### Parameters:

| Argument         | Description                                        |
| ---------------- | -------------------------------------------------- |
| `--video_input`  | Path to the input video file                       |
| `--srt_file`     | Path to the subtitle file (.srt)                   |
| `--output_video`     | Path to save the final processed video with subtitles and audio                   |
| `--tmp_audio` | Temporary path to store extracted audio             |
| `-tmp_video`   | Temporary path to store the processed video without audio |
| `--pre_position`  | JSON file containing predefined safe zones for subtitle placement file                       |
| `--max_chars_per_line`     | Maximum number of characters per subtitle line (default: 40)                   |
| `--opacity` | Transparency level of the subtitle background (0 = fully transparent, 1 = fully opaque, default: 0.8)             |
| `--batch_size`   | Number of frames processed in a batch (default: 4) |
