# Optimal Subtitle Placement

## 📌 Project Description

Optimal Subtitle Placement is a project designed to **intelligently overlay subtitles on videos** while avoiding key visual elements. It ensures **readable, non-obtrusive subtitles** by utilizing **YOLO-based object detection** to determine safe zones for text placement.

## ✨ Key Features

- 🚀 **YOLO-based object detection** to avoid occluding important visual elements (e.g., faces, news tickers, graphs).
- 🌍 **Multi-language support** including Arabic, CJK (Chinese, Japanese, Korean), Thai, and Latin scripts.
- 🔄 **Dynamic subtitle rendering**, adjusting text placement and font selection based on frame characteristics.
- ⚡ **Batch video processing** for improved efficiency.
- 🎯 **Ensures subtitle clarity** by avoiding cluttered regions in video frames.

---

## 🛠 Installation Instructions

### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/your-repo/optimal-subtitle-placement.git
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
                --batch_size 4
```

### Parameters:

| Argument         | Description                                        |
| ---------------- | -------------------------------------------------- |
| `--video_input`  | Path to the input video file                       |
| `--srt_file`     | Path to the subtitle file (.srt)                   |
| `--output_video` | Path to save the final processed video             |
| `--batch_size`   | Number of frames processed in a batch (default: 4) |

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---