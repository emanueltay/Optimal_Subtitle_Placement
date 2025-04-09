# Optimal Subtitle Positioning through Deep Learning

## 📌 Project Description

Optimal Subtitle Placement is a project designed to **intelligently overlay subtitles on videos** while avoiding key visual elements. It ensures **readable, non-obtrusive subtitles** by utilizing **YOLO-based object detection** to determine safe zones for text placement.

## ✨ Key Features

- 🚀 **YOLO-based object detection** to avoid occluding important visual elements (e.g., faces, news tickers, graphs).
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

### 3️⃣ **📁 Project Directory Structure**

The following structure organizes all required components for optimal subtitle placement:

```
project/
├── Config/                         # Configuration files
│   ├── config.json                 # Global settings (paths, model, etc.)
│   └── prepositions.json           # Subtitle safe zone pre-defined regions
│
├── Input/                          # Input files
│   ├── video.mp4                   # Sample input video
│   └── subtitles.ttml              # Corresponding subtitle file
│
├── Model/                          # YOLO model storage
│   └── model.pt                    # Fine-tuned YOLO model
│
├── Output/                         # Output folder
│   └── subtitles_updated.ttml      # Generated output with positioned subtitles
│
├── main.py                         # Main entry point (subtitle placement logic)
└── requirements.txt                # Python dependencies
```

### 🔧 System Workflow Overview

1. **Frame Extraction**
   - The system begins by extracting frames directly from the input video using its **native frames-per-second (FPS)** rate.
   - This ensures accurate alignment between video frames and subtitle timestamps defined in the **TTML** file.

2. **Subtitle Timeframe Filtering**
   - Frames are **filtered by subtitle timeframes**, retaining only those that fall within the **start and end times** of each subtitle.
   - This reduces unnecessary computation and ensures focus on relevant visual content only.

3. **Batch Processing for Detection**
   - Frames corresponding to each subtitle are grouped into **batches**.
   - These batches are passed through a **YOLO-based object detection model** to identify key visual elements such as **faces**, **logos**, or **in-scene text**.
   - To optimize performance, detection is **downsampled to 3 FPS**, balancing speed with contextual accuracy.

4. **Safe Zone Comparison**
   - Detected bounding boxes are compared against a predefined list of **subtitle-safe zones** stored in a **JSON file**.
   - These safe zones are defined as **percentage-based coordinates** relative to the frame, making them resolution-independent.

5. **Fallback & Dynamic Region Selection**
   - If a predefined safe zone overlaps with detected objects, the system attempts to **shift the region** horizontally or vertically.
   - If all preferred regions are obstructed, **fallback zones or dynamically computed positions** are used instead.
   - This hierarchical decision-making ensures subtitles remain **unobtrusive** and **visually clear**.

6. **TTML Update**
   - The chosen subtitle region is inserted into each subtitle’s `<region>` tag within the **TTML** file.
   - All used regions are added to the **TTML `<layout>` section**, ensuring proper rendering across subtitle viewers.


## 🚀 Usage Instructions

### Run the main program:

```bash
python3 main.py \
  --video input/video.mp4 \
  --ttml input/subtitles.ttml \
  --output output/subtitles_updated.ttml \
  --resize 854 480 \
  --stream_fps 5 \
  --log_path logs/profiling.txt
```

### Parameters:

| Argument        | Description                                                                 |
| --------------- | --------------------------------------------------------------------------- |
| `--video`       | Path to the input video file (e.g. `input/video.mp4`)                       |
| `--ttml`        | Path to the input subtitle file in TTML format                              |
| `--stream_fps`  | Optional FPS sampling for processing (e.g. `5` means process 5 frames/sec)  |
| `--resize`      | Optional resize dimensions for YOLO detection (e.g. `854 480`)              |
| `--output`      | Output path for the updated TTML file                                       |
| `--log_path`    | Optional path to save profiling summary log (e.g. `logs/profiling.txt`)     |

### Subtitle Style Configuration

The global subtitle style is defined within the TTML file using the `<style>` tag under the `<styling>` section. This ensures **consistent formatting** for all subtitles rendered across the video.

> **Note:** All style-related attributes (e.g., font size, color, background) should be configured in the `config.json` file.

#### Config.json
```json
{
    "region_json_path": "Config/subtitle_regions.json",
    "model_path": "Model/model.pt",
    "safe_zone_history_length": 3,
    "subtitle_style": {
        "xml:id": "s0",
        "tts:color": "white",
        "tts:fontSize": "80%",
        "tts:fontFamily": "sansSerif",
        "tts:backgroundColor": "black",
        "tts:displayAlign": "center",
        "tts:wrapOption": "wrap"
    }
}
```

#### Subtitle_regions.json
```json
{
  "bottom":        { "percentages": [0.1, 0.9, 0.9, 1.0], "priority": 5 },
  "above_1":       { "percentages": [0.1, 0.85, 0.9, 1.0], "priority": 5 },
  "above_2":       { "percentages": [0.1, 0.8, 0.9, 0.95], "priority": 5 },
  "above_3":       { "percentages": [0.1, 0.75, 0.9, 0.9], "priority": 5 },
  "middle3":       { "percentages": [0.1, 0.7, 0.9, 0.85], "priority": 4 },
  "between_m3_m2_1": { "percentages": [0.1, 0.65, 0.9, 0.8], "priority": 4 },
  "between_m3_m2_2": { "percentages": [0.1, 0.6, 0.9, 0.75], "priority": 4 },
  "middle2":       { "percentages": [0.1, 0.55, 0.9, 0.7], "priority": 3 }
}
```

### Current Style Attributes

| Attribute               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `xml:id="s0"`          | Unique identifier for the style, referenced in `<body style="s0">`.        |
| `tts:color="white"`    | Text color for the subtitles.                                               |
| `tts:fontSize="80%"`   | Font size relative to screen size; should be kept at or below 80% for clarity. |
| `tts:fontFamily="sansSerif"` | Font style used for a clean and simple display.                     |
| `tts:backgroundColor="black"` | Background box color to enhance contrast and visibility.           |
| `tts:displayAlign="center"`  | Aligns text vertically in the middle of the defined region.         |
| `tts:wrapOption="wrap"`      | Allows long lines to wrap instead of overflowing.                   |

---

### 📏 Subtitle Size Notes

- The current `extent` Y value for regions is set at **15%** of the screen height.
- This configuration works best with `tts:fontSize` set to **80% or less**.
- Increasing the font size beyond this may cause subtitle overflow or clipping within the frame.

---