import cv2
import re
import textwrap
import numpy as np
import arabic_reshaper
import pysrt
import subprocess
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict
    
class RenderSubtitle:
    '''
    RenderSubtitle Class: Optimal Subtitle Placement & Rendering

    The `RenderSubtitle` class is designed to handle **multilingual subtitle rendering** while ensuring 
    subtitles do not obstruct key visual elements in a video. It achieves this through **language detection, 
    optimal text positioning, and dynamic font selection**. This class integrates **OpenCV, PIL, and FFmpeg** 
    to process subtitles efficiently in real-time.

    ### Features:

    ✔ **Universal Font Support**: Uses `GoNotoKurrent-Regular.ttf` to handle multiple languages, 
    including **Latin, CJK (Chinese, Japanese, Korean), Arabic, Indic, and Thai**.

    ✔ **Language-Aware Rendering**: Automatically detects the script used in the subtitle text and 
    adjusts **font settings** for improved readability.

    ✔ **Smart Subtitle Placement**: Computes a **safe zone** dynamically to avoid covering faces, 
    logos, and on-screen text, ensuring **optimal readability** without obstructing important visual elements.

    ✔ **Adaptive Text Wrapping & Styling**: Dynamically wraps subtitle text within the **safe zone**, 
    adjusts **font size** based on the language, and adds a **semi-transparent background** to enhance readability.

    ✔ **Fast Subtitle Lookup**: Parses `.srt` subtitle files and **pre-indexes** subtitles to enable 
    quick retrieval and **frame-based rendering**.

    ✔ **Real-Time FPS Synchronization**: Extracts the **frame rate (FPS)** of the video using FFmpeg 
    to ensure **accurate subtitle timing** and **seamless playback**.

    ### Main Methods:

    - **`get_font(font_size)`** → Loads a **universal multilingual font** to ensure consistent text rendering.
    - **`detect_language(text)`** → Detects the script type (e.g., Latin, CJK, Arabic) for **language-aware rendering**.
    - **`render_subtitle_multi_new(frame, subtitle_text, safe_zone, frame_width, frame_height, max_chars_per_line=40, opacity=0.8)`**  
    → **Overlays subtitles onto the video frame** while ensuring optimal positioning and text clarity.
    - **`parse_srt_file(srt_file)`** → Reads an `.srt` file and **prepares subtitles** for fast frame-based retrieval.
    - **`get_subtitles_for_frames(frame_times, subtitle_dict)`** → Retrieves **matching subtitles** based on timestamps.
    - **`get_video_fps(video_path)`** → Extracts the **frame rate (FPS)** from a video to maintain **synchronization**.

    This class enables **smooth, non-obtrusive subtitle placement** in various languages while preserving 
    **video clarity and key visual elements**. It is optimized for **news videos, movies, and multilingual 
    media content** to enhance the viewer's experience.
    '''

    def get_font(font_size):
        """
        Loads the universal OpenSans font.

        Parameters:
            font_size (int): The desired font size.

        Returns:
            PIL.ImageFont: The loaded font.
        """
        font_path = "/content/optimal_subtitle/Font/GoNotoKurrent-Regular.ttf"
        return ImageFont.truetype(font_path, font_size)

    def detect_language(text):
        """
        Detects if the text contains Latin, CJK, Arabic, or Indic characters.

        Returns:
            str: "latin", "cjk", "arabic", "indic", "thai" based on detected script.
        """

        # Extract the first two words
        words = re.findall(r'\b\w+\b', text)  # Split text into words
        text_snippet = " ".join(words[:2])  # Take only the first two words

        if any("\u0600" <= ch <= "\u06FF" for ch in text_snippet):  # Arabic script range
            return "arabic"
        elif any("\u4E00" <= ch <= "\u9FFF" for ch in text_snippet):  # Chinese script range
            return "cjk"
        elif any("\u3040" <= ch <= "\u30FF" for ch in text_snippet):  # Japanese script range
            return "cjk"
        elif any("\uAC00" <= ch <= "\uD7AF" for ch in text_snippet):  # Korean script range
            return "cjk"
        elif any("\u0900" <= ch <= "\u097F" for ch in text_snippet):  # Devanagari script (Hindi, Marathi, Sanskrit)
            return "indic"
        elif any("\u0E00" <= ch <= "\u0E7F" for ch in text_snippet):  # Thai script
            return "thai"
        return "latin"  # Default to Latin if nothing is detected

    def render_subtitle_multi_new(frame, subtitle_text, safe_zone, frame_width, frame_height, max_chars_per_line=40, opacity=0.8):
        """
        Render multi-line subtitles centered within the safe zone with a semi-transparent background.

        Parameters:
            frame (numpy array): The frame on which to render subtitles.
            subtitle_text (str): The text to display.
            safe_zone (tuple): (x1, y1, x2, y2) defining subtitle placement.
            frame_width (int): Width of the frame.
            frame_height (int): Height of the frame.
            opacity (float): Background opacity (0 = fully transparent, 1 = fully opaque).

        Returns:
            numpy array: The frame with subtitles rendered at an optimal position.
        """
        x1, y1, x2, y2 = safe_zone
        language = RenderSubtitle.detect_language(subtitle_text)  # **Detect language**
        font_size = 28 if language == "cjk" else 26  # Adjust font size for CJK characters
        font = RenderSubtitle.get_font(font_size)  # Load correct font

        # **Handle Right-to-Left (RTL) text (e.g., Arabic)**
        if language == "arabic":
            subtitle_text = get_display(arabic_reshaper.reshape(subtitle_text))

        # **Calculate max width available for text**
        max_text_width = x2 - x1 - 30  # Ensure padding (30)

        # **Estimate Average Character Width Dynamically Using Subtitle Text**
        if len(subtitle_text) > 0:
            char_width = sum(font.getbbox(char)[2] - font.getbbox(char)[0] for char in subtitle_text) / len(subtitle_text)
        else:
            char_width = font_size // 2  # Fallback for empty text

        # **Determine Maximum Characters That Fit in Safe Zone**
        estimated_max_chars = max_text_width // char_width

        # **Use the Minimum of User-Defined or Estimated Max Chars**
        final_max_chars_per_line = min(estimated_max_chars, max_chars_per_line)

        # **Dynamically wrap text based on max character limit**
        wrapped_lines = []
        for line in subtitle_text.split("\n"):  # Handle existing line breaks
            new_lines = textwrap.wrap(line, width=int(final_max_chars_per_line))
            if new_lines:  # Only extend if wrapping produced text
                wrapped_lines.extend(new_lines)

        # **Fallback to prevent empty wrapped_lines**
        if not wrapped_lines:
            wrapped_lines = [" "]  # Ensures at least one blank line

        # **Measure Text Size**
        text_sizes = [font.getbbox(line) for line in wrapped_lines]
        text_width = max(size[2] - size[0] for size in text_sizes)  # Width (right - left)
        text_height = text_sizes[0][3] - text_sizes[0][1]  # Height (bottom - top)
        total_text_height = sum(size[3] - size[1] for size in text_sizes) + (len(wrapped_lines) - 1) * 10  # Extra spacing

        # **Center Text Within Safe Zone**
        text_x = x1 + (x2 - x1 - text_width) // 2  # **Horizontally centered**
        text_y = y1 + (y2 - y1 - total_text_height) // 2 - 20# **Vertically centered**

        # **Define Background Box**
        bg_x1 = max(text_x - 15, 0)
        bg_y1 = max(text_y - 5, 0)
        bg_x2 = min(text_x + text_width + 15, frame_width - 1)
        bg_y2 = min(text_y + total_text_height + 15, frame_height - 1)

        # **Create Semi-Transparent Background**
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # Black background
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)  # Blend overlay with frame

        # **Render Text Using PIL (for better font handling)**
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)

        y_offset = text_y
        for line in wrapped_lines:
            line_width = font.getbbox(line)[2] - font.getbbox(line)[0]  # Measure width
            line_x = x1 + (x2 - x1 - line_width) // 2  # Center per line
            draw.text((line_x, y_offset), line, font=font, fill=(255, 255, 255))  # White text
            y_offset += text_height + 10  # Extra line spacing

        return np.array(frame_pil)  # Convert back to OpenCV format

    def parse_srt_file(srt_file):
        """
        Reads and parses an SRT file, pre-indexing subtitles for fast lookup.

        Parameters:
            srt_file (str): Path to the .srt file.

        Returns:
            dict: A dictionary where keys are integer timestamps (seconds),
                and values are subtitle texts.
        """
        subs = pysrt.open(srt_file)
        subtitle_dict = defaultdict(lambda: None)  # Default to None for missing frames

        for sub in subs:
            start_time = int(sub.start.minutes * 60 + sub.start.seconds)  # Round to nearest second
            end_time = int(sub.end.minutes * 60 + sub.end.seconds)
            subtitle_text = sub.text.replace("\n", " ")  # Convert newlines to spaces

            # ✅ Store subtitles for all frames in the time range
            for t in range(start_time, end_time + 1):
                subtitle_dict[t] = subtitle_text

        return subtitle_dict  # Faster lookups using a dictionary
    
    def get_subtitles_for_frames(frame_times, subtitle_dict):
        """
        Retrieves subtitles for a batch of frame timestamps.

        Parameters:
            frame_times (list): List of timestamps (in seconds).
            subtitle_dict (dict): Pre-indexed subtitle dictionary.

        Returns:
            list: List of subtitle texts corresponding to each frame timestamp.
        """
        return [subtitle_dict.get(int(time), "") for time in frame_times]
    
    def get_video_fps(video_path):
        """Extracts FPS from a video using FFmpeg."""
        cmd = ["ffmpeg", "-i", video_path]

        # ✅ Use stdout and stderr explicitly
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # ✅ Parse FPS from FFmpeg output
        for line in result.stderr.split("\n"):
            if "Stream" in line and "Video" in line and "fps" in line:
                fps_value = float(line.split("fps")[0].strip().split()[-1])  # Extract FPS
                return fps_value

        return 30  # Default to 30 FPS if not found