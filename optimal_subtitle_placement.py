from ultralytics import YOLO
import cv2
import json
import re
import argparse
import numpy as np
import pysrt
import subprocess
import torch
import os
import math
import time
from collections import Counter, deque, defaultdict
import xml.etree.ElementTree as ET

# Global in-memory cache for dynamic & shifted safe zones
safe_zone_cache = {}
used_safe_zones = {}  # Dictionary to store all assigned safe zones

safe_zone_history = deque(maxlen=3)  # Stores past safe zones for consistency (4)
region_json_path = "subtitle_regions_scaled_test.json"

# Load the trained model
model = YOLO("Model/fine-tuned_model(YOLO12)(100_epochs)/best.pt")

# Automatically select GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move model to the selected device
model.to(device).float() 

# Optimize PyTorch settings
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True
torch.set_num_threads(torch.get_num_threads())  # Use optimal number of CPU threads

class SubtitlePlacement:

    def detect_objects(frames):
        """
        Perform batch object detection on multiple frames.

        Parameters:
            frames (list): List of frames (each a NumPy array).

        Returns:
            list: A list containing detections for each frame.
                Each element is a NumPy array of detections.
        """
        # Run the model in batch mode
        results = model(frames, batch=len(frames), verbose=False)  # Run batch inference

        # Extract detections for each frame
        batch_detections = []
        for result in results:
            detections = result.boxes.data.cpu().numpy()  # Convert detections to NumPy array
            batch_detections.append(detections)

        return batch_detections  # List of detections per frame

    def calculate_safe_zone_with_prepositions(frame_width, frame_height, detections, pre_positions, subtitle_height=30, margin=10, shift_x=20):
        """
        Calculate the safe zone for subtitle placement using pre-defined positions.
        If blocked, it attempts to shift left/right before moving vertically.
        If no predefined position works, falls back to a dynamic safe zone and caches it in memory.

        Returns:
            tuple: (position_name, coordinates)
        """

        def zones_overlap(zone1, zone2, threshold=0.05):  # threshold
            """Checks if two zones overlap significantly (IoA)."""
            x1a, y1a, x2a, y2a = zone1
            x1b, y1b, x2b, y2b = zone2

            # Calculate overlap rectangle
            inter_x1 = max(x1a, x1b)
            inter_y1 = max(y1a, y1b)
            inter_x2 = min(x2a, x2b)
            inter_y2 = min(y2a, y2b)

            # Check if there's any intersection
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return False

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            zone_area = (x2a - x1a) * (y2a - y1a)

            # Intersection over Area (IoA)
            iou = inter_area / zone_area

            return iou > threshold

        # Step 1: Try Predefined Positions
        for position_name, position in sorted(pre_positions.items(), key=lambda x: x[1].get("priority", 0), reverse=True):
            x1, y1, x2, y2 = position["coordinates"]

            # if not any(zones_overlap((x1, y1, x2, y2), detection[:4]) for detection in detections):
            if not any(zones_overlap((x1, y1, x2, y2), detection[:4], threshold=0.1) for detection in detections):
                used_safe_zones[position_name] = {
                    "coordinates": [x1, y1, x2, y2]
                }
                return (position_name, (x1, y1, x2, y2))

            # Step 2: Try shifting left and right
            min_width = max(0.6 * frame_width, 600)
            for shift_dir in ["left", "right"]:
                attempts = 0
                shift_value = shift_x
                while attempts < 10:
                    if shift_dir == "left":
                        new_x1 = max(0, x1 - shift_value)
                        new_x2 = max(min_width, x2 - shift_value)
                    else:
                        new_x1 = min(frame_width - min_width, x1 + shift_value)
                        new_x2 = min(frame_width, x2 + shift_value)

                    shifted_zone = (new_x1, y1, new_x2, y2)
                    if new_x2 > new_x1 and not any(zones_overlap(shifted_zone, detection[:4]) for detection in detections):
                        shifted_name = f"shifted_{position_name}"
                        used_safe_zones[shifted_name] = {
                            "coordinates": [new_x1, y1, new_x2, y2]
                        }
                        return (shifted_name, shifted_zone)

                    shift_value *= 1.5
                    attempts += 1

        # Step 3: Try Cached Safe Zone (if exists)
        cache_key = (frame_width, frame_height, tuple(tuple(d) for d in detections))
        if cache_key in safe_zone_cache:
            return safe_zone_cache[cache_key]

        # Step 4: Dynamic fallback position (bottom-up shifting)
        fallback_position_name = "dynamic_position"
        proposed = (0, frame_height - subtitle_height - margin, frame_width, frame_height - margin)
        while True:
            if all(not zones_overlap(proposed, detection[:4]) for detection in detections):
                safe_zone_cache[cache_key] = (fallback_position_name, proposed)
                used_safe_zones[fallback_position_name] = {
                    "coordinates": list(proposed)
                }
                return (fallback_position_name, proposed)

            # Shift upwards
            x1, y1, x2, y2 = proposed

            new_y1 = y1 - subtitle_height - margin
            new_y2 = y2 - subtitle_height - margin

            if new_y1 < 0:
                break
            proposed = (x1, new_y1, x2, new_y2)

        # Step 5: Final Fallback to Top
        fallback_position_name = "fallback_top"
        final_safe_zone = (0, margin, frame_width, subtitle_height + margin)
        safe_zone_cache[cache_key] = (fallback_position_name, final_safe_zone)
        used_safe_zones[fallback_position_name] = {
            "coordinates": list(final_safe_zone)
        }
        return (fallback_position_name, final_safe_zone)
    
    def get_used_safe_zones():
        """
        Returns the used safe zones as a dictionary without the "priority" field.
        This can be directly used for updating the TTML layout.
        """
        return {
            key: {"coordinates": value["coordinates"]}
            for key, value in used_safe_zones.items()
        }

    def get_subtitle_size(frame_height):
        """
        Dynamically calculate subtitle height and margin based on frame resolution.

        Parameters:
            frame_height (int): Height of the video frame.

        Returns:
            tuple: (subtitle_height, margin)
        """
        subtitle_height = max(0.15 * frame_height, 30)  # Minimum 18px for readability
        margin = max(0.02 * frame_height, 5)  # Minimum 5px to avoid text touching edges

        return int(subtitle_height), int(margin)
    
    def get_pixel_pre_positions_from_json(json_path, frame_width, frame_height):
        """
        Reads percentage-based layout from a JSON file and converts to pixel coordinates.
        
        Args:
            json_path (str): Path to the JSON file containing percentages.
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.
        
        Returns:
            dict: Dictionary of region names mapped to pixel coordinates and priority.
        """
        with open(json_path, 'r') as f:
            percentage_data = json.load(f)
        
        pixel_positions = {}
        for region, data in percentage_data.items():
            x1_pct, y1_pct, x2_pct, y2_pct = data["percentages"]
            pixel_positions[region] = {
                "coordinates": [
                    int(x1_pct * frame_width),
                    int(y1_pct * frame_height),
                    int(x2_pct * frame_width),
                    int(y2_pct * frame_height)
                ],
                "priority": data["priority"]
            }
        
        return pixel_positions

    def process_frames_batch(frames, process_fps=3, video_fps=30):
        frame_interval = video_fps // process_fps
        selected_indices = list(range(0, len(frames), frame_interval))
        if not selected_indices:
            selected_indices = [0]
        selected_frames = [frames[i] for i in selected_indices]

        batch_detections = SubtitlePlacement.detect_objects(selected_frames)
        frame_height, frame_width = frames[0].shape[:2]
        pre_positions = SubtitlePlacement.get_pixel_pre_positions_from_json(region_json_path, frame_width, frame_height)
        subtitle_height, margin = SubtitlePlacement.get_subtitle_size(frame_height)

        batch_safe_zones = [
            SubtitlePlacement.calculate_safe_zone_with_prepositions(
                frame_width, frame_height, batch_detections[i], pre_positions, subtitle_height, margin
            )[0]
            for i in range(len(selected_frames))
        ]

        # Detect sudden change in safe zones
        if len(set(batch_safe_zones)) > 1:
            final_safe_zone = batch_safe_zones[-1]  # Use most recent zone
        else:
            final_safe_zone = batch_safe_zones[0]  # All zones consistent

        safe_zone_history.append(final_safe_zone)
        return final_safe_zone
    
class RenderSubtitle:

    def convert_ttml_time_to_seconds(ttml_time):
        """
        Converts TTML time format (HH:MM:SS.mmm, MM:SS.mmm, SS.mmm, or SS,mmm) to seconds.

        Parameters:
            ttml_time (str): TTML-formatted time.

        Returns:
            float: Time in seconds (with millisecond precision).
        """

        # Remove trailing 's' if present and replace ',' with '.'
        ttml_time = ttml_time.rstrip('s').replace(',', '.')

        # Use regex to extract time components
        match = re.match(r"(?:(\d+):)?(?:(\d+):)?(\d+)(?:\.(\d+))?", ttml_time)

        if not match:
            raise ValueError(f"Invalid TTML time format: {ttml_time}")

        # Extract components safely
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        milliseconds = int(match.group(4)) if match.group(4) else 0

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    def parse_subtitle_file(file_path):
        """
        Parses either an SRT or TTML subtitle file and extracts subtitles.

        Parameters:
            file_path (str): Path to the subtitle file.

        Returns:
            list: List of subtitles in the format:
                [
                    {"start": start_time, "end": end_time, "text": "subtitle text", "region": "region_id"}
                ]
        """
        extension = os.path.splitext(file_path)[-1].lower()
        subtitle_data = []

        if extension == ".srt":
            subs = pysrt.open(file_path)
            for sub in subs:
                start_time = (
                    sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
                )
                end_time = (
                    sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
                )
                text = sub.text.replace("\n", " ")  # Convert newlines to spaces

                subtitle_data.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "region": None  # SRT doesn't support regions
                })

        elif extension == ".ttml":
            # Register TTML Namespaces
            ET.register_namespace('', "http://www.w3.org/ns/ttml")  # Default TTML namespace
            ET.register_namespace('ttp', "http://www.w3.org/ns/ttml#parameter")
            ET.register_namespace('tts', "http://www.w3.org/ns/ttml#styling")
            ET.register_namespace('ttm', "http://www.w3.org/ns/ttml#metadata")

            # Parse TTML File
            tree = ET.parse(file_path)
            root = tree.getroot()
            ns = {'ttml': 'http://www.w3.org/ns/ttml'}

            # Extract Subtitle Data
            for p in root.findall('.//ttml:p', ns):
                start_time = RenderSubtitle.convert_ttml_time_to_seconds(p.attrib.get("begin", "0.0s"))
                end_time = RenderSubtitle.convert_ttml_time_to_seconds(p.attrib.get("end", "0.0s"))
                text = " ".join(p.itertext()).strip()
                region = p.attrib.get("region", None)

                subtitle_data.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "region": region
                })

        else:
            raise ValueError("Unsupported subtitle format. Only SRT and TTML are supported.")

        return subtitle_data
    
    def get_video_fps(video_path):
        """Extracts FPS from a video using FFmpeg."""
        cmd = ["ffmpeg", "-i", video_path]

        # Use stdout and stderr explicitly
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Parse FPS from FFmpeg output
        for line in result.stderr.split("\n"):
            if "Stream" in line and "Video" in line and "fps" in line:
                fps_value = float(line.split("fps")[0].strip().split()[-1])  # Extract FPS
                return fps_value

        return 30  # Default to 30 FPS if not found

    def generate_updated_ttml(ttml_file_path, output_ttml_path, json_data, subtitle_data, frame_width, frame_height):
        """
        Generates a new TTML file with updated subtitle styles, layout regions, and assigned regions for subtitles.

        Parameters:
            ttml_file_path (str): Path to the input TTML file.
            output_ttml_path (str): Path to save the updated TTML file.
            json_data (dict): JSON data containing subtitle positions.
            subtitle_data (list): List of subtitles with timestamps and regions.
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.

        Returns:
            None (Writes updated TTML file to disk)
        """

        # Load TTML File
        tree = ET.parse(ttml_file_path)
        root = tree.getroot()

        # Load TTML File
        tree = ET.parse(ttml_file_path)
        root = tree.getroot()

        # Preserve All Original Root Attributes (Ensuring All Namespaces Remain)
        root_attribs = root.attrib.copy()  # Copy attributes before modification

        # Extract Namespace (from <tt> root tag)
        namespace_uri = root.tag.split("}")[0].strip("{")  # Extracts URI from "{namespace}tag"
        ns = {"ttml": namespace_uri} if namespace_uri else {}

        # Restore All Root Attributes (Explicitly Add Missing Namespaces)
        root.attrib.clear()
        root.attrib.update(root_attribs)  # Restore original attributes

        # Ensure `xmlns:tts` is Explicitly Set (if missing)
        if "xmlns:tts" not in root.attrib:
            root.set("xmlns:tts", "http://www.w3.org/ns/ttml#styling")  # Add missing styling namespace

        # Find or Create the <head> Element (Using Preserved Namespace)
        head_element = root.find(f'.//{{{namespace_uri}}}head', ns)
        if head_element is None:
            head_element = ET.Element(f"{{{namespace_uri}}}head")
            root.insert(0, head_element)  # Insert <head> as the first child

        # Find or Create the <styling> Element
        styling_element = head_element.find('.//ttml:styling', ns)
        if styling_element is None:
            styling_element = ET.Element("{http://www.w3.org/ns/ttml}styling")
            head_element.insert(0, styling_element)  # Insert before layout

        # Remove Any Existing <style> Elements (Always Replacing)
        for style in styling_element.findall('.//ttml:style', ns):
            styling_element.remove(style)

        # Define and Add the New Style Element
        new_style = ET.Element("{http://www.w3.org/ns/ttml}style", attrib={
            "xml:id": "s0",
            "tts:color": "white",
            "tts:fontSize": "70%",
            "tts:fontFamily": "sansSerif",
            "tts:backgroundColor": "black",
            "tts:displayAlign": "center",
            "tts:wrapOption": "wrap"
        })
        styling_element.append(new_style)

        # Find or Create the <layout> Element
        layout_element = head_element.find('.//ttml:layout', ns)
        if layout_element is None:
            layout_element = ET.Element("{http://www.w3.org/ns/ttml}layout")
            head_element.append(layout_element)

        # Remove ALL existing <region> elements inside <layout>
        for region in list(layout_element):
            layout_element.remove(region)

        # Insert Subtitle Regions from JSON
        for region_name, region_data in json_data.items():
            x1, y1, x2, y2 = region_data["coordinates"]

            # print(frame_height,frame_width)

            # Convert absolute pixel values to TTML percentages
            origin_x = (x1 / frame_width) * 100
            origin_y = (y1 / frame_height) * 100
            extent_x = ((x2 - x1) / frame_width) * 100
            extent_y = ((y2 - y1) / frame_height) * 100

            # Construct the region XML element
            region_element = ET.Element("{http://www.w3.org/ns/ttml}region", attrib={
                "tts:origin": f"{math.ceil(origin_x)}% {math.ceil(origin_y)}%",
                "tts:extent": f"{math.ceil(extent_x)}% {math.ceil(extent_y)}%",
                "tts:displayAlign": "center",
                "tts:textAlign": "center",
                "xml:id": region_name
            })

            # Add to <layout>
            layout_element.append(region_element)

        # Find All <p> Elements (Subtitles) and Update Regions
        for p in root.findall('.//ttml:p', ns):
            start_time = RenderSubtitle.convert_ttml_time_to_seconds(p.attrib.get("begin", "0.0s"))
            end_time = RenderSubtitle.convert_ttml_time_to_seconds(p.attrib.get("end", "0.0s"))

            # Find Matching Subtitle
            matched_subtitle = next((sub for sub in subtitle_data if sub["start"] <= start_time <= sub["end"]), None)

            if matched_subtitle:
                if matched_subtitle["region"] is not None:
                    p.attrib["region"] = matched_subtitle["region"]  # Assign Correct Region
                elif "region" in p.attrib:
                    del p.attrib["region"]  # Remove `region` if it's None

        # Save Updated TTML File
        tree.write(output_ttml_path, encoding="utf-8", xml_declaration=True)

class Main:
    def log_profiling_summary(video_duration, load_duration, run_duration, total_video_read_time, total_yolo_time, total_region_assign_time, ttml_generation_time, output_path, log_path=None):
        minutes, seconds = divmod(video_duration, 60)
        run_minutes, run_seconds = divmod(run_duration, 60)

        summary_lines = [
            "\n📊 PROFILING SUMMARY",
            f"🎬 Video Duration: {int(minutes)}m {int(seconds)}s",
            f"📦 Load Time: {load_duration:.2f}s",
            f"🚀 Run Time: {int(run_minutes)}m {int(run_seconds)}s",
            f"📥 Video Read Time: {total_video_read_time:.2f}s",
            f"🔍 YOLO Detection Time: {total_yolo_time:.2f}s",
            f"📐 Region Assignment Time: {total_region_assign_time:.4f}s",
            f"📝 TTML Generation Time: {ttml_generation_time:.4f}s",
            f"✅ Output TTML saved to: {output_path}"
        ]

        if log_path:
            with open(log_path, 'w') as f:
                f.write("\n".join(summary_lines))
            print(f"\n📝 Profiling summary written to: {log_path}")

    def main(video_input_path, ttml_file, output_path, resize_resolution=None, stream_fps=None, log_path=None):
        # Start Load Timer
        start_load_time = time.time()

        # Define file paths
        file_path = ttml_file

        # Load Video Metadata
        fps = RenderSubtitle.get_video_fps(video_input_path)
        print(f"Corrected FPS: {fps}")

        subtitle_data = RenderSubtitle.parse_subtitle_file(file_path)

        cap = cv2.VideoCapture(video_input_path)

        # Original Resolution for TTML generation
        original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"🎞 Frame Dimensions: {original_frame_width}x{original_frame_height}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Use resized resolution for detection if specified
        frame_width = original_frame_width
        frame_height = original_frame_height
        if resize_resolution:
            frame_width, frame_height = resize_resolution
            print(f"🔄 Resized Frame Dimensions for Detection: {frame_width}x{frame_height}")

        # End Load Timer
        end_load_time = time.time()
        load_duration = end_load_time - start_load_time
        print(f"📦 Total Load Time: {load_duration:.2f} seconds")

        # Start Run Timer
        start_run_time = time.time()

        frame_buffer = []
        timestamp_buffer = []
        subtitle_index = 0
        frame_number = 0
        total_video_read_time = 0
        total_yolo_time = 0
        total_region_assign_time = 0

        while cap.isOpened():
            read_start = time.time()
            ret, frame = cap.read()
            read_end = time.time()

            total_video_read_time += (read_end - read_start)
            if not ret:
                break

            frame_time = frame_number / fps

            # Apply frame skipping if stream_fps is specified
            if stream_fps is not None and frame_number % int(fps // stream_fps) != 0:
                frame_number += 1
                continue

            # Resize frame for faster processing
            if resize_resolution:
                frame = cv2.resize(frame, resize_resolution)

            if subtitle_index < len(subtitle_data):
                current_subtitle = subtitle_data[subtitle_index]
                current_start = current_subtitle["start"]
                current_end = current_subtitle["end"]

                if current_start <= frame_time <= current_end:
                    frame_buffer.append(frame)
                    timestamp_buffer.append(frame_time)

                if frame_time > current_end:
                    if frame_buffer:
                        detect_start = time.time()
                        processed_frames = SubtitlePlacement.process_frames_batch(frame_buffer)
                        detect_end = time.time()
                        total_yolo_time += (detect_end - detect_start)

                        region_start = time.time()
                        current_subtitle["region"] = processed_frames
                        region_end = time.time()
                        total_region_assign_time += (region_end - region_start)

                        frame_buffer.clear()
                        timestamp_buffer.clear()
                        subtitle_index += 1

            frame_number += 1

        # Final batch
        if frame_buffer and subtitle_index < len(subtitle_data):
            detect_start = time.time()
            processed_frames = SubtitlePlacement.process_frames_batch(frame_buffer)
            detect_end = time.time()
            total_yolo_time += (detect_end - detect_start)
            subtitle_data[subtitle_index]["region"] = processed_frames

        cap.release()

        # End Run Timer
        end_run_time = time.time()
        run_duration = end_run_time - start_run_time

        # Generate TTML Layout using original resolution
        ttml_gen_start = time.time()
        layout = SubtitlePlacement.get_used_safe_zones()
        RenderSubtitle.generate_updated_ttml(file_path, output_path, layout, subtitle_data, original_frame_width, original_frame_height)
        ttml_gen_end = time.time()
        ttml_generation_time = ttml_gen_end - ttml_gen_start

        Main.log_profiling_summary(
            video_duration=video_duration,
            load_duration=load_duration,
            run_duration=run_duration,
            total_video_read_time=total_video_read_time,
            total_yolo_time=total_yolo_time,
            total_region_assign_time=total_region_assign_time,
            ttml_generation_time=ttml_generation_time,
            output_path=output_path,
            log_path=log_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Subtitle Placement")

    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--ttml", required=True, help="Path to input TTML subtitle file")
    parser.add_argument("--output", required=True, help="Path to save updated TTML file")
    parser.add_argument("--resize", nargs=2, type=int, default=None, help="Resize resolution (width height), e.g. --resize 640 360")
    parser.add_argument("--stream_fps", type=int, default=None, help="Reduce processing FPS by specifying a lower FPS (e.g. 5)")
    parser.add_argument("--log_path", type=str, default=None, help="Optional path to save profiling summary log (e.g. profiling.txt)")

    args = parser.parse_args()

    resize_resolution = tuple(args.resize) if args.resize else None

    # Run main with additional optional parameters
    Main.main(
        video_input_path=args.video,
        ttml_file=args.ttml,
        output_path=args.output,
        resize_resolution=resize_resolution,
        stream_fps=args.stream_fps,
        log_path=args.log_path
    )