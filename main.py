import os
import cv2
import torch
from optimal_subtitle_placement import SubtitlePlacement
from render_subtitle import RenderSubtitle

def main_program(video_input_path, final_video_path, srt_file_path, tmp_audio_path, tmp_video_path, pre_position, max_chars_per_line, opacity, batch_size=4):
        """
        Processes a video by overlaying subtitles optimally and maintaining synchronization with the original audio.

        Parameters:
            video_input_path (str): Path to the input video file.
            final_video_path (str): Path to save the final processed video with subtitles and audio.
            srt_file_path (str): Path to the subtitle file (.srt) to be used.
            tmp_audio_path (str): Temporary path to store extracted audio.
            tmp_video_path (str): Temporary path to store the processed video without audio.
            pre_position (dict): Predefined safe zones for subtitle placement to avoid overlaying key elements.
            max_chars_per_line (int): Maximum number of characters per subtitle line for wrapping and readability.
            opacity (float): Transparency level of the subtitle background (0 = fully transparent, 1 = fully opaque).
            batch_size (int, optional): Number of frames to process in a batch for optimization. Default is 4.

        Steps:
            1. Extracts the FPS from the input video to ensure frame synchronization.
            2. Extracts the audio from the original video for merging after processing.
            3. Loads and parses the subtitle file.
            4. Opens the video file and initializes video writing settings.
            5. Detects and utilizes GPU if available for accelerated processing.
            6. Reads video frames in batches and applies subtitle placement using predefined safe zones.
            7. Dynamically adjusts subtitle wrapping based on max characters per line.
            8. Renders subtitles with background opacity settings for better visibility.
            9. Saves the processed frames to a temporary video file.
            10. Merges the processed video with the extracted audio while maintaining sync.
            11. Cleans up temporary files to optimize storage.

        Returns:
            Saves the final processed video with embedded subtitles and synchronized audio at the specified output path.
        """
        
        # ✅ Define file paths in /content/
        video_input_path = video_input_path
        final_video_path = final_video_path
        srt_file_path = srt_file_path

        # ✅ Create temporary paths inside /content/
        audio_path = tmp_audio_path
        output_video_path = tmp_video_path

        # ✅ Extract FPS dynamically
        fps = RenderSubtitle.get_video_fps(video_input_path)
        print(f"✅ Corrected FPS: {fps}")

        # ✅ Extract audio from the original video
        os.system(f"ffmpeg -i {video_input_path} -q:a 0 -map a {audio_path}")

        # ✅ Load Pre-indexed Subtitles
        subtitle_data = RenderSubtitle.parse_srt_file(srt_file_path)

        # ✅ Open Video File
        cap = cv2.VideoCapture(video_input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ✅ Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # ✅ Automatically select GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model.to(device).half()  # Use FP16 for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.set_num_threads(8)

        # ✅ Process Video with Batching
        batch_size = batch_size
        frame_buffer = []
        timestamp_buffer = []
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no more frames

            frame_time = frame_number / fps  # Convert frame number to timestamp
            frame_buffer.append(frame)
            timestamp_buffer.append(frame_time)

            # ✅ Process in batch when buffer reaches batch_size
            if len(frame_buffer) == batch_size:
                subtitles = RenderSubtitle.get_subtitles_for_frames(timestamp_buffer, subtitle_data)  # ✅ Batch subtitle lookup
                processed_frames = SubtitlePlacement.process_frames_batch(frame_buffer, subtitles, pre_position, max_chars_per_line, opacity)  # ✅ Batch processing

                for processed_frame in processed_frames:
                    out.write(processed_frame)  # Write frame to temporary video file

                frame_buffer.clear()
                timestamp_buffer.clear()  # Reset batch buffers

            frame_number += 1

        # ✅ Process remaining frames if they exist
        if frame_buffer:
            subtitles = RenderSubtitle.get_subtitles_for_frames(timestamp_buffer, subtitle_data)
            processed_frames = SubtitlePlacement.process_frames_batch(frame_buffer, subtitles)
            for processed_frame in processed_frames:
                out.write(processed_frame)

        cap.release()
        out.release()

        print(f"✅ Video without audio saved at: {output_video_path}")

        # ✅ Merge Video & Audio with Correct FPS & Sync Fixes
        os.system(f"ffmpeg -i {output_video_path} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v libx264 -preset fast -crf 23 -c:a copy -vsync vfr {final_video_path}")

        # ✅ Clean up the temporary files
        os.remove(audio_path)
        os.remove(output_video_path)

        print(f"✅ Final video with audio saved at: {final_video_path}")