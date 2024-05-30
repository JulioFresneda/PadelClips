import torch
import cv2
import numpy as np
import subprocess
import re, os
from moviepy.editor import VideoFileClip, concatenate_videoclips
def read_yolo_txt(file_path):
    """
    Reads a YOLO format .txt file and returns a list of detections.

    Each detection is a tuple: (class_label, x_center, y_center, width, height)

    Parameters:
        file_path (str): Path to the YOLO format .txt file.

    Returns:
        List[Tuple[int, float, float, float, float]]: List of detections.
    """
    detections = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_label = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            detections.append((class_label, x_center, y_center, width, height))

    return detections


def split_video(source_path, segment_length=30):
    """ Split the video into segments of 'segment_length' seconds. """
    video = VideoFileClip(source_path)
    duration = int(video.duration)
    segments = []

    for start in range(0, duration, segment_length):
        end = min(start + segment_length, duration)
        segment = video.subclip(start, end)
        segment_path = f"{source_path}_{start}_{end}.mp4"
        segment.write_videofile(segment_path, codec="libx264")
        segments.append(segment_path)

    return segments

def process_segments(segments, model):
    """ Run model inference on each video segment and save the results. """
    processed_clips = []
    for segment in segments:
        # Replace 'model' with your actual model inference call
        result_clip_path = model(segment, stream=False, imgsz=1920, save=True, line_width=4, save_txt=True, save_conf=True)
        processed_clips.append(VideoFileClip(result_clip_path))

    return processed_clips

def concatenate_segments(processed_clips, output_path):
    """ Concatenate all processed segments into a final video. """
    final_clip = concatenate_videoclips(processed_clips)
    final_clip.write_videofile(output_path, codec="libx264")

