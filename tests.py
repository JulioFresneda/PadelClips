import io
import os
import pickle
import random
import subprocess
import re
import sys
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, transfx, vfx

from padelClipsPackage import Game

from bs4 import BeautifulSoup

from padelClipsPackage.Point import Shot
from padelClipsPackage.Shot import Category, Position
from padelClipsPackage.aux import extract_frame_from_video, crop_and_save_image

import base64
import ssl
import hashlib


from PIL import Image
import os
import base64




clip1 = '/home/juliofgx/PycharmProjects/PadelClips/making/heatmap_base_1.png'
clip2 = '/home/juliofgx/PycharmProjects/PadelClips/making/heatmap_base_2.png'
clip3 = '/home/juliofgx/PycharmProjects/PadelClips/making/heatmap_base_3.png'
clips = [clip1, clip2, clip3]

target_resolution = (1920, 1080)
transition_duration = 1


c1 = ImageClip(clip1, duration=3).resize(target_resolution).set_start(0).crossfadein(1)
c2 = ImageClip(clip2, duration=3).resize(target_resolution).set_start(2).crossfadein(1)
c3 = ImageClip(clip3, duration=3).resize(target_resolution).set_start(4).crossfadein(1)

# Manually create overlapping sections by trimming the last part of each clip
final_clip = CompositeVideoClip([c1, c2, c3])

# Write the output file
final_clip.write_videofile('output.mp4', codec="libx264", fps=30)