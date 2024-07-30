import pandas as pd

from padelClipsPackage import ComposeVideo
from padelClipsPackage.Point import Shot
from padelClipsPackage.aux import get_video_fps, extract_frame_from_video, crop_and_save_image
from padelClipsPackage.Frame import Frame
from padelClipsPackage.Frame import Label
from padelClipsPackage.Game import Game
import numpy as np

from padelClipsPackage.ComposeVideo import ComposeVideo, points_to_json, shots_to_json, json_points_to_video

ball_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/ball_inference.xlsx"
players_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/players_inference.xlsx"
players_ft_npz = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/players_inference_features.npz"
video_path = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/1set_fixed.mp4"

frames = Frame.load_from_excel(ball_excel, players_excel, mapping={'ball': {0: Label.BALL}, 'players': {0: Label.NET, 1: Label.PLAYER}})
print("Frames loaded.")
fps = get_video_fps(video_path)

game = Game(frames, fps, np.load(players_ft_npz))


jshots = shots_to_json(game.get_shots())
json_points_to_video(jshots, video_path, "//dataset/padel_pove/1set/shots.mp4", margin=20)

#ComposeVideo(game, "/home/juliofgx/PycharmProjects/PadelClips/material", video_path)