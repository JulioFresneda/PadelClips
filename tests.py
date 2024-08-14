import json

import pandas as pd

from padelClipsPackage import ComposeVideo, aux
from padelClipsPackage.Point import Shot
from padelClipsPackage.aux import get_video_fps, extract_frame_from_video, crop_and_save_image
from padelClipsPackage.Frame import Frame
from padelClipsPackage.Frame import Label
from padelClipsPackage.Game import Game
import numpy as np


from padelClipsPackage.ComposeVideo import ComposeVideo, points_to_json, shots_to_json, json_points_to_video

#aux.split_train_test_valid("/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_2/train/train/")



ball_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/ball_inference.xlsx"
players_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/players_inference.xlsx"
players_ft_npz = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/players_inference_features.npz"
video_path = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_fixed.mp4"

frames = Frame.load_from_excel(ball_excel, players_excel, mapping={'ball': {0: Label.BALL}, 'players': {0: Label.PLAYER, 1: Label.NET}})
print("Frames loaded.")
fps = get_video_fps(video_path)


game = Game(frames, fps, np.load(players_ft_npz))


points = points_to_json(game.gameStats.top_x_longest_points(10))
with open('/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/points.json', 'w') as json_file:
    json.dump(points, json_file, indent=4)
json_points_to_video(points, video_path, "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/points.mp4", margin=0)

#ComposeVideo(game, "/home/juliofgx/PycharmProjects/PadelClips/material", video_path)