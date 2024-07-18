import pandas as pd

from padelLynxPackage import ComposeVideo
from padelLynxPackage.aux import get_video_fps, extract_frame_from_video, crop_and_save_image
from padelLynxPackage.Frame import Frame
from padelLynxPackage.Frame import Label
from padelLynxPackage.Game import Game
import numpy as np

from padelLynxPackage.ComposeVideo import ComposeVideo

ball_excel = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/segment3/ball_inference.xlsx"
players_excel = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/segment3/players_inference.xlsx"
players_ft_npz = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/segment3/players_inference_features.npz"

frames = Frame.load_from_excel(ball_excel, players_excel,
                               mapping={'ball': {0: Label.BALL}, 'players': {0: Label.NET, 1: Label.PLAYER}})
# frames = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/runs/detect/predict4/labels")
# frames_net = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/predicted/labels_net/labels", mapping={0:Label.NET, 1:Label.PLAYER})
# frames = Frame.merge_frame_list(frames, frames_net)

video_path = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/padel5_segment3.mp4"
fps = get_video_fps(video_path)
print(fps)
game = Game(frames, fps, np.load(players_ft_npz))

ComposeVideo(game, "/home/juliofgx/PycharmProjects/padelLynx/material", video_path)