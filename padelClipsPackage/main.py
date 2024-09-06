import json


from padelClipsPackage.aux import get_video_fps
from padelClipsPackage.Frame import Frame
from padelClipsPackage.Frame import Label
from padelClipsPackage.Game import Game
import numpy as np


from padelClipsPackage.ComposeVideo import points_to_json, json_points_to_video, ComposeVideo

ball_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/ball_inference.xlsx"
players_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/players_inference.xlsx"
players_ft_npz = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/players_inference_features.npz"
video_path = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/1set/1set_fixed.mp4"
making_path = "/home/juliofgx/PycharmProjects/PadelClips/making"
resources_path = "/home/juliofgx/PycharmProjects/PadelClips/resources"
#start = 23700
#end = 25200

frames = Frame.load_from_excel(ball_excel, players_excel, mapping={'ball': {0: Label.BALL}, 'players': {0: Label.PLAYER, 1: Label.NET}})
print("Frames loaded.")
fps = get_video_fps(video_path)

player_features = np.load(players_ft_npz)
def get_player_features(tag):
    pf = player_features[str(int(tag))]
    return pf

player_features_dict = {str(int(key)): player_features[key] for key in player_features.files}

#game = Game(frames, fps, player_features_dict, start=start, end=end)
game = Game(frames, fps, player_features_dict)

output_path = '/home/juliofgx/PycharmProjects/PadelClips/output.mp4'
video = ComposeVideo(game, making_path, resources_path, video_path, output_path)