import json


from padelClipsPackage.aux import get_video_fps
from padelClipsPackage.Frame import Frame
from padelClipsPackage.Frame import Label
from padelClipsPackage.Game import Game
import numpy as np


from padelClipsPackage.ComposeVideo import points_to_json, json_points_to_video

ball_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/ball_inference.xlsx"
players_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/players_inference.xlsx"
players_ft_npz = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/players_inference_features.npz"
video_path = "/media/juliofgx/OS/2set_fixed.mp4"

start = 18000
end = 19800

frames = Frame.load_from_excel(ball_excel, players_excel, mapping={'ball': {0: Label.BALL}, 'players': {0: Label.PLAYER, 1: Label.NET}})
print("Frames loaded.")
fps = get_video_fps(video_path)

player_features = np.load(players_ft_npz)
def get_player_features(tag):
    pf = player_features[str(int(tag))]
    return pf

player_features_dict = {str(int(key)): player_features[key] for key in player_features.files}

game = Game(frames, fps, player_features_dict, start=start, end=end)

points = points_to_json(game.gameStats.top_x_longest_points(10))
with open('/home/juliofgx/PycharmProjects/PadelClips/playtime_segments.json', 'w') as json_file:
    json.dump(points, json_file, indent=4)
json_points_to_video(points, video_path, "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/points.mp4", margin=60)
