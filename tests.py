import pandas as pd

from padelLynxPackage.aux import *
from padelLynxPackage.Frame import Frame


ball_excel = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/ball_inference.xlsx"
players_excel = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/players_inference.xlsx"
players_ft_npz = "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/players_inference_features.npz"

frames = Frame.load_from_excel(ball_excel, players_excel, mapping={'ball':{0:Label.BALL}, 'players':{0:Label.NET, 1:Label.PLAYER}})
#frames = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/runs/detect/predict4/labels")
#frames_net = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/predicted/labels_net/labels", mapping={0:Label.NET, 1:Label.PLAYER})
#frames = Frame.merge_frame_list(frames, frames_net)

fps = get_video_fps("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/padel5_segment3.mp4")
print(fps)
game = Game(frames, fps, np.load(players_ft_npz))




