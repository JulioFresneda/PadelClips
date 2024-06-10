from padelLynxPackage.aux import *
from padelLynxPackage.Frame import Frame


#split_train_test_valid("dataset/padel5/train_2_ball/train")

#video_to_frames("dataset/padel5/segment4/padel5_segment4.mp4", "dataset/padel5/segment4/frames", steps=1)

frames = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/segment4/labels", None)
fps = get_video_fps("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/segment4/padel5_segment4.mp4")
print(fps)
game = Game(frames, fps)
print(game)




