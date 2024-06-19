from padelLynxPackage.aux import *
from padelLynxPackage.Frame import Frame




#video_to_frames("dataset/padel5/padel5_segment3.mp4", "dataset/padel5/train_2_net/images", steps=1, limit=1001)
#split_train_test_valid("dataset/padel5/train_2_net/train")
frames = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/runs/detect/predict4/labels")
frames_net = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/predicted/labels_net/labels", mapping={0:Label.NET, 1:Label.PLAYER})
frames = Frame.merge_frame_list(frames, frames_net)

fps = get_video_fps("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/segment4/padel5_segment4.mp4")
print(fps)
game = Game(frames, fps)




