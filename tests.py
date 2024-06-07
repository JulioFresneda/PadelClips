from padelLynxPackage.aux import *
from padelLynxPackage.Frame import Frame


#split_train_test_valid("dataset/padel5/train_2_ball/train")

#video_to_frames("dataset/padel5/padel5_segment2.mp4", "dataset/padel5/predicted_2/images", steps=1)

frames = Frame.load_frames("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/predicted_2/images", "/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/predicted_2/labels")
fps = get_video_fps("/home/juliofgx/PycharmProjects/padelLynx/dataset/padel5/padel5_segment2.mp4")
# print(fps)
game = Game(frames, fps)
print(game)












# Example usage
frame1_path = '/home/juliofgx/PycharmProjects/padelLynx/dataset/padel3/images_and_labels/images/train/frame_000056.jpg'  # Replace with the actual path to the first frame image
frame2_path = '/home/juliofgx/PycharmProjects/padelLynx/dataset/padel3/images_and_labels/images/train/frame_000603.jpg'  # Replace with the actual path to the second frame image

players_frame1 = [
    Player(0, PlayerFeatures(frame1_path, (0.395290, 0.319505, 0.078383, 0.233500))),
    Player(1, PlayerFeatures(frame1_path, (0.471216, 0.179373, 0.034344, 0.081042))),
    Player(2, PlayerFeatures(frame1_path, (0.904661, 0.740586, 0.176943, 0.342505))),
    Player(3, PlayerFeatures(frame1_path, (0.550384, 0.186144, 0.035779, 0.093000)))
]

players_frame2 = [
    Player(0, PlayerFeatures(frame2_path, (0.523228, 0.177764, 0.028372, 0.093204))),
    Player(1, PlayerFeatures(frame2_path, (0.601738, 0.213988, 0.035560, 0.095801))),
    Player(2, PlayerFeatures(frame2_path, (0.317607, 0.331940, 0.071365, 0.257528))),
    Player(3, PlayerFeatures(frame2_path, (0.861568, 0.631167, 0.180234, 0.366889)))
]

# Match players between the two frames
#matches = Game.match_players(players_frame1, players_frame2)

# Print the matches
#print(matches)




# clustering.cluster_positions(game, 11)

for i in range(4):
    pass
    # clustering.print_player_heatmap(game, i)

# x_positions, y_positions, frame_numbers, size = game.get_ball_features(add_nan=True)

#
# split_train_test_valid("dataset/padel3/images_and_labels")

# cut_video("dataset/padel3/padel3_segment.mp4",  2000, 2100, "dataset/padel3/padel3_segment_3.mp4")
# cut_video("dataset/padel3/padel3_segment.mp4",  10000, 15000, "dataset/padel3/padel3_segment_4.mp4")
