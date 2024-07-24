from padelLynxPackage import ComposeVideo

video_path = "//dataset/padel5/padel5_segment3.mp4"
output = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel5/output.mp4"
json_path = "//dataset/padel5/padel5_segment3_points.json"

export.json_points_to_video(json_path, video_path, output)