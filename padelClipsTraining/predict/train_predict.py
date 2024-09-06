import json

from padelClipsPackage.predictPointModel import PredictPointModel

def generate_features_df(tracks):
    features_path = '/home/juliofgx/PycharmProjects/PadelClips/padelClipsTraining/predict/1set_ft.xlsx'
    points_path = '/home/juliofgx/PycharmProjects/PadelClips/padelClipsTraining/predict/1set_points.json'

    with open(points_path) as f:
        true_labels = json.load(f)

    ppm = PredictPointModel(tracks, true_labels['frames'])
    ppm.initialize_df(features_path)

def train_predict_model():
    features_path = '/home/juliofgx/PycharmProjects/PadelClips/padelClipsTraining/predict/1set_ft.xlsx'
    PredictPointModel.learn(features_path)


#train_predict_model()