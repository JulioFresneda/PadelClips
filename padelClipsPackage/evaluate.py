import json
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np

from padelClipsPackage.Frame import Frame
from padelClipsPackage.Game import Game
from padelClipsPackage.Object import Label
from padelClipsPackage.aux import get_video_fps


class GameModel(BaseEstimator):
    def __init__(self, frames_data, fps, player_features, actual_points,
                 max_bottom_mountains=2,
                 max_top_mountains=2,
                 max_height_top_mountains=0.1,
                 jumps_min_frames=60,
                 jumps_max_allowed=0.2,
                 jumps_max_num=1,
                 slow_balls_min_frames=20,
                 slow_balls_min_velocity=2,
                 disc_min_frames=60,
                 disc_frames_disc=10,
                 disc_min_occurrences=2):
        # Store parameters as attributes
        self.frames_data = frames_data
        self.fps = fps
        self.player_features = player_features
        self.actual_points = actual_points

        # Explicitly set each hyperparameter as an attribute
        self.max_bottom_mountains = max_bottom_mountains
        self.max_top_mountains = max_top_mountains
        self.max_height_top_mountains = max_height_top_mountains
        self.jumps_min_frames = jumps_min_frames
        self.jumps_max_allowed = jumps_max_allowed
        self.jumps_max_num = jumps_max_num
        self.slow_balls_min_frames = slow_balls_min_frames
        self.slow_balls_min_velocity = slow_balls_min_velocity
        self.disc_min_frames = disc_min_frames
        self.disc_frames_disc = disc_frames_disc
        self.disc_min_occurrences = disc_min_occurrences

        # Store all parameters in a config dict for easier management
        self.config = {
            'max_bottom_mountains': self.max_bottom_mountains,
            'max_top_mountains': self.max_top_mountains,
            'max_height_top_mountains': self.max_height_top_mountains,
            'jumps_min_frames': self.jumps_min_frames,
            'jumps_max_allowed': self.jumps_max_allowed,
            'jumps_max_num': self.jumps_max_num,
            'slow_balls_min_frames': self.slow_balls_min_frames,
            'slow_balls_min_velocity': self.slow_balls_min_velocity,
            'disc_min_frames': self.disc_min_frames,
            'disc_frames_disc': self.disc_frames_disc,
            'disc_min_occurrences': self.disc_min_occurrences
        }


    def fit(self, X=None, y=None):
        # Wrap hyperparameters into a dictionary before passing to Game
        game_config = {
            'max_bo ttom_mountains': self.max_bottom_mountains,
            'max_top_mountains': self.max_top_mountains,
            'max_height_top_mountains': self.max_height_top_mountains,
            'jumps_min_frames': self.jumps_min_frames,
            'jumps_max_allowed': self.jumps_max_allowed,
            'jumps_max_num': self.jumps_max_num,
            'slow_balls_min_frames': self.slow_balls_min_frames,
            'slow_balls_min_velocity': self.slow_balls_min_velocity,
            'disc_min_frames': self.disc_min_frames,
            'disc_frames_disc': self.disc_frames_disc,
            'disc_min_occurrences': self.disc_min_occurrences
        }
        self.game = Game(self.frames_data, self.fps, self.player_features, game_config)
        return self

    def score(self, X=None, y=None):
        # Use the Evaluate class to calculate the F1 score
        evaluator = Evaluate(self.actual_points, self.frames_data, self.fps, self.player_features)
        f1_score = evaluator.eval(self.game)
        #print(f"Evaluated config: {self.config} with F1 Score: {f1_score}")
        return f1_score


class Evaluate:
    def __init__(self, actual_points, frames, fps, player_features):
        self.actual_points = actual_points
        self.frames = frames
        self.fps = fps
        self.player_features = player_features

    def eval(self, game):
        # And you have a 'game' object with a 'points' list containing predicted points
        # Each point has start() and end() methods to get the start and end frame.
        predicted_points = game.points  # This is a list of predicted point objects
        actual_points = self.actual_points

        # Convert actual points to a set of frame ranges for easy comparison
        actual_ranges = set()
        for start, end in actual_points:
            actual_ranges.update(range(start, end + 1))

        # Calculate True Positives, False Positives, and False Negatives
        TP = 0
        FP = 0
        FN = 0

        # For each predicted point, check if it overlaps with any actual point
        for predicted in predicted_points:
            predicted_range = set(range(predicted.start(), predicted.end() + 1))

            if predicted_range & actual_ranges:  # if there's any overlap
                TP += 1
            else:
                FP += 1

        # For each actual point, check if it overlaps with any predicted point
        for start, end in actual_points:
            actual_range = set(range(start, end + 1))

            overlaps = False
            for predicted in predicted_points:
                predicted_range = set(range(predicted.start(), predicted.end() + 1))
                if predicted_range & actual_range:
                    overlaps = True
                    break

            if not overlaps:
                FN += 1

        # Calculate precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1_score:.2f}')

        return f1_score

# Load data and initialize objects
ball_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/ball_inference.xlsx"
players_excel = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/players_inference.xlsx"
players_ft_npz = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_full/players_inference_features.npz"
video_path = "/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/2set_fixed.mp4"

player_features = np.load(players_ft_npz)
player_features_dict = {str(int(key)): player_features[key] for key in player_features.files}

frames = Frame.load_from_excel(ball_excel, players_excel, mapping={'ball': {0: Label.BALL}, 'players': {0: Label.PLAYER, 1: Label.NET}})
print("Frames loaded.")
fps = get_video_fps(video_path)

with open("/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/points.json") as f:
    actual_points = json.load(f)['frames']

# Define the search space
search_space = {
    'max_bottom_mountains': Integer(2, 4),
    'max_top_mountains': Integer(2, 6),
    'max_height_top_mountains': Real(0.1, 0.5, prior='log-uniform'),
    'jumps_min_frames': [i for i in range(60, 121, 10)],
    'jumps_max_allowed': Real(0.2, 0.8, prior='log-uniform'),
    'jumps_max_num': Integer(1, 4),
    'slow_balls_min_frames': [i for i in range(20, 121, 20)],
    'slow_balls_min_velocity': Integer(2, 12),
    'disc_min_frames': [i for i in range(60, 181, 60)],
    'disc_frames_disc': [i for i in range(10, 40, 10)],
    'disc_min_occurrences': [2, 3, 4]
}

# Define the Bayesian Optimization process
opt = BayesSearchCV(
    estimator=GameModel(frames, fps, player_features_dict, actual_points),
    search_spaces=search_space,
    n_iter=20,  # Number of iterations
    cv=3,  # Cross-validation strategy, adjust as necessary,
    n_jobs=-1
)

# Create dummy data
X_dummy = np.zeros((10, 1))  # 10 samples, 1 feature (can be any shape)
y_dummy = np.zeros(10)       # 10 labels (can be any shape)

# Fit the optimizer
opt.fit(X=X_dummy, y=y_dummy)  # Pass the dummy data
# Output the best found parameters
print(f"Best hyperparameters: {opt.best_params_}")
print(f"Best score: {opt.best_score_}")
