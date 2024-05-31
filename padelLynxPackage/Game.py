import os, re
from enum import Enum
import numpy as np

from dtaidistance import dtw
from sklearn.cluster import KMeans, DBSCAN

import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

from padelLynxPackage.Frame import Object, Frame, Label
from padelLynxPackage.Player import *
import random

from padelLynxPackage.PositionTracker import PositionTracker


class Game:
    def __init__(self, frames, fps):
        self.frames = frames
        self.fps = int(fps)
        self.track_ball()
        self.players = self.initialize_players()
        self.tag_players()


    def __str__(self):
        print("Game: " + str(len(self.frames)) + " frames")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Game({str(len(self.frames))})"


    def track_ball(self):
        self.position_tracker = PositionTracker(self.frames, Label.BALL)

    def tag_players(self):
        for i, frame in enumerate(self.frames):
            if i%10 == 0:
                print("Tagging frame " + str(i) + " out of " + str(len(self.frames)), end='\r')
            self.tag_players_in_frame(frame, plot=False)


    def initialize_players(self):
        self.most_representative_frame = self.find_frame_with_average_confidence()
        players = []
        idx_to_names = {0: "A", 1:"B", 2:"C", 3:"D"}
        for idx, mr_player in enumerate(self.most_representative_frame.players()):
            yolo_info = mr_player.get_yolo()
            mr_player_features = PlayerFeatures(self.most_representative_frame.frame_path, yolo_info)
            game_player = Player(idx_to_names[idx], mr_player_features)
            players.append(game_player)

        return players
    def find_frame_with_average_confidence(self):
        best_frame = None
        best_average_confidence = 0.0

        for frame in self.frames:
            # Filter objects with class_label == 1
            class_1_objects = [obj for obj in frame.objects if obj.class_label == 1]

            # Check if there are exactly four such objects
            if len(class_1_objects) == 4:
                # Calculate the average confidence of these objects
                average_confidence = sum(obj.conf for obj in class_1_objects) / len(class_1_objects)

                # Find the frame where this average deviation is minimized
                if best_average_confidence < average_confidence:
                    best_average_confidence = average_confidence
                    best_frame = frame

        return best_frame


    def tag_players_in_frame(self, frame: Frame, plot=False):
        """
        Matches players between two frames based on extracted features and plots the results.

        Parameters:
        features_frame1 (dict): The features of players in the first frame.
        features_frame2 (dict): The features of players in the second frame.

        Returns:
        dict: A dictionary mapping player indices in frame 1 to player indices in frame 2.
        """
        matches = []
        pairs = {}

        tags = {}
        for player in self.players:
            tags[player.tag] = player

        objects = []
        for obj in frame.players():
            objects.append(obj)


        for tag in tags.keys():
            for idx, obj in enumerate(objects):
                obj_ft = PlayerFeatures(frame.frame_path, obj.get_yolo())
                dist = PlayerFeatures.get_distance(tags[tag].player_features, obj_ft)
                pairs[(tag, idx)] = dist

        while len(pairs.keys()) > 0:
            lowest_dist = float('inf')
            lowest_pair = (None, None)
            for (tag, idx), dist in pairs.items():
                if dist < lowest_dist:
                    lowest_dist = dist
                    lowest_pair = (tag, idx)
            matches.append(lowest_pair)
            tag = lowest_pair[0]
            idx = lowest_pair[1]
            tmp = list(pairs.keys()).copy()
            for pair in tmp:
                if tag == pair[0]:
                    pairs.pop(pair)
                elif idx == pair[1]:
                    pairs.pop(pair)

        print(matches)
        for match in matches:
            frame.players()[match[1]].tag = match[0]

        if(plot):

            for match in matches:
                img1, color_hist1, deep_feat1 = tags[match[0]].player_features.get_player_features()
                img2, color_hist2, deep_feat2 = PlayerFeatures(frame.frame_path, frame.players()[match[1]].get_yolo()).get_player_features()

                # Plot the results
                plt.figure(figsize=(24, 6))

                # Plot the cropped player image from the first frame
                plt.subplot(1, 3, 1)
                plt.title(f'Frame {self.most_representative_frame.frame_number} - Player {match[0]}')
                plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                # Plot the color histogram from the first frame
                plt.subplot(1, 3, 2)
                plt.title(f'Frame {self.most_representative_frame.frame_number} - Player {match[0]} Color Histogram')
                for channel, color in zip(range(3), ('r', 'g', 'b')):
                    plt.plot(color_hist1[channel * 256:(channel + 1) * 256], color=color)

                # Plot the deep features from the first frame
                plt.subplot(1, 3, 3)
                plt.title(f'Frame {self.most_representative_frame.frame_number} - Player {match[0]} Deep Features')
                plt.plot(deep_feat1[:100])  # Display the first 100 features for visualization

                plt.figure(figsize=(24, 6))

                # Plot the cropped player image from the second frame
                plt.subplot(1, 3, 1)
                plt.title(f'Frame {frame.frame_number} - Player {match[0]}')
                plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                # Plot the color histogram from the second frame
                plt.subplot(1, 3, 2)
                plt.title(f'Frame {frame.frame_number} - Player {match[0]} Color Histogram')
                for channel, color in zip(range(3), ('r', 'g', 'b')):
                    plt.plot(color_hist2[channel * 256:(channel + 1) * 256], color=color)

                # Plot the deep features from the second frame
                plt.subplot(1, 3, 3)
                plt.title(f'Frame {frame.frame_number} - Player {match[0]} Deep Features')
                plt.plot(deep_feat2[:100])  # Display the first 100 features for visualization

                plt.show()

        return matches
