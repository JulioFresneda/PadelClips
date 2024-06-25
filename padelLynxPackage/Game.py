import os, re
from enum import Enum
import numpy as np

from dtaidistance import dtw
from sklearn.cluster import KMeans, DBSCAN

import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

from padelLynxPackage import aux
from padelLynxPackage.Frame import Frame, Label
from padelLynxPackage.Player import *
import random

from padelLynxPackage.PositionTracker import PositionTracker
import numpy as np
import ast



class Game:
    def __init__(self, frames, fps, player_features):
        self.frames = frames
        self.player_features = player_features

        self.players = self.initialize_players()
        self.tag_players_in_frames()
        self.smooth_players_tags()


        self.fps = int(fps)
        self.detect_net()
        self.ball_playtime = self.track_ball()
        self.longest_points(10)



        self.ball_playtime.plot_tracks_with_net_and_players(self.ball_playtime.closed_tracks, frame_start=0)



    def smooth_players_tags(self):


        player_pos = {'A':[], 'B':[], 'C':[], 'D':[]}
        player_idx = {'A': [], 'B': [], 'C': [], 'D': []}
        for frame in self.frames:
            for player in frame.players():
                player_pos[player.tag].append((player.x, player.y))
                player_idx[player.tag].append(frame.frame_number)

        for playertag in player_pos.keys():
            smoothed = aux.apply_kalman_filter(player_pos[playertag])
            for s, fn in zip(smoothed, player_idx[playertag]):
                self.frames[fn].update_player_position(playertag, s[0], s[1])


    def get_player_features(self, tag):
        pf = self.player_features[str(int(tag))]
        return pf


    def detect_net(self):
        best_net_frame = self.find_frame_with_average_confidence(label=Label.NET, num_objects=1)
        self.net = [obj for obj in best_net_frame.objects if obj.class_label == Label.NET.value][0]

    def order_frames(self, frames):
        max_frame = max(obj.frame_number for obj in frames)

        # Create a list of None with the size of the maximum frame number + 1
        result = [None] * (max_frame + 1)

        # Place each object in its corresponding position
        for obj in frames:
            result[obj.frame_number] = obj

        return result

    def longest_points(self, top_n):
        top_x_lists = []
        for pt, kind in self.ball_playtime.playtime.items():
            if kind == 'point':
                top_x_lists.append(pt)

        sorted_keys = sorted(top_x_lists, key=lambda x: abs(x[0] - x[1]), reverse=True)

        for i, top in enumerate(sorted_keys[:top_n]):
            print("Game " + str(i) + ": " + self.frame_to_timestamp(top[0]) + " -> " + self.frame_to_timestamp(top[1]))

    def frame_to_timestamp(self, frame_number):
        # Calculate total seconds
        total_seconds = frame_number / self.fps

        # Hours, minutes, and seconds
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        # Format the timestamp as HH:MM:SS
        timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"

        return timestamp



    def __str__(self):
        print("Game: " + str(len(self.frames)) + " frames")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Game({str(len(self.frames))})"


    def track_ball(self):
        self.position_tracker = PositionTracker(self.frames, self.fps, self.net)
        return self.position_tracker

    def tag_players_in_frames(self):
        for i, frame in enumerate(self.frames):
            if i%100 == 0:
                print("Tagging frame " + str(i) + " out of " + str(len(self.frames)), end='\r')
            self.tag_players_in_frame(frame)
            for player in frame.players():
                if player.tag != 'A' and player.tag != 'B' and player.tag != 'C' and player.tag != 'D':
                    frame.objects.remove(player)



    def initialize_players(self):
        self.most_representative_frame_players = self.find_frame_with_average_confidence(label=Label.PLAYER, num_objects=4)
        players = []
        idx_to_names = {0: "A", 1:"B", 2:"C", 3:"D"}
        for idx, mr_player in enumerate(self.most_representative_frame_players.players()):
            mr_player_tag = mr_player.tag

            mr_player_features = self.get_player_features(mr_player_tag)
            game_player = Player(idx_to_names[idx], mr_player_features)
            players.append(game_player)

        return players
    def find_frame_with_average_confidence(self, label: Label, num_objects = 1):
        best_frame = None
        best_average_confidence = 0.0

        label_number = label.value

        for frame in self.frames:
            # Filter objects with class_label == 1
            class_objects = [obj for obj in frame.objects if obj.class_label == label_number]

            # Check if there are exactly four such objects
            if len(class_objects) == num_objects:
                # Calculate the average confidence of these objects
                average_confidence = sum(obj.conf for obj in class_objects) / len(class_objects)

                # Find the frame where this average deviation is minimized
                if best_average_confidence < average_confidence:
                    best_average_confidence = average_confidence
                    best_frame = frame

        return best_frame


    def tag_players_in_frame(self, frame: Frame):
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

        players_from_frame = []
        for obj in frame.players():
            players_from_frame.append(obj)


        for tag in tags.keys():
            for obj in players_from_frame:
                player_in_frame_ft = self.get_player_features(obj.tag)
                dist = Player.features_distance(tags[tag].template_features, player_in_frame_ft)
                pairs[(tag, obj.tag)] = dist

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


        for match in matches:
            frame.update_player_tag(match[1], match[0])




        return matches
