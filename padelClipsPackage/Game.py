import os, re
from enum import Enum
import numpy as np
from PIL import ImageFont, Image, ImageDraw

from dtaidistance import dtw
from moviepy.video.VideoClip import ImageClip
from sklearn.cluster import KMeans, DBSCAN

import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

from padelClipsPackage.Frame import Frame, Label
from padelClipsPackage.FramesController import FramesController
from padelClipsPackage.GameStats import GameStats
from padelClipsPackage.Object import PlayerTemplate, Player, PlayerPosition
from padelClipsPackage.Point import Point
from padelClipsPackage.aux import apply_kalman_filter
import random

from padelClipsPackage.PositionTracker import PositionTracker
import numpy as np
import ast

from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import os
import subprocess

class Game:
    def __init__(self, frames, fps, player_features):
        self.frames_controller = FramesController(frames)
        self.player_features = player_features

        self.players = self.set_player_templates()

        # Tag frames
        self.frames_controller.tag_frames(self.players, self.player_features)

        self.fps = int(fps)
        self.detect_net()
        self.points = self.cook_points()
        self.categorize_shots()
        #self.merge_points_too_close()

        self.gameStats = GameStats(self.frames, self.points, self.net)
        self.gameStats.print_game_stats()


    def categorize_shots(self):
        for point in self.points:
            for shot in point.shots:
                inf_frame = shot.inflection.frame_number
                player_pos = self.frames[inf_frame].player(shot.tag)
                shot.categorize(player_pos)

    def merge_points_too_close(self, margin=60):
        points = self.points.copy()
        points_merged = []

        buffer = points[0]

        for i in range(1, len(points)):
            diff = points[i].first_frame() - points[i - 1].last_frame()
            if diff <= margin:
                buffer.merge(points[i])

            else:
                points_merged.append(buffer)
                buffer = points[i]

        self.points = points_merged

    def get_players(self):
        return self.players.copy()

    def tag_position_in_players(self):

        for tplayer in self.players:
            tplayer.set_position_oocam(self.frames[0].player(tplayer.tag).x, self.frames[0].player(tplayer.tag).y)

        for frame in self.frames:
            Player.position_players(frame.players())

            for tplayer in self.players:
                player = frame.player(tplayer.tag)
                oocam_x = tplayer.position_oocam[0]
                oocam_y = tplayer.position_oocam[1]
                try:

                    if player.position == PlayerPosition.OVER_RIGHT:
                        if player.x > oocam_x:
                            oocam_x = player.x
                        if player.y < oocam_y:
                            oocam_y = player.y
                    elif player.position == PlayerPosition.OVER_LEFT:
                        if player.x < oocam_x:
                            oocam_x = player.x
                        if player.y < oocam_y:
                            oocam_y = player.y
                    elif player.position == PlayerPosition.UNDER_RIGHT:
                        if player.x > oocam_x:
                            oocam_x = player.x
                        if player.y > oocam_y:
                            oocam_y = player.y
                    elif player.position == PlayerPosition.UNDER_LEFT:
                        if player.x < oocam_x:
                            oocam_x = player.x
                        if player.y > oocam_y:
                            oocam_y = player.y
                except:
                    pass


                tplayer.set_position_oocam(oocam_x, oocam_y)



    def fill_players_out_of_camera(self):

        if len(self.frames[0].players()) < 4:
            break_loop = False

            for i in range(len(self.frames)):
                if break_loop:
                    break
                if len(self.frames[i].players()) == 4:
                    players = self.frames[i].players()
                    break_loop = True
                    for j in range(i):
                        for p in players:
                            if not self.frames[j].has_player(p.tag):
                                tplayer = [player for player in self.players if player.tag == p.tag][0]

                                _p = Player(Label.PLAYER.value, tplayer.position_oocam[0], tplayer.position_oocam[1], 0.0, 0.0, None, p.tag)
                                _p.tag_position(p.position)
                                self.frames[j].objects.append(_p)

                                Player.position_players(self.frames[j].players())






        for i in range(len(self.frames)):
            if len(self.frames[i].players()) < 4 and len(self.frames[i-1].players()) == 4:
                players = self.frames[i-1].players()
                for p in players:
                    if not self.frames[i].has_player(p.tag):
                        x, y = 0, 0
                        for tp in self.players:
                            if tp.tag == p.tag:
                                x, y = tp.position_oocam[0], tp.position_oocam[1]

                        _p = Player(Label.PLAYER.value, x, y, 0.0, 0.0, None, p.tag)
                        _p.tag_position(p.position)
                        self.frames[i].objects.append(_p)

                        Player.position_players(self.frames[i].players())

    def cook_points(self):
        tracks = self.track_ball()
        Point.net = self.net
        Point.frames = self.frames
        Point.players = self.players

        points = []
        for track in tracks:
            point = Point(track)
            points.append(point)
        return points

    def get_shots(self, player_tag=None, category=None):
        shots = []
        for point in self.points:
            for shot in point.shots:
                if (shot.tag == player_tag or player_tag is None) and (shot.category == category or category is None):
                    shots.append(shot)

        return shots









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
        for point in self.points:
            top_x_lists.append(point)

        sorted_keys = sorted(top_x_lists, key=lambda x: abs(x.first_frame() - x.last_frame()), reverse=True)

        print("Game duration: " + self.frame_to_timestamp(self.frames[-1].frame_number))

        list_sorted = sorted_keys if len(sorted_keys) <= top_n else sorted_keys[:top_n]

        for i, top in enumerate(list_sorted):
            print(
                "Game " + str(i) + ": " + self.frame_to_timestamp(top.first_frame()) + " -> " + self.frame_to_timestamp(
                    top.last_frame()))

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
        points = PositionTracker(self.frames, self.fps, self.net)
        return points.points








    def set_player_templates(self):
        frame_template = self.frames_controller.get_template_players()

        players = []
        idx_to_names = {0: "A", 1: "B", 2: "C", 3: "D"}

        def get_player_features(tag):
            pf = self.player_features[str(int(tag))]
            return pf

        for idx, mr_player in enumerate(frame_template.players()):
            mr_player_tag = mr_player.tag

            mr_player_features = get_player_features(mr_player_tag)
            game_player = PlayerTemplate(idx_to_names[idx], mr_player_features, frame_template.frame_number, mr_player)
            players.append(game_player)

        return players




