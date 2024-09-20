import time
import pickle

from padelClipsPackage.Frame import Label
from padelClipsPackage.FramesController import FramesController
from padelClipsPackage.GameStats import GameStats
from padelClipsPackage.Object import PlayerTemplate
from padelClipsPackage.Point import Point

from padelClipsPackage.PositionTracker import PositionTrackerV2
from padelClipsPackage.Shot import Position, Shot
from padelClipsPackage.TagAlgorithm import TagAlgorithm
from padelClipsPackage.Visuals import Visuals
import rust_functions
from padelClipsPackage.aux import compute_homography, transform_coordinates


# TODO
#

class Game:
    def __init__(self, frames, fps, player_features, start=0, end=None):
        self.player_features = player_features
        self.start = start
        self.end = len(frames) if end is None else end

        self.net = None
        self.players_boundaries_vertical = None
        self.players_boundaries_horizontal = None
        self.fps = int(fps)

        self.frames_controller = FramesController(frames[start:end])

        self.set_net()
        self.tag_players_v2()


        Point.game = self
        Shot.game = self
        self.track_ball_v2()
        #Visuals.plot_tracks_with_net_and_players(self.position_tracker, self.net, self.players_boundaries_vertical, points=self.points)

        print("Points loaded.")



        self.gameStats = GameStats(self)
        self.gameStats.print_game_stats()

    def load_hyperparameters(self, hyperparameters):
        self.default_hyperparameters = {
            'static_ball_min_frames': 120,
            'static_ball_min_diff_allowed': 0.005,
            'max_bottom_mountains': 2,
            'max_top_mountains': 4,
            'max_height_top_mountains': 0.1,
            'jumps_min_frames': 120,
            'jumps_max_allowed': 0.4,
            'jumps_max_num': 2,
            'slow_balls_min_frames': 60,
            'slow_balls_min_velocity': 8,
            'disc_min_frames': 120,
            'disc_frames_disc': 20,
            'disc_min_occurrences': 3
        }

        if hyperparameters is None:
            self.hyperparameters = self.default_hyperparameters
        else:
            self.hyperparameters = hyperparameters
            for default in self.default_hyperparameters.keys():
                if default not in self.hyperparameters.keys():
                    self.hyperparameters[default] = self.default_hyperparameters[default]
        print(f"Evaluated config: {self.hyperparameters}")



    def tag_players_v2(self):

        origin_fn = self.frames_controller.template_players.frame_number

        TagAlgorithm.tag_player_positions(self.frames_controller.frame_list, self.net)
        self.player_templates = self.set_player_templates()

        tag_alg = TagAlgorithm(self.frames_controller.frame_list, origin_fn, self.player_templates, self.player_features, self.net)

        tag_alg.tag()






        for frame in self.frames_controller.frame_list:
            for player in frame.players():
                player.tag = player.new_tag
                player.new_tag = None

        player_pos, player_idx = self.get_pos_and_idx()

        #Visuals.plot_player_positions(player_pos, player_idx)
        self.frames_controller.smooth_player_tags(player_pos, player_idx, len(self.frames_controller))
        self.load_players_boundaries()

    def get_player_positions_scaled(self, tag):
        perspective_points = [
            (self.players_boundaries_horizontal[Position.BOTTOM]['left'], self.players_boundaries_vertical[Position.BOTTOM]),
            (self.players_boundaries_horizontal[Position.BOTTOM]['right'], self.players_boundaries_vertical[Position.BOTTOM]),
            (self.players_boundaries_horizontal[Position.TOP]['left'], self.players_boundaries_vertical[Position.TOP]),
            (self.players_boundaries_horizontal[Position.TOP]['right'], self.players_boundaries_vertical[Position.TOP])
        ]

        real_world_points = [(0, 1), (1, 1), (0, 0), (1, 0)]
        H = compute_homography(perspective_points, real_world_points)

        player_coords = self.frames_controller.get_player_positions(tag)
        transformed_coords = transform_coordinates(H, player_coords)

        return transformed_coords













    def tag_players_in_frames(self, player_features):
        self.player_features = player_features
        self.player_templates = self.set_player_templates()
        #self.tag_player_positions()

        # Tag frames
        start_time = time.time()
        #player_pos, player_idx = rust_functions.tag_frames(self.frames_controller.frame_list, self.players,
                                                           #self.player_features)
        #self.tag_frames()
        #player_pos, player_idx = self.get_pos_and_idx()


        #Visuals.plot_player_positions(player_pos, player_idx)
        print(f"Frames tagged: {time.time() - start_time} seconds")
        self.frames_controller.smooth_player_tags(player_pos, player_idx, len(self.frames_controller))
        self.load_players_boundaries()


    def get_pos_and_idx(self, foot=False):

        positions = {}
        index = {}
        for player in self.player_templates:
            positions[player.tag] = []
            index[player.tag] = []

        for frame in self.frames_controller.frame_list:
            for player in frame.players():
                if not foot:
                    positions[player.tag].append((player.x, player.y))
                else:
                    positions[player.tag].append((player.x, player.get_foot()))
                index[player.tag].append(frame.frame_number)

        return positions, index

    def tag_player_positions(self):
        for frame in self.frames_controller.frame_list:
            for p in frame.players():
                if p.get_foot() < self.net.get_foot():
                    p.position = Position.TOP
                else:
                    p.position = Position.BOTTOM

            while len(frame.player_templates(Position.BOTTOM)) > 2:
                sorted(frame.player_templates(Position.BOTTOM), key=lambda p: p.get_foot())[0].position = Position.TOP
            while len(frame.player_templates(Position.TOP)) > 2:
                sorted(frame.player_templates(Position.TOP), key=lambda p: p.get_foot())[-1].position = Position.BOTTOM

    def tag_frames(self):
        self.propagate_tags()
        self.fill_empty_tags()


    def fill_empty_tags(self):

        last_player = None
        for tag in [p.tag for p in self.player_templates]:
            for frame in self.frames_controller.frame_list:
                player = frame.player(tag)
                if player is None and last_player is not None:
                    np = last_player.copy()
                    frame.add_object(np)
                    last_player = np
                if player is not None:
                    last_player = player




    def assign_tag_lowest_dist(self, players, templates):
        def get_player_features(tag):
            pf = self.player_features[str(int(tag))]
            return pf

        matches = []
        pairs = {}

        tags = {}
        for player in templates:
            tags[player.tag] = player


        for tag in tags.keys():
            for obj in players:
                player_in_frame_ft = get_player_features(obj.tag)
                dist = PlayerTemplate.features_distance(tags[tag].template_features, player_in_frame_ft)
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
            for player in players:
                if player.tag == match[1]:
                    player.new_tag = match[0]



    def propagate_tags(self):
        frame_list = self.frames_controller.frame_list


        players_lf = []
        for i in range(len(frame_list)):
            players = frame_list[i].player_templates()

            for p_lf in players_lf:
                for p in players:
                    if p_lf.tag == p.tag:
                        p.new_tag = p_lf.new_tag
            wo_new_tag = [p for p in players if p.new_tag is None]

            wo_new_tag_bottom = [p for p in wo_new_tag if p.position is Position.BOTTOM]
            wo_new_tag_top = [p for p in wo_new_tag if p.position is Position.TOP]

            if len(wo_new_tag_bottom) > 0:
                self.assign_tag_lowest_dist(wo_new_tag_bottom, [t for t in self.player_templates if t.position is Position.BOTTOM])
            if len(wo_new_tag_top) > 0:
                self.assign_tag_lowest_dist(wo_new_tag_top,
                                            [t for t in self.player_templates if t.position is Position.TOP])

            players_lf = players


        for frame in frame_list:
            for player in frame.players():
                player.tag = player.new_tag
                player.new_tag = None










    def load_players_boundaries(self):
        max_y = -1
        min_y = 1

        max_x_top = -1
        min_x_top = 1
        max_x_bottom = -1
        min_x_bottom = 1

        for frame in self.frames_controller.frame_list:

            players_ordered = sorted(frame.players(), key=lambda obj: obj.y + obj.height / 2)
            for i, player in enumerate(players_ordered):
                if len(frame.players()) == 4:
                    if i < 2:
                        player.position = Position.TOP
                    else:
                        player.position = Position.BOTTOM
                else:
                    if player.y + player.height / 2 > self.net.y + self.net.height / 2:
                        player.position = Position.BOTTOM
                    else:
                        player.position = Position.TOP

            for player in frame.players():
                if player.position == Position.TOP:
                    if player.y + player.height / 2 < min_y:
                        min_y = player.y + player.height / 2
                    if player.x - player.width / 2 < min_x_top:
                        min_x_top = player.x - player.width / 2
                    if player.x + player.width / 2 > max_x_top:
                        max_x_top = player.x + player.width / 2

                if player.position == Position.BOTTOM:
                    if player.y + player.height / 2 > max_y:
                        max_y = player.y + player.height / 2
                    if player.x - player.width / 2 < min_x_bottom:
                        min_x_bottom = player.x - player.width / 2
                    if player.x + player.width / 2 > max_x_bottom:
                        max_x_bottom = player.x + player.width / 2

        self.players_boundaries_vertical = {Position.TOP: min_y, Position.BOTTOM: max_y}
        self.players_boundaries_horizontal = {Position.TOP: {'left': min_x_top, 'right': max_x_top},
                                              Position.BOTTOM: {'left': min_x_bottom, 'right': max_x_bottom}}

    def get_players(self):
        return self.player_templates.copy()

    def set_net(self):
        best_net_frame = self.frames_controller.template_net
        self.net = [obj for obj in best_net_frame.objects if obj.class_label == Label.NET.value][0]

    def __str__(self):
        print("Game: " + str(len(self.frames_controller)) + " frames")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Game({str(len(self.frames_controller))})"

    def track_ball_v2(self):
        self.position_tracker = PositionTrackerV2(self.frames_controller, self.fps, self.net,
                                                  self.players_boundaries_vertical)
        self.points = self.position_tracker.points
        self.tracks = self.position_tracker.tracks

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
            mr_player.new_tag = idx_to_names[idx]
            game_player.position = mr_player.position

            players.append(game_player)

        return players
