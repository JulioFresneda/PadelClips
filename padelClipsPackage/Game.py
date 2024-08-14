import time

from padelClipsPackage.Frame import Label
from padelClipsPackage.FramesController import FramesController
from padelClipsPackage.GameStats import GameStats
from padelClipsPackage.Object import PlayerTemplate
from padelClipsPackage.Point import Point

from padelClipsPackage.PositionTracker import PositionTrackerV2
from padelClipsPackage.Shot import Position
from padelClipsPackage.Visuals import Visuals
import rust_functions


class Game:
    def __init__(self, frames, fps, player_features):
        self.net = None
        self.players_boundaries = None
        self.fps = int(fps)
        self.frames_controller = FramesController(frames)

        self.load_player_info(player_features)
        self.set_net()

        Point.game = self
        self.track_ball_v2()

        #self.categorize_shots()
        print("Points loaded.")

        self.gameStats = GameStats(self.frames_controller, self.points, self.net)
        self.gameStats.print_game_stats()

    def load_player_info(self, player_features):
        self.player_features = player_features
        self.players = self.set_player_templates()

        # Tag frames
        start_time = time.time()
        player_features_dict = {str(int(key)): player_features[key] for key in player_features.files}
        player_pos, player_idx = rust_functions.tag_frames(self.frames_controller.frame_list, self.players,
                                                           player_features_dict)
        print(f"Frames tagged: {time.time() - start_time} seconds")
        self.frames_controller.smooth_player_tags(player_pos, player_idx, len(self.frames_controller))
        self.load_players_boundaries()

    def load_players_boundaries(self):
        max_y = -1
        min_y = 1
        for frame in self.frames_controller.frame_list:


            players_ordered = sorted(frame.players(), key=lambda obj: obj.y+obj.height/2)
            for i, player in enumerate(players_ordered):
                if len(frame.players()) == 4:
                    if i<2:
                        player.position = Position.TOP
                    else:
                        player.position = Position.BOTTOM
                else:
                    if player.y + player.height/2 > self.net.y + self.net.height/2:
                        player.position = Position.BOTTOM
                    else:
                        player.position = Position.TOP


            for player in frame.players():
                if player.y - player.height/2 < min_y and player.position == Position.TOP:
                    min_y = player.y - player.height/2
                if player.y + player.height/2 > max_y and player.position == Position.BOTTOM:
                    max_y = player.y + player.height/2

        self.players_boundaries = {Position.TOP: min_y, Position.BOTTOM: max_y}


    def get_players(self):
        return self.players.copy()





    def set_net(self):
        best_net_frame = self.frames_controller.template_net
        self.net = [obj for obj in best_net_frame.objects if obj.class_label == Label.NET.value][0]

    def __str__(self):
        print("Game: " + str(len(self.frames_controller)) + " frames")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Game({str(len(self.frames_controller))})"

    def track_ball_v2(self):
        self.position_tracker = PositionTrackerV2(self.frames_controller, self.fps, self.net, self.players_boundaries)
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
            players.append(game_player)

        return players
