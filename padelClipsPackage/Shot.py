import math
from enum import Enum

from padelClipsPackage.Object import PlayerPosition
from padelClipsPackage.PositionInFrame import PositionInFrame


class Position(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3





class Category(Enum):
    NONE = 0
    END_GLOBE = 1
    START_GLOBE = 2
    FULL_GLOBE = 5
    SMASH = 3
    SERVE = 4
    LOW_VOLLEY = 6


class Direction(Enum):
    UP = 0
    DOWN = 1


class Shot:
    game = None

    def __init__(self, pifs):
        self.frames_in_shot = None
        self.pifs = pifs
        self.inflexions = []
        self.direction = self.get_direction()

        self.category = self.categorize()

        self.striker, self.receiver, self.striker_pif, self.receiver_pif = self.set_players()

        self.check_category()

    def __len__(self):
        return self.pifs[-1].frame_number - self.pifs[0].frame_number

    def __str__(self):
        striker = "None" if self.striker is None else self.striker.tag
        print("Shot " + str(self.start()) + "/" + str(self.end()) + ", " + str(
            self.category) + ", " + striker + ", " + str(len(self.inflexions)) + " inflexions")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        striker = "None" if self.striker is None else self.striker.tag
        return "Shot " + str(self.start()) + "/" + str(self.end()) + ", " + str(
            self.category) + ", " + striker + ", " + str(len(self.inflexions)) + " inflexions"

    def start(self):
        return self.pifs[0].frame_number

    def end(self):
        return self.pifs[-1].frame_number

    @staticmethod
    def join_shots(shots_list):
        shot = shots_list[0]
        for s in shots_list[1:]:
            shot.pifs += s.pifs
            shot.frames_in_shot += s.frames_in_shot
            shot.inflexions.append(s.striker_pif)
        if shots_list[-1].category is Category.START_GLOBE:
            shot.category = Category.START_GLOBE
        shot.receiver = shots_list[-1].receiver
        shot.receiver_pif = shots_list[-1].receiver_pif

        return shot

    @staticmethod
    def join_globes(start, end):

        start.pifs += end.pifs
        start.frames_in_shot += end.frames_in_shot
        start.category = Category.FULL_GLOBE
        start.receiver = end.receiver
        start.receiver_pif = end.receiver_pif
        return start

    def check_category(self):
        if self.category is Category.NONE and self.striker is not None:
            if self.striker.x - self.striker.width / 2 < self.striker_pif.x < self.striker.x + self.striker.width / 2:
                if self.striker_pif.y < self.striker.y - self.striker.height / 2:
                    self.category = Category.SMASH
                elif self.striker_pif.y > self.striker.y:
                    self.category = Category.LOW_VOLLEY

    def set_players(self):
        self.frames_in_shot = Shot.game.frames_controller.frame_list[
                              self.pifs[0].frame_number - self.game.start:self.pifs[
                                                                              -1].frame_number + 1 - self.game.start]
        striker, s_pif = self.set_striker()
        receiver, r_pif = self.set_receiver()

        return striker, receiver, s_pif, r_pif

    def update_striker_with_conf_v2(self, conf):
        if self.category is not Category.END_GLOBE and self.striker_pif is not None:
            for frame, pif in zip(self.frames_in_shot, self.pifs):
                if pif == self.striker_pif:
                    players = frame.players()
                    distances = [(self.get_distances(pif, player), player) for player in players]
                    min_dist = self.get_distances(self.striker_pif, self.striker)

                    for d in sorted(distances, key=lambda dist: dist[0]):
                        if d[1].position is not self.striker.position and abs(d[0] - min_dist) <= conf:
                            self.striker = d[1]
                            self.striker_pif = pif
                            return True
        return False

    def update_striker_with_conf(self, conf):
        if self.category is not Category.END_GLOBE:
            for frame, pif in zip(self.frames_in_shot, self.pifs):
                if pif.y >= Shot.game.players_boundaries_vertical[Position.TOP] - 0.1:

                    players = frame.players()
                    distances = [(self.get_distances(pif, player), player) for player in players]
                    min_dist = min(distances, key=lambda d: d[0])[0]

                    for d in sorted(distances, key=lambda d: d[0]):
                        if d[1].position is not self.striker.position and abs(d[0] - min_dist) <= conf:
                            self.striker = d[1]
                            self.striker_pif = pif
                            return True
        return False

    def set_striker(self, position=None):
        if self.category is not Category.END_GLOBE:
            for frame, pif in zip(self.frames_in_shot, self.pifs):
                if pif.y >= Shot.game.players_boundaries_vertical[Position.TOP] - 0.1:
                    if pif.y > self.game.net.y + self.game.net.height / 2 + 0.25:
                        players = frame.players(positions=[Position.BOTTOM])
                        distances = [self.get_distances(pif, player, area=False) for player in players]
                        return players[distances.index(min(distances))], pif
                    else:

                        players = frame.players(positions=position)
                        distances = [self.get_distances(pif, player, area=False) for player in players]
                        return players[distances.index(min(distances))], pif

        return None, None

    def set_receiver(self):
        if self.category is not Category.START_GLOBE:
            for frame, pif in reversed(list(zip(self.frames_in_shot, self.pifs))):
                if pif.y >= Shot.game.players_boundaries_vertical[Position.TOP] - 0.1:
                    if pif.y > self.game.net.y + self.game.net.height / 2 + 0.25:
                        players = frame.players(positions=[Position.BOTTOM])
                    else:
                        players = frame.players()
                    distances = [self.get_distances(pif, player, area=False) for player in players]
                    return players[distances.index(min(distances))], pif
        return None, None

    def get_direction(self):
        if self.pifs[-1].y - self.pifs[0].y > 0:
            return Direction.DOWN
        else:
            return Direction.UP

    def categorize(self, min_velocity=10):
        #speed = PositionInFrame.speed_list(self.pifs)
        if self.pifs[0].y < 0.2 and self.direction is Direction.DOWN:
            return Category.END_GLOBE
        elif self.pifs[-1].y < 0.2 and self.direction is Direction.UP:
            return Category.START_GLOBE
        else:
            return Category.NONE

    def get_distances(self, ball, player, method='euclidean', area=False):

        ball_x = ball.x
        ball_y = ball.y
        if not area:
            player_x = player.x
            player_y = player.y
        else:
            if ball_x >= player.x:
                player_x = player.x + player.width / 2
            else:
                player_x = player.x - player.width / 2
            if ball_y >= player.y:
                player_y = player.y + player.height / 2
            else:
                player_y = player.y - player.height / 2

        # Values: euclidean, x, sum
        if method == 'x':
            return abs(ball_x - player_x)
        elif method == 'y':
            return abs(ball_y - player_y)
        elif method == 'euclidean':
            return Shot.euclidean_distance(ball_x, ball_y, player_x, player_y)
        elif method == 'sum':
            return sum(abs(ball_x - player_x) + abs(ball_y - player_y))
        elif method == 'both':
            return abs(ball_x - player_x), abs(ball_y - player_y)

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
