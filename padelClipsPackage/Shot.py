import math
from enum import Enum

from padelClipsPackage.Object import PlayerPosition
from padelClipsPackage.PositionInFrame import PositionInFrame


class Position(Enum):
    TOP = 0
    BOTTOM = 1

class Category(Enum):
    NONE = 0
    GLOBE = 1
    SMASH = 2


class CategoryV2(Enum):
    NONE = 0
    END_GLOBE = 1
    START_GLOBE = 2
    SMASH = 3
    SERVE = 4

class Direction(Enum):
    UP = 0
    DOWN = 1

class ShotV2:
    game = None
    def __init__(self, pifs):
        self.pifs = pifs
        self.direction = self.get_direction()
        self.category = self.categorize()

        self.striker, self.receiver, self.striker_pif, self.receiver_pif = self.set_players()

        self.check_smash()


    def check_smash(self):
        if self.category is CategoryV2.NONE and self.striker is not None:
            if self.striker_pif.y < self.striker.y - self.striker.height/2:
                if self.striker.x - self.striker.width/2 < self.striker_pif.x < self.striker.x + self.striker.width/2:
                    self.category = CategoryV2.SMASH
    def set_players(self):
        self.frames_in_shot = ShotV2.game.frames_controller.frame_list[
                              self.pifs[0].frame_number-self.game.start:self.pifs[-1].frame_number + 1-self.game.start]
        stricker, s_pif = self.set_stricker()
        receiver, r_pif = self.set_receiver()

        return stricker, receiver, s_pif, r_pif

    def set_stricker(self):
        if self.category is not CategoryV2.END_GLOBE:
            for frame, pif in zip(self.frames_in_shot, self.pifs):
                if pif.y >= ShotV2.game.players_boundaries[Position.TOP] - 0.1:
                    if pif.y > self.game.net.y + self.game.net.height/2 + 0.25:
                        players = frame.players(positions=[Position.BOTTOM])
                    else:
                        players = frame.players()
                    distances = [self.get_distances(pif, player) for player in players]
                    return players[distances.index(min(distances))], pif
        return None, None

    def set_receiver(self):
        if self.category is not CategoryV2.START_GLOBE:
            for frame, pif in reversed(list(zip(self.frames_in_shot, self.pifs))):
                if pif.y >= ShotV2.game.players_boundaries[Position.TOP] - 0.1:
                    if pif.y > self.game.net.y + self.game.net.height / 2 + 0.25:
                        players = frame.players(positions=[Position.BOTTOM])
                    else:
                        players = frame.players()
                    distances = [self.get_distances(pif, player) for player in players]
                    return players[distances.index(min(distances))], pif
        return None, None

    def get_direction(self):
        if self.pifs[-1].y - self.pifs[0].y > 0:
            return Direction.DOWN
        else:
            return Direction.UP


    def categorize(self, min_velocity = 10):
        #speed = PositionInFrame.speed_list(self.pifs)
        if self.pifs[0].y < 0.1 and self.direction is Direction.DOWN:
            return CategoryV2.END_GLOBE
        elif self.pifs[-1].y < 0.1 and self.direction is Direction.UP:
            return CategoryV2.START_GLOBE
        else:
            return CategoryV2.NONE


    def get_distances(self, ball, player, method='euclidean'):
        # Values: euclidean, x, sum
        if method == 'x':
            return abs(ball.x - player.x)
        elif method == 'y':
            return abs(ball.y - player.y)
        elif method == 'euclidean':
            return Shot.euclidean_distance(ball.x, ball.y, player.x, player.y)
        elif method == 'sum':
            return sum(abs(ball.x-player.x)+abs(ball.y-player.y))
        elif method == 'both':
            return abs(ball.x-player.x), abs(ball.y-player.y)



class Shot:
    def __init__(self, pifs, position, hit=None, bottom_on_net=False):
        self.hit_player = None
        self.category = Category.NONE
        self.pifs = pifs
        self.position = position
        self.bottom_on_net = bottom_on_net
        if hit is None:
            self.calculate_hit()
        else:
            self.hit = hit

        self.check_globe()


    def check_globe(self, min_globe = 0.1):
        if min(self.pifs, key=lambda pif: pif.y).y <= min_globe:
            self.category = Category.GLOBE

    def check_smash(self, min_velocity = 10):
        if self.hit_player is not None:
            if self.hit_player.y - self.hit_player.height/2 >= self.hit.y:

                hit_index = self.pifs.index(self.hit)

                is_smash = False
                if len(self.pifs) > hit_index+1:
                    if PositionInFrame.speed(self.pifs[hit_index], self.pifs[hit_index+1]) > min_velocity:
                        is_smash = True
                if is_smash:
                    self.category = Category.SMASH

    def tag_hit_player(self, players):
        min_dist = float('inf')
        hit_player = None
        for player in players:
            if player.position == self.position:
                dist = self.get_distances(self.hit, player)
                if dist < min_dist:
                    min_dist = dist
                    hit_player = player
        self.hit_player = hit_player
        self.check_smash()



    def get_distances(self, ball, player, method='euclidean'):
        # Values: euclidean, x, sum
        if method == 'x':
            return abs(ball.x - player.x)
        elif method == 'y':
            return abs(ball.y - player.y)
        elif method == 'euclidean':
            return Shot.euclidean_distance(ball.x, ball.y, player.x, player.y)
        elif method == 'sum':
            return sum(abs(ball.x-player.x)+abs(ball.y-player.y))
        elif method == 'both':
            return abs(ball.x-player.x), abs(ball.y-player.y)


    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
    def calculate_hit(self):
        if self.position == Position.BOTTOM:
            self.hit = Shot.find_last_local_maximum(self.pifs)
        else:
            self.hit = Shot.find_last_local_minimum(self.pifs)

    @staticmethod
    def find_last_local_maximum(values):
        last_local_max = values[0]
        n = len(values)

        for i in range(n):
            # Check if the current element is a local maximum
            if (i == 0 and n > 1 and values[i].y > values[i + 1].y) or \
                    (i == n - 1 and n > 1 and values[i].y > values[i - 1].y) or \
                    (0 < i < n - 1 and values[i].y > values[i - 1].y and values[i].y > values[i + 1].y):
                last_local_max = values[i]

        return last_local_max

    @staticmethod
    def find_last_local_minimum(values):
        last_local_min = None
        n = len(values)

        if n == 0:
            return None  # Return None if the list is empty

        for i in range(n):
            # Check if the current element is a local minimum
            if (i == 0 and n > 1 and values[i].y < values[i + 1].y) or \
                    (i == n - 1 and n > 1 and values[i].y < values[i - 1].y) or \
                    (0 < i < n - 1 and values[i].y < values[i - 1].y and values[i].y < values[i + 1].y):
                last_local_min = values[i]

        # If no local minimum is found, return the first value
        return last_local_min if last_local_min is not None else values[0]

    @staticmethod
    def velocity(pif_a, pif_b, scale=10):
        distance = abs(pif_b.y - pif_a.y)
        time = abs(pif_b.frame_number - pif_a.frame_number)
        velocity = distance / time
        return velocity * scale


    def start(self):
        return self.pifs[0].frame_number

    def end(self):
        return self.pifs[-1].frame_number