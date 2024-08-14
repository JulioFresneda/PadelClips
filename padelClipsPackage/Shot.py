import math
from enum import Enum

class Position(Enum):
    TOP = 0
    BOTTOM = 1

class Category(Enum):
    NONE = 0
    GLOBE = 1
    SMASH = 2

class Shot:
    def __init__(self, pifs, position, hit=None, bottom_on_net=False):
        self.pifs = pifs
        self.position = position
        self.bottom_on_net = bottom_on_net
        if hit is None:
            self.calculate_hit()
        else:
            self.hit = hit




    def tag_hit_player(self, players):
        min_dist = float('inf')
        tag = None
        for player in players:
            if player.position == self.position:
                dist = self.get_distances(self.hit, player)
                if dist < min_dist:
                    min_dist = dist
                    tag = player.tag
        self.hit_player = tag



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