from enum import Enum

from padelClipsPackage import Track
import math

from padelClipsPackage.Object import PlayerPosition
from padelClipsPackage.Visuals import Visuals
from padelClipsPackage.aux import format_seconds


class Shot:
    def __init__(self, pifs, position_tag=None, player_position=None):
        self.pifs = pifs
        self.position = position_tag
        self.tag = None
        self.category = self.Category.NONE

        if self.position == Position.TOP:
            min_pif = min(self.pifs, key=lambda x: x.y)
            if min_pif != self.pifs[0] and min_pif != self.pifs[-1]:
                self.inflection = min_pif
            else:
                self.inflection = self.pifs[0]
        else:
            max_pif = max(self.pifs, key=lambda x: x.y)
            if max_pif != self.pifs[0] and max_pif != self.pifs[-1]:
                self.inflection = max_pif
            else:
                self.inflection = self.pifs[0]

    @staticmethod
    def velocity(pif_a, pif_b, scale=10):
        distance = abs(pif_b.y - pif_a.y)
        time = abs(pif_b.frame_number - pif_a.frame_number)
        velocity = distance / time
        return velocity * scale

    class Category(Enum):
        NONE = 0
        GLOBE = 1
        SMASH = 2

    def categorize(self, player_position):
        if self.check_globe():
            self.category = self.Category.GLOBE
        elif self.check_smash(player_position):
            self.category = self.Category.SMASH

    def check_smash(self, player_position, min_height_ball_player=0.15, min_length=30, min_velocity=2):
        middle = not self.is_inflection_at_start() and not self.is_inflection_at_end()
        if not middle:
            return False
        correct_pos_vertical = player_position.y - self.inflection.y >= min_height_ball_player
        correct_pos_horizontal = player_position.x - player_position.width / 2 <= self.inflection.x <= player_position.x + player_position.width / 2
        min_length_ok = self.last_frame() - self.first_frame() >= min_length

        velocity_ok = Shot.velocity(self.inflection, self.pifs[self.inflection_index() + 1]) >= min_velocity
        return correct_pos_vertical and correct_pos_horizontal and min_length_ok and velocity_ok

    def inflection_index(self):
        for i, pif in enumerate(self.pifs):
            if pif == self.inflection:
                return i

    def check_globe(self, min_height=0.02, min_frame_window=20):
        globe = False
        if self.is_inflection_at_start() or self.inflection.y > min_height:
            return globe
        for i in range(1, len(self.pifs) - 1):
            if self.inflection == self.pifs[i]:
                if self.pifs[i - 1].y <= min_height and self.inflection.frame_number - self.pifs[
                    i - 1].frame_number >= min_frame_window:
                    globe = True
                elif self.pifs[i + 1].y <= min_height and abs(
                        self.inflection.frame_number - self.pifs[i + 1].frame_number) >= min_frame_window:
                    globe = True
                break
        return globe

    def is_inflection_at_start(self):
        return self.inflection == self.pifs[0]

    def is_inflection_at_end(self):
        return self.inflection == self.pifs[-1]

    def tag_shot(self, tag):
        self.tag = tag

    def first_frame(self):
        return self.pifs[0].frame_number

    def last_frame(self):
        return self.pifs[-1].frame_number


def shortest_player(pif, position, max_distance=1):
    all_players = Point.game.frames_controller.get(pif.frame_number).players()
    all_players = sorted(all_players, key=lambda player: player.y)
    if all_players is not None:
        if position == 'over':
            if len(all_players) > 2:
                players = all_players[:2]
            else:
                players = all_players
        else:
            if len(all_players) > 2:
                players = all_players[2:]
            else:
                players = all_players

        if len(players) > 0:
            dis = Point.euclidean_distance(players[0].x, players[0].y, pif.x, pif.y)
            tag = players[0].tag

            if len(players) > 1:
                for p in players[1:]:
                    _dis = Point.euclidean_distance(p.x, p.y, pif.x, pif.y)
                    if _dis < dis:
                        dis = _dis
                        tag = p.tag

            if dis <= max_distance:
                return tag

    if position == 'under' and pif.y > 0.8:
        if pif.x > 0.5:
            return 'C'
        else:
            return 'D'


class Point:
    game = None

    def __init__(self, track: Track.Track()):
        if len(track.pifs) > 0:
            self.track = track
            self.shots_v2 = []

            self.load_shots_v2()

        else:
            self.track = Track.Track()
            self.shots = {}


    def load_shots_v2(self):
        tags = ['A', 'B', 'C', 'D']
        player_pos = {}
        player_distances = {}
        for tag in tags:
            player_pos[tag] = self.game.frames_controller.get_player_positions(tag=tag, start=self.start(), end=self.end())
            player_distances[tag] = self.get_distances(self.track.pifs, player_pos[tag], only_x=False)



        self.shots = {}
        for tag in tags:
            self.shots[tag] = []

        def get_min(mins):
            min = float('inf')
            index = None
            key = None
            for k in mins.keys():
                m = mins[k][0]
                if m < min:
                    min = m
                    index = mins[k][1]
                    key = k

            return min, key, index

        def get_current(index):
            current = {}
            for tag in tags:
                current[tag] = player_distances[tag][index]
            return current

        def get_dist(tag, index):
            return player_distances[tag][index]


        start_index = 0
        mins = {}
        for tag in tags:
            mins[tag] = None

        while start_index < len(self.track)-1:

            for tag in tags:
                mins[tag] = (get_dist(tag, start_index), start_index)

            for tag in tags:
                index = start_index
                found_min = False
                while index < len(self.track)-1 and not found_min:
                    index += 1
                    try:
                        dist = get_dist(tag, index)
                    except:
                        print(index)
                    if dist <= mins[tag][0]:
                        mins[tag] = (get_dist(tag, index), index)
                    else:
                        found_min = True




            min_player, min_tag, min_index = get_min(mins)
            self.shots[min_tag].append((self.track.pifs[min_index].frame_number, min_player))
            start_index = min_index + 1

            for tag in tags:
                mins[tag] = None







    def get_distances(self, ball_pifs, player_pifs, only_x=False):
        distances = []
        for ball, player in zip(ball_pifs, player_pifs):
            if only_x:
                distances.append(abs(ball.x-player.x))
            else:
                distances.append(Point.euclidean_distance(ball.x, ball.y, player.x, player.y))
        return distances



    def merge(self, point):
        self.track.pifs += point.pifs.pifs
        self.shots = self.shots + point.shots

    def __len__(self):
        return len(self.track)

    def __str__(self):
        print("Point from " + str(self.start()) + " to " + str(self.end()))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Point from " + str(self.start()) + " to " + str(self.end())

    def point_frames(self):
        return self.game.frames_controller.get(self.start(), self.end())

    def start(self):
        return self.track.pifs[0].frame_number

    def end(self):
        return self.track.pifs[-1].frame_number

    def how_many_shots_by_player(self, tag):
        return len([s for s in self.shots if s.tag == tag])

    def tag_shots(self, tagger):
        tagger()






    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance





    def position_in_field(self, y, net):
        if y < net.y+net.height/2:
            return Position.TOP
        else:
            return Position.BOTTOM
class Position(Enum):
    TOP = 0
    BOTTOM = 1