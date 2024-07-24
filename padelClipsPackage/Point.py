from enum import Enum

from padelClipsPackage import Track
import math

from padelClipsPackage.Object import PlayerPosition
from padelClipsPackage.aux import format_seconds


class Shot:
    def __init__(self, pifs, position_tag=None, player_position=None):
        self.pifs = pifs
        self.position = position_tag
        self.tag = None
        self.category = self.Category.NONE

        if self.position == 'over':
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
        return velocity*scale




    class Category(Enum):
        NONE = 0
        GLOBE = 1
        SMASH = 2

    def categorize(self, player_position):
        if self.check_globe():
            self.category = self.Category.GLOBE
        elif self.check_smash(player_position):
            self.category = self.Category.SMASH

    def check_smash(self, player_position, min_height_ball_player = 0.15, min_length = 30, min_velocity = 2):
        middle = not self.is_inflection_at_start() and not self.is_inflection_at_end()
        if not middle:
            return False
        correct_pos_vertical = player_position.y - self.inflection.y  >= min_height_ball_player
        correct_pos_horizontal = player_position.x - player_position.width/2 <= self.inflection.x <= player_position.x + player_position.width/2
        min_length_ok = self.last_frame() - self.first_frame() >= min_length

        velocity_ok = Shot.velocity(self.inflection, self.pifs[self.inflection_index()+1]) >= min_velocity
        return correct_pos_vertical and correct_pos_horizontal and min_length_ok and velocity_ok


    def inflection_index(self):
        for i, pif in enumerate(self.pifs):
            if pif == self.inflection:
                return i

    def check_globe(self, min_height = 0.02, min_frame_window = 20):
        globe = False
        if self.is_inflection_at_start() or self.inflection.y > min_height:
            return globe
        for i in range(1, len(self.pifs)-1):
            if self.inflection == self.pifs[i]:
                if self.pifs[i-1].y <= min_height and self.inflection.frame_number - self.pifs[i-1].frame_number >= min_frame_window:
                    globe = True
                elif self.pifs[i+1].y <= min_height and abs(self.inflection.frame_number - self.pifs[i+1].frame_number) >= min_frame_window:
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


class Point:
    net = None
    frames = None
    players = None

    def __init__(self, track: Track):
        if len(track.track) > 0:
            self.track = track
            self.shots = self.track_to_shots()

            self.tag_shots(self.tagger_inflexion)

            self.print_shots()
        else:
            self.track = []
            self.shots = []

    def merge(self, point):
        self.track.track += point.track.track
        self.shots = self.shots + point.shots

    def __len__(self):
        return len(self.shots)

    def __str__(self):
        print("Point from " + str(self.first_frame()) + " to " + str(self.last_frame()))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Point from " + str(self.first_frame()) + " to " + str(self.last_frame())

    def point_frames(self):
        return self.frames[self.first_frame():self.last_frame()]

    def first_frame(self):
        return self.shots[0].pifs[0].frame_number

    def last_frame(self):
        return self.shots[-1].pifs[-1].frame_number

    def how_many_shots_by_player(self, tag):
        return len([s for s in self.shots if s.tag == tag])


    def tag_shots(self, tagger):
        tagger()


    def tagger_closest(self):
        self.shots[0].tag_shot(self.shortest_player(self.shots[0].pifs[0], self.shots[0].position))
        if len(self.shots) > 1:
            for shot in self.shots[1:]:
                closest_value = 1.0
                closest_player = None
                if shot.position == 'over':
                    pos_allowed = [PlayerPosition.OVER_RIGHT, PlayerPosition.OVER_LEFT]
                else:
                    pos_allowed = [PlayerPosition.UNDER_RIGHT, PlayerPosition.UNDER_LEFT]
                for pif in shot.pifs:
                    players = self.frames[pif.frame_number].players(pos_allowed)
                    for p in players:
                        dist = self.euclidean_distance(p.x, p.y, pif.x, pif.y)
                        if dist < closest_value:
                            closest_value = dist
                            closest_player = p.tag
                shot.tag_shot(closest_player)





    def tagger_inflexion(self):
        self.shots[0].tag_shot(self.shortest_player(self.shots[0].pifs[0], self.shots[0].position))

        for shot in self.shots:

            tag = self.shortest_player(shot.inflection, shot.position)

            if tag is None:
                neighbours = {}
                for pif in shot.pifs:
                    ntag = self.shortest_player(pif, shot.position)
                    if ntag is not None:
                        neighbours[abs(pif.frame_number - shot.inflection.frame_number)] = ntag
                found = False
                for i in range(len(shot.pifs)):
                    if i in neighbours.keys():
                        if not found:
                            tag = neighbours[i]
                            found = True

            shot.tag_shot(tag)

    def print_shots(self):
        for shot in self.shots:
            if len(shot.pifs) > 0:
                print("Player " + str(shot.tag) + ": " + format_seconds(shot.inflection.frame_number,
                                                                        30) + " > " + format_seconds(
                    shot.pifs[0].frame_number, 30) + " -> " + format_seconds(shot.pifs[-1].frame_number, 30))
        print('\n')

    def shortest_player(self, pif, position, max_distance=1):
        all_players = Point.frames[pif.frame_number].players()
        all_players = sorted(all_players, key=lambda player: player.y)
        if all_players != None:
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

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def track_to_shots(self, min_length=3):
        shots = []
        buffer = []
        if len(self.track.track) > 0:
            initial_pos = self.position_over_the_net(self.track.track[0].y, Point.net)

            for pif in self.track.track:
                pif_pos = self.position_over_the_net(pif.y, Point.net)
                if pif_pos == initial_pos or initial_pos == 'middle' or pif_pos == 'middle' and initial_pos == 'under':
                    buffer.append(pif)
                    if initial_pos == 'middle':
                        initial_pos = self.position_over_the_net(pif.y, Point.net)

                else:
                    s = Shot(buffer, position_tag=initial_pos)

                    shots.append(s)
                    buffer = [pif]
                    initial_pos = self.position_over_the_net(pif.y, Point.net)
                    if initial_pos == 'middle':
                        initial_pos = 'under'
            if len(buffer) > 0:
                s = Shot(buffer, position_tag=initial_pos)

                shots.append(s)
            return shots
        else:
            print(shots)
            return shots

    def position_over_the_net(self, y, net):
        net_upper_pos = net.y - net.height / 2
        net_lower_pos = net.y + net.height / 2
        if y < net_upper_pos:
            return 'over'
        elif y > net_lower_pos:
            return 'under'
        else:
            return 'middle'
