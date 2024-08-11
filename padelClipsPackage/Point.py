from enum import Enum

from padelClipsPackage import Track
import math

from padelClipsPackage.Shot import Shot, Position
from padelClipsPackage.aux import apply_kalman_filter

class Point:
    game = None

    def __init__(self, track: Track.Track()):
        if len(track.pifs) > 0:
            self.track = track
            self.shots = []

            self.cook_shots()

        else:
            self.track = Track.Track()
            self.shots = {}

    def cook_shots(self):
        all_pifs = self.track.pifs
        shots_top = []

        buffer_top = []
        buffer_bottom = []

        for pif in all_pifs:
            if pif.y > self.game.net.y + self.game.net.height / 2:
                if len(buffer_top) > 0:
                    #shot = ShotV2(buffer_top, Position.TOP)
                    #shot.tag_hit_player(self.game.frames_controller.get(shot.hit.frame_number).players())
                    shots_top.append(buffer_top)
                    buffer_top = []

                buffer_bottom.append(pif)

            else:
                if len(buffer_bottom) > 0:
                    shot = Shot(buffer_bottom, Position.BOTTOM)

                    self.shots.append(shot)
                    buffer_bottom = []
                buffer_top.append(pif)

        if len(buffer_bottom) > 0:
            shot = Shot(buffer_bottom, Position.BOTTOM)

            self.shots.append(shot)
        elif len(buffer_top) > 0:
            shots_top.append(buffer_top)

        shots_top_mountains = []
        for subshot in shots_top:
            shots_top_mountains.append(self.find_mountains(subshot))

        for mountains in shots_top_mountains:
            self.top_mountains_to_shots(mountains)

        for shot in self.shots:
            shot.tag_hit_player(self.game.frames_controller.get(shot.hit.frame_number).players())

        self.shots = sorted(self.shots, key=lambda shot: shot.hit.frame_number)
        shots_clean = self.shots
        last_shot = None
        for shot in self.shots:
            if last_shot is not None and last_shot.position == Position.BOTTOM and not last_shot.bottom_on_net:
                if shot.position == Position.BOTTOM and not shot.bottom_on_net:
                    shots_clean.remove(last_shot)
            last_shot = shot

        if shots_clean[-1].hit in shots_clean[-1].pifs[-3:]:
            shots_clean.remove(shots_clean[-1])

        self.shots = shots_clean

    def top_mountains_to_shots(self, shots_top_mountains, min_length=20, min_height=0.1, min_dist=30):
        if len(shots_top_mountains) == 1:
            self.shots.append(Shot(shots_top_mountains[0], Position.TOP))
        elif len(shots_top_mountains) == 2:
            last_local_min = Shot.find_last_local_minimum(shots_top_mountains[1])
            i = shots_top_mountains[1].index(last_local_min)

            while i > 0 and shots_top_mountains[1][i].y < self.game.players_boundaries[Position.TOP]:
                i -= 1

            self.shots.append(Shot(shots_top_mountains[1], position=Position.TOP, hit=shots_top_mountains[1][i]))

        else:
            count = len(shots_top_mountains) - 1
            top = True
            while count > 0:
                comply_min_lenght = shots_top_mountains[count][-1].frame_number - shots_top_mountains[count][
                    0].frame_number > min_length

                lowest = 1
                highest = -1
                for pif in shots_top_mountains[count]:
                    if pif.y < lowest:
                        lowest = pif.y
                    if pif.y > highest:
                        highest = pif.y
                comply_min_height = highest - lowest > min_height
                comply_min_dist = shots_top_mountains[count][-1].frame_number - shots_top_mountains[count][
                    0].frame_number >= min_dist

                if comply_min_lenght:
                    if top:
                        last_local_min = Shot.find_last_local_minimum(shots_top_mountains[count])
                        i = shots_top_mountains[count].index(last_local_min)

                        while i > 0 and shots_top_mountains[count][i].y < self.game.players_boundaries[Position.TOP]:
                            i -= 1

                        self.shots.append(
                            Shot(shots_top_mountains[count], position=Position.TOP,
                                 hit=shots_top_mountains[count][i]))
                        top = False

                    else:
                        last_local_max = Shot.find_last_local_maximum(shots_top_mountains[count])
                        i = shots_top_mountains[count].index(last_local_max)

                        while i > 0 and shots_top_mountains[count][i].y < self.game.players_boundaries[Position.TOP]:
                            i -= 1

                        self.shots.append(
                            Shot(shots_top_mountains[count], position=Position.BOTTOM,
                                 hit=shots_top_mountains[count][i], bottom_on_net=True))
                        top = True
                count -= 1

    def find_mountains(self, lst):
        if len(lst) < 3:
            return [lst]  # If the list is too small, return it as a single "mountain"

        mountains = []
        i = 1

        while i < len(lst) - 1:
            # Check if the current element is a local maximum
            if lst[i - 1].y > lst[i].y < lst[i + 1].y:
                # Find the left local minimum
                left_min = i - 1
                while left_min > 0 and lst[left_min - 1].y > lst[left_min].y:
                    left_min -= 1

                # Find the right local minimum
                right_min = i + 1
                while right_min < len(lst) - 1 and lst[right_min + 1].y > lst[right_min].y:
                    right_min += 1

                # Append the current mountain to the list
                mountains.append(lst[left_min:right_min + 1])

                # Skip to the next element after the current right_min to avoid overlap
                i = right_min
            else:
                i += 1

        return mountains

    def get_distances(self, ball_pifs, player_pifs, only_x=False):
        distances = []
        for ball, player in zip(ball_pifs, player_pifs):
            if only_x:
                distances.append(abs(ball.x - player.x))
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

    def duration(self):
        return self.end() - self.start()
    def start(self):
        return self.track.pifs[0].frame_number

    def end(self):
        return self.track.pifs[-1].frame_number

    def how_many_shots_by_player(self, tag):
        return len([s for s in self.shots if s.hit_player == tag])

    def tag_shots(self, tagger):
        tagger()

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
