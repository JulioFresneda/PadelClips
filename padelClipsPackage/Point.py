from enum import Enum

from padelClipsPackage import Track
import math

from padelClipsPackage.Shot import Shot, Position, Shot, Category
from padelClipsPackage.aux import apply_kalman_filter, apply_kalman_filter_pifs


class Point:
    game = None

    def __init__(self, track: Track.Track() = None):
        if track is not None and len(track.pifs) > 0:
            self.track = track
            self.shots = []

            self.cook_shots_v2()
            self.join_shots_same_striker_and_receiver()
            self.improve_striker_recognition()
            self.join_globes()

        else:
            self.track = Track.Track()
            self.shots = []


    def join_globes(self):
        new_shots = self.shots.copy()
        for i in range(len(self.shots) - 1):
            if self.shots[i].category is Category.START_GLOBE and self.shots[i + 1].category is Category.END_GLOBE:
                new_shots.append(Shot.join_globes(self.shots[i], self.shots[i+1]))
                new_shots.remove(self.shots[i])
                new_shots.remove(self.shots[i+1])

        self.shots = sorted(new_shots, key=lambda s: s.start())


    def improve_striker_recognition(self):

        # In serve, last inflexion under net
        for shot in self.shots:
            if shot.category is Category.SERVE:
                for inf in reversed(shot.inflexions):
                    if inf.y > Shot.game.net.y + Shot.game.net.height / 2:
                        shot.striker_pif = inf
                        break

        # If three consecutive TOP/BOTTOMs, the middle one is different
        for i in range(1, len(self.shots)-1):
            left = self.shots[i-1]
            middle = self.shots[i]
            right = self.shots[i+1]
            changed = None

            is_not_changed = (left != changed and middle != changed and right != changed)
            is_not_none = left.striker is not None and right.striker is not None and middle.striker is not None

            if is_not_none:
                same_pos = left.striker.position == right.striker.position == middle.striker.position

                if is_not_changed and same_pos:
                    middle.update_striker_with_conf_v2(conf=0.15)
                    changed = middle


        new_shots = self.shots.copy()
        for i in range(1, len(self.shots) - 1):
            if self.shots[i].striker is not None and self.shots[i + 1].striker is not None:
                if self.shots[i].striker.tag == self.shots[i + 1].striker.tag:
                    new_shots.append(Shot.join_shots([self.shots[i - 1], self.shots[i]]))
                    new_shots.remove(self.shots[i - 1])
                    new_shots.remove(self.shots[i])

        self.shots = sorted(new_shots, key=lambda s: s.start())






    def join_shots_same_striker_and_receiver(self):
        grouped_objects = []
        current_group = []

        for shot in self.shots:
            if not current_group:
                # Start a new group with the first object
                current_group.append(shot)
            elif shot.striker is not None and current_group[-1].striker is not None and shot.striker.tag == current_group[-1].striker.tag and shot.category is Category.NONE:
                current_group.append(shot)
            else:
                # Finish the current group and start a new one
                grouped_objects.append(current_group)
                current_group = [shot]

        # Don't forget to add the last group
        if current_group:
            grouped_objects.append(current_group)

        new_shots = []
        for group in grouped_objects:
            new_shots.append(Shot.join_shots(group))


        self.shots = new_shots


    def cook_shots_v2(self, min_length=5):
        all_pifs = self.track.pifs
        shots = Point.segment_by_extrema(all_pifs, key_func_x=lambda pif: pif.x, key_func_y=lambda pif: pif.y)


        shots = self.merge_shots(shots)


        self.shots = []
        for shot in shots:
            if shot[-1].frame_number - shot[0].frame_number >= min_length:
                new_shot = Shot(shot)
                self.shots.append(new_shot)
        if len(self.shots) > 0:
            if self.shots[0].category is Category.NONE:
                self.shots[0].category = Category.SERVE


    def merge_shots(self, shots):
        result = []
        i = 0
        threshold = self.game.net.y + self.game.net.height/2
        while i < len(shots):
            current_list = shots[i]

            # If we're not at the last list, check if the next list exists and the conditions hold
            if i < len(shots) - 1:
                next_list = shots[i + 1]

                # Check if all elements in the current list are below the threshold
                # and all elements in the next list are above or equal to the threshold
                if all(x.y < threshold for x in current_list) and all(y.y >= threshold for y in next_list):
                    # If conditions are met, join the two lists
                    result.append(current_list + next_list)
                    i += 2  # Skip the next list as it's already joined
                else:
                    result.append(current_list)
                    i += 1
            else:
                # Append the last list if no more lists to check
                result.append(current_list)
                i += 1
        return result



    @staticmethod
    def split_point_list(point, shots):
        subpoints = []
        start_index = 0

        for shot in shots:
            segment = point.shots[start_index:point.shots.index(shot)]
            subpoints.append(segment)
            start_index = point.shots.index(shot) + 1

        # Add the final segment
        if len(point.shots) > start_index:
            subpoints.append(point.shots[start_index:])

        subpoints_ready = []
        for subpoint in subpoints:
            new_point = Point()
            new_point.shots = subpoint
            for shot in subpoint:
                new_point.track.pifs += shot.pifs
            subpoints_ready.append(new_point)
        return subpoints_ready




    @staticmethod
    def split_point(point, shot):
        left_track = []
        left_shots = []
        right_track = []
        right_shots = []

        for pif in point.track.pifs:
            if pif.frame_number < shot.pifs[0].frame_number:
                left_track.append(pif)
            elif pif.frame_number > shot.pifs[-1].frame_number:
                right_track.append(pif)

        index = point.shots.index(shot)
        for i, shot in enumerate(point.shots):
            if i < index:
                left_shots.append(shot)
            elif i > index:
                right_shots.append(shot)

        point_left = Point()
        point_left.track.pifs = left_track
        point_left.shots = left_shots

        point_right = Point()
        point_right.track.pifs = right_track
        point_right.shots = right_shots

        return point_left, point_right

    @staticmethod
    def find_local_extrema(arr, key_func_y, key_func_x):
        """Find indices of local minima and maxima in a list based on key functions for y and x."""
        extrema_indices = []

        n = len(arr)
        arr = apply_kalman_filter_pifs(arr, obs=0.2, trans=0.05)

        for i in range(1, n - 1):
            prev_val_y = key_func_y(arr[i - 1])
            curr_val_y = key_func_y(arr[i])
            next_val_y = key_func_y(arr[i + 1])

            prev_val_x = key_func_x(arr[i - 1])
            curr_val_x = key_func_x(arr[i])
            next_val_x = key_func_x(arr[i + 1])

            # Check for local maximum considering both y and x
            if (prev_val_y < curr_val_y > next_val_y) or (prev_val_x < curr_val_x > next_val_x):
                extrema_indices.append(i)
            # Check for local minimum considering both y and x
            elif (prev_val_y > curr_val_y < next_val_y) or (prev_val_x > curr_val_x < next_val_x):
                extrema_indices.append(i)

        return extrema_indices

    @staticmethod
    def segment_by_extrema(arr, key_func_x, key_func_y):
        """Segment the list into parts that start and end with a local extremum based on a key function."""
        extrema_indices = Point.find_local_extrema(arr, key_func_x, key_func_y)
        segments = []

        # Adding the start and end points for full segmentation
        extrema_indices = [0] + extrema_indices + [len(arr) - 1]

        for i in range(len(extrema_indices) - 1):
            start = extrema_indices[i]
            end = extrema_indices[i + 1]
            segments.append(arr[start:end + 1])

        return segments


    def cook_shots(self):
        all_pifs = self.track.pifs
        shots_top = []

        buffer_top = []
        buffer_bottom = []

        for pif in all_pifs:
            if pif.y > self.game.net.y + self.game.net.height / 2:
                if len(buffer_top) > 0:
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
            shots_top_mountains.append(Point.find_mountains(subshot))

        for mountains in shots_top_mountains:
            self.top_mountains_to_shots(mountains)

        for shot in self.shots:
            shot.tag_hit_player(self.game.frames_controller.get(shot.hit.frame_number).player_templates())

        self.shots = sorted(self.shots, key=lambda shot: shot.hit.frame_number)
        shots_clean = self.shots
        last_shot = None
        for shot in self.shots:
            if last_shot is not None and last_shot.position == Position.BOTTOM and not last_shot.bottom_on_net:
                if shot.position == Position.BOTTOM and not shot.bottom_on_net:
                    shots_clean.remove(last_shot)
            last_shot = shot

        #if shots_clean[-1].hit in shots_clean[-1].pifs[-3:]:
        #    shots_clean.remove(shots_clean[-1])

        self.shots = shots_clean





    def top_mountains_to_shots(self, shots_top_mountains, min_length=20, min_height=0.1, min_dist=30):
        if len(shots_top_mountains) == 1:
            self.shots.append(Shot(shots_top_mountains[0], Position.TOP))
        elif len(shots_top_mountains) == 2:
            last_local_min = Shot.find_last_local_minimum(shots_top_mountains[1])
            i = shots_top_mountains[1].index(last_local_min)

            while i > 0 and shots_top_mountains[1][i].y < self.game.players_boundaries_vertical[Position.TOP]:
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

                        while i > 0 and shots_top_mountains[count][i].y < self.game.players_boundaries_vertical[Position.TOP]:
                            i -= 1

                        self.shots.append(
                            Shot(shots_top_mountains[count], position=Position.TOP,
                                 hit=shots_top_mountains[count][i]))
                        top = False

                    else:
                        last_local_max = Shot.find_last_local_maximum(shots_top_mountains[count])
                        i = shots_top_mountains[count].index(last_local_max)

                        while i > 0 and shots_top_mountains[count][i].y < self.game.players_boundaries_vertical[Position.TOP]:
                            i -= 1

                        self.shots.append(
                            Shot(shots_top_mountains[count], position=Position.BOTTOM,
                                 hit=shots_top_mountains[count][i], bottom_on_net=True))
                        top = True
                count -= 1

    @staticmethod
    def find_mountains(lst, smooth=False, max_height = 1):
        if smooth:

            lst = apply_kalman_filter_pifs(lst, obs=0.1, trans=0.03)

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

        mountains_height_comply = []
        for mountain in mountains:
            min_y = min(mountain, key=lambda pif: pif.y).y
            max_y = max(mountain, key=lambda pif: pif.y).y
            if abs(max_y-min_y) < max_height:
                mountains_height_comply.append(mountain)
        return mountains_height_comply

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
        try:
            return self.track.pifs[-1].frame_number
        except:
            print(self.shots)

    def how_many_shots_by_player(self, tag):
        return len([s for s in self.shots if s.striker is not None and s.striker.tag == tag])

    def tag_shots(self, tagger):
        tagger()

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
