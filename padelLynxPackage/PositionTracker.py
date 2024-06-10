import math
import matplotlib.pyplot as plt
import matplotlib
from padelLynxPackage.Frame import *


class PositionInFrame:
    def __init__(self, x, y, frame_number):
        self.x = x
        self.y = y
        self.frame_number = frame_number

    def nearest(self, positions_in_frame):
        nearest = PositionInFrame(float('inf'), float('inf'), -1)
        for pif in positions_in_frame:
            if PositionInFrame.distance_to(self, pif) < PositionInFrame.distance_to(self, nearest):
                nearest = pif
        return nearest

    @staticmethod
    def distance_to(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def calculate_variance(objects, only_vertical=True):
        distances = []
        n = len(objects)
        for i in range(n):
            for j in range(i + 1, n):  # Avoid computing the same distance twice
                if not only_vertical:
                    distance = PositionInFrame.distance_to(objects[i], objects[j])
                else:
                    distance = abs(objects[i].y - objects[j].y)
                distances.append(distance)

        # Calculate the mean of the distances
        mean_distance = sum(distances) / len(distances)

        # Calculate the variance
        variance = sum((x - mean_distance) ** 2 for x in distances) / len(distances)
        return variance

    def __str__(self):
        print("PIF " + str(self.x) + ", " + str(self.y) + " in " + str(self.frame_number))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"PIF({str(self.x)},{str(self.y)}) in {str(self.frame_number)}"


class Track:
    def __init__(self):
        self.track = []
        self.static = True
        self.globe = False
        self.variance_to_dynamic = 0.000005
        self.variance_distance = 5


    def get_pif(self, frame_number):
        pass
    def get_distance_tracked(self):
        dist = 0.0
        for i, pif in enumerate(self.track):
            if i > 0:
                dist += PositionInFrame.distance_to(self.track[i - 1], pif)
        return dist



    def purge_static_subtracks(self):
        if len(self.track) > self.variance_distance:
            subtracks = [self.track[i:i + self.variance_distance] for i in
                         range(0, len(self.track) - self.variance_distance)]

            st_to_remove = []

            for subtrack in subtracks:
                variance = PositionInFrame.calculate_variance(subtrack)
                if variance < self.variance_to_dynamic:
                    st_to_remove.append(subtrack)

            st_list = []
            for st_tr in st_to_remove:
                for pif in st_tr:
                    if pif not in st_list:
                        st_list.append(pif)

            subtracks = self.split_and_remove(self.track.copy(), st_list)
            return subtracks
        return []

    def split_and_remove(self, items, to_remove):
        result = []  # This will hold all the sublists
        current_sublist = []  # This holds the current sublist being constructed

        for item in items:
            if item in to_remove:
                if current_sublist:  # Only add the sublist if it's not empty
                    result.append(current_sublist)
                    current_sublist = []  # Reset for the next sublist
            else:
                current_sublist.append(item)  # Add item to the current sublist

        if current_sublist:  # Add the last sublist if not empty
            result.append(current_sublist)

        return result

    def check_static(self):
        if len(self.track) > 1:
            # if len(self.track) < self.variance_distance:
            variance = PositionInFrame.calculate_variance(self.track)
            if variance > self.variance_to_dynamic:
                self.static = False

            # else:
            #    subtracks = [self.track[i:i + self.variance_distance] for i in range(0, len(self.track)-self.variance_distance)]
            #    for subtrack in subtracks:
            #        variance = PositionInFrame.calculate_variance(subtrack)
            #        if variance > self.variance_to_dynamic:
            #            self.static = False

    def last_frame(self):
        if len(self.track) == 0:
            return -1
        return self.track[-1].frame_number

    def first_frame(self):
        if len(self.track) == 0:
            return -1
        return self.track[0].frame_number

    def last_pif(self):
        if len(self.track) == 0:
            return None
        return self.track[-1]

    def add_pif(self, pif, check_static=False):
        if len(self.track) == 0 or self.last_frame() < pif.frame_number:
            self.track.append(pif)
            if check_static:
                self.check_static()
            return True
        else:
            return False

    def __str__(self):
        print("Track with" + str(len(self.track)) + " objects")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Track({str(len(self.track))} objects)"


class PositionTracker:
    def __init__(self, frames, fps, class_name: Label):
        self.closed_tracks = []
        self.open_tracks = []
        self.fps = fps
        self.frames = frames

        self.manage_tracks()

        self.identify_breaks()


    def identify_breaks(self):
        pass



    def manage_tracks(self):
        for i, frame in enumerate(self.frames):
            if i % 100 == 0:
                print("Tracking balls from frame " + str(i) + "/" + str(len(self.frames)), end='\r')
            self.track_ball(frame)
            self.close_tracks(frame.frame_number, tolerance=10)

        clean = self.clean_tracks(self.closed_tracks)
        self.closed_tracks = clean

        #self.plot_tracks(self.closed_tracks, frame_start=3600, frame_end=9000)

        tracks_with_globe = self.detect_globes(self.closed_tracks)
        self.closed_tracks = tracks_with_globe
        self.globes = [track for track in self.closed_tracks if track.globe]

        # self.plot_tracks([track for track in self.closed_tracks if track.static is False])
        self.plot_tracks(self.closed_tracks, frame_start=3600, frame_end=9000, print_globes=False)

    def detect_globes(self, tracks):
        starts_index = {}
        end_index = {}
        for i, track in enumerate(tracks):
            if self.is_start_globe(track):
                starts_index[i] = track.last_frame()
            if self.is_end_globe(track):
                end_index[i] = track.first_frame()

        joins = []
        for e_pos, e_frame in sorted(end_index.items(), key=lambda item: item[1]):
            closest_pos = -1
            closest_frame = -1
            for s_pos, s_frame in sorted(starts_index.items(), key=lambda item: item[1], reverse=True):
                if s_frame > closest_frame and s_frame < e_frame:
                    closest_frame = s_frame
                    closest_pos = s_pos
            if closest_pos != -1:
                joins.append([closest_pos, e_pos])
                for s_pos, s_frame in starts_index.copy().items():
                    if s_frame < e_frame:
                        starts_index.pop(s_pos)



        joins = self.merge_lists(joins)
        print(joins)

        new_tracks = tracks.copy()
        for join in joins:
            globe = Track()
            globe.globe = True
            for i in join:
                globe.track += tracks[i].track


            index = new_tracks.index(tracks[join[0]])

            for i in join:
                new_tracks.remove(tracks[i])
            new_tracks.insert(index, globe)


        return new_tracks

    def merge_lists(self, lists):
        if not lists:
            return lists

        merged = [lists[0]]  # Start with the first list

        for current in lists[1:]:
            if merged[-1][-1] == current[
                0]:  # Check if last element of the last merged list equals the first element of the current list
                merged[-1].extend(current[1:])  # Extend the last merged list with the rest of the current list
            else:
                merged.append(current)  # If no match, just add the current list to merged list

        return merged

    def is_start_globe(self, track: Track, min=3, high=0.2):
        if len(track.track) >= min and track.track[-1].y < high:
            start_globe = track.track[-min:]
            is_globe = start_globe[0].y > start_globe[-1].y
            return is_globe
        return False

    def is_end_globe(self, track: Track, min=3, high=0.2):
        if len(track.track) >= min and track.track[0].y < high:
            end_globe = track.track[:min]
            is_globe = end_globe[0].y < end_globe[-1].y
            return is_globe
        return False

    def clean_tracks(self, tracks):
        clean = tracks.copy()
        self.remove_short_tracks(clean, minimum_length=2)
        self.remove_static_balls(clean, minimum_length=3)

        # self.remove_short_tracks(tracks, minimum_length=3)
        # self.remove_shadow_tracks(tracks, margin=0)
        # self.keep_valuable_tracks(tracks, percentage=0.25)


        return clean

    def remove_static_balls(self, tracks, minimum_length = 3):
        for track in tracks:
            track.track = self.filter_consecutive_objects_with_min_count(track.track, min_count=minimum_length)

    def filter_consecutive_objects_with_min_count(self, objects, variance=0.005, min_count=3):
        filtered_objects = []
        current_group = []

        for i in range(len(objects)):
            if i == 0:
                # Start the first group
                current_group.append(objects[i])
            else:
                # Calculate the difference in y positions
                y_diff = abs(objects[i].y - objects[i - 1].y)

                if y_diff <= variance:
                    # Continue the current group
                    current_group.append(objects[i])
                else:
                    # Check the size of the current group before ending it
                    if len(current_group) < min_count:
                        filtered_objects.extend(current_group)
                    # Start a new group
                    current_group = [objects[i]]

        # Check the last group collected
        if len(current_group) < min_count:
            filtered_objects.extend(current_group)

        return filtered_objects



    def keep_valuable_tracks(self, tracks, percentage=0.5):
        max_distance = max(track.get_distance_tracked() for track in tracks)

        # Step 2: Set the threshold as a percentage of the maximum value
        threshold = max_distance * percentage

        # Step 3: Filter the list to remove objects below the threshold
        filtered_objects = [track for track in tracks if track.get_distance_tracked() < threshold]
        for not_valuable in filtered_objects:
            tracks.remove(not_valuable)

    def remove_shadow_tracks(self, tracks, margin=0):
        shadow = []
        for check_shadow in tracks:
            for track in tracks:
                if check_shadow not in shadow and track.first_frame() + margin < check_shadow.first_frame() and check_shadow.last_frame() + margin < track.last_frame():
                    shadow.append(check_shadow)

        for track in shadow:
            tracks.remove(track)

    def remove_short_tracks(self, tracks, minimum_length=3):
        to_remove = [track for track in tracks if len(track.track) <= minimum_length]
        for track in to_remove:
            tracks.remove(track)

    def close_tracks(self, frame_number, tolerance=5):
        to_close = [track for track in self.open_tracks if track.last_frame() + tolerance < frame_number]
        for track in to_close:
            track.check_static()
            self.closed_tracks.append(track)
            self.open_tracks.remove(track)

    def track_ball(self, frame, max_distance = 0.3):

        balls = frame.balls()
        balls_to_track = balls.copy()

        distances = {}
        for ball in balls:
            for track in self.open_tracks:
                distance = PositionInFrame.distance_to(PositionInFrame(ball.x, ball.y, None),
                                                                track.last_pif())

                if max_distance > distance:
                    distances[(ball, track)] = distance

        while len(balls_to_track) > 0 and len(distances.keys()) > 0:
            s_ball, s_track = self.get_shortest(distances)
            s_track.add_pif(PositionInFrame(s_ball.x, s_ball.y, frame.frame_number))
            balls_to_track.remove(s_ball)

        for ball in balls_to_track:
            track = Track()
            track.add_pif(PositionInFrame(ball.x, ball.y, frame.frame_number))
            self.open_tracks.append(track)

    def get_shortest(self, distances):
        shortest = float('inf')
        s_ball, s_track = None, None
        for (ball, track), distance in distances.items():
            if distance < shortest:
                shortest = distance
                s_ball = ball
                s_track = track

        distances_keys = list(distances.keys()).copy()
        for (ball, track) in distances_keys:
            if ball == s_ball or track == s_track:
                distances.pop((ball, track))

        return s_ball, s_track

    def plot_tracks(self, tracks, frame_start=-1, frame_end=float('inf'), print_globes = False):  # Adding frame_rate parameter
        # Ensure matplotlib is imported
        matplotlib.use('TkAgg')
        plt.figure(figsize=(20, 6))
        ax1 = plt.gca()  # Get the current axis
        for track in tracks:
            # Filter positions within the specified frame range
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]

            if frame_numbers:  # Only plot if there are frames in the range
                # Convert frame numbers to seconds
                time_seconds = [pif.frame_number / self.fps for pif in track.track if
                                frame_start <= pif.frame_number <= frame_end]

                if print_globes:
                    color = 'orange' if track.globe else 'blue'

                    ax1.plot(time_seconds, y_positions, marker='o',
                         label=f'Track starting at frame {track.track[0].frame_number}', color=color)
                else:
                    ax1.plot(time_seconds, y_positions, marker='o',
                             label=f'Track starting at frame {track.track[0].frame_number}')


        ax1.set_xlabel('Time in Seconds')  # Primary x-axis for time in seconds
        ax1.set_ylabel('Y Position of Ball')
        ax1.set_title('Track of Ball Y Positions Over Time')
        ax1.legend()
        ax1.grid(True)

        max_time = max(time_seconds) if time_seconds else 0  # Calculate the max time if time_seconds is not empty
        ax1.set_xticks(range(0, int(max_time) + 1, 1))

        # Create secondary x-axis
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.set_xlabel('Frame Number')
        # Set the limits for the secondary x-axis
        ax2.set_xlim(ax1.get_xlim()[0] * self.fps, ax1.get_xlim()[1] * self.fps)
        ax2.set_xticks(ax1.get_xticks() * self.fps)  # Align ticks with primary axis

        plt.show()
