import math
import matplotlib.pyplot as plt

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
    def calculate_variance(objects, only_vertical = True):
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
        self.variance_to_dynamic = 0.000005
        self.variance_distance = 5



    def get_distance_tracked(self):
        dist = 0.0
        for i, pif in enumerate(self.track):
            if i > 0:
                dist += PositionInFrame.distance_to(self.track[i-1], pif)
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
            #if len(self.track) < self.variance_distance:
            variance = PositionInFrame.calculate_variance(self.track)
            if variance > self.variance_to_dynamic:
                self.static = False

            #else:
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
        if class_name == Label.BALL:
            for i, frame in enumerate(self.frames):
                print("Tracking balls from frame " + str(i) + "/" + str(len(self.frames)), end='\r')
                self.track_ball(frame)
                self.close_tracks(frame.frame_number, tolerance=20)

            self.clean_tracks(self.closed_tracks)

            #self.plot_tracks([track for track in self.closed_tracks if track.static is False])
            self.plot_tracks(self.closed_tracks)

    def clean_tracks(self, tracks):
        self.remove_short_tracks(tracks, minimum_length=3)


        #self.remove_short_tracks(tracks, minimum_length=3)
        #self.remove_shadow_tracks(tracks, margin=0)
        self.keep_valuable_tracks(tracks, percentage=0.25)

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
        to_remove = [track for track in tracks if len(track.track) < minimum_length]
        for track in to_remove:
            tracks.remove(track)

    def close_tracks(self, frame_number, tolerance=5):
        to_close = [track for track in self.open_tracks if track.last_frame() + tolerance < frame_number]
        for track in to_close:
            track.check_static()
            self.closed_tracks.append(track)
            self.open_tracks.remove(track)

    def track_ball(self, frame):

        balls = frame.balls()
        balls_to_track = balls.copy()


        distances = {}
        for ball in balls:
            for track in self.open_tracks:
                distances[(ball, track)] = PositionInFrame.distance_to(PositionInFrame(ball.x, ball.y, None),
                                                                       track.last_pif())

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

    def plot_tracks(self, tracks, frame_start=-1, frame_end=float('inf')):  # Adding frame_rate parameter
        import matplotlib.pyplot as plt  # Ensure matplotlib is imported

        plt.figure(figsize=(20, 6))
        ax1 = plt.gca()  # Get the current axis
        for track in tracks:
            # Filter positions within the specified frame range
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]

            if frame_numbers:  # Only plot if there are frames in the range
                # Convert frame numbers to seconds
                time_seconds = [frame_number / self.fps for frame_number in frame_numbers]
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
