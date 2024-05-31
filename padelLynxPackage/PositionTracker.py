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

    def __str__(self):
        print("PIF " + str(self.x) + ", " + str(self.y) + " in " + str(self.frame_number))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"PIF({str(self.x)},{str(self.y)}) in {str(self.frame_number)}"


class Track:
    def __init__(self):
        self.track = []

    def last_frame(self):
        if len(self.track) == 0:
            return -1
        return self.track[-1].frame_number

    def last_pif(self):
        if len(self.track) == 0:
            return None
        return self.track[-1]

    def add_pif(self, pif):
        if len(self.track) == 0 or self.last_frame() < pif.frame_number:
            self.track.append(pif)
            return True
        else:
            return False

    def __str__(self):
        print("Track with" + str(len(self.track)) + " objects")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Track({str(len(self.track))} objects)"


class PositionTracker:
    def __init__(self, frames, class_name: Label):
        self.closed_tracks = []
        self.open_tracks = []



        self.frames = frames
        if class_name == Label.BALL:
            for i, frame in enumerate(self.frames):
                print("Tracking balls from frame " + str(i) + "/" + str(len(self.frames)), end='\r')
                self.track_ball(frame)
                self.close_tracks(frame.frame_number, tolerance=50)

            self.plot_tracks(self.closed_tracks, 0, 250)

    def close_tracks(self, frame_number, tolerance=5):
        to_close = [track for track in self.open_tracks if track.last_frame() + tolerance < frame_number]
        for track in to_close:
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

    def plot_tracks(self, tracks, frame_start, frame_end):
        plt.figure(figsize=(10, 6))
        for track in tracks:
            # Filter positions within the specified frame range
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]

            if frame_numbers:  # Only plot if there are frames in the range
                plt.plot(frame_numbers, y_positions, marker='o',
                         label=f'Track starting at frame {track.track[0].frame_number}')

        plt.xlabel('Frame Number')
        plt.ylabel('Y Position of Ball')
        plt.title('Track of Ball Y Positions Over Frames')
        plt.legend()
        plt.grid(True)
        plt.show()
