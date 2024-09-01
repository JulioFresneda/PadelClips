import json
from collections import defaultdict

import matplotlib
import numpy as np

from padelClipsPackage.Point import Point
from padelClipsPackage.PositionInFrame import PositionInFrame
from padelClipsPackage.Shot import Position
from padelClipsPackage.Track import *
from padelClipsPackage.Visuals import Visuals
from padelClipsPackage.predictPointModel import PredictPointModel
from padelClipsTraining.predict import train_predict


class PositionTrackerV2:
    def __init__(self, frames_controller, fps, net, players_boundaries):

        self.fps = fps
        self.frames_controller = frames_controller
        self.players_boundaries = players_boundaries
        self.net = net
        # Load phase: Equivalence track-tag
        self.tracks = self.load_tracks(delete_statics=False)
        # Merge phase: Join consecutive tracks


        # LEARN PHASE
        #with open("/home/juliofgx/PycharmProjects/PadelClips/dataset/padel_pove/2set/points.json") as f:
        #    actual_points = json.load(f)['frames']
        #ppm = PredictPointModel(self.tracks, actual_points)
        #ppm.initialize_df()

        # Clean phase: Detect split conditions, like empty seconds, multiple non-static balls...
        #self.clean_tracks()
        #self.remove_static_tracks()
        self.merge_tracks()
        #self.points = self.track_to_points()
        #

        # PREDICT FROM MODEL
        #Visuals.plot_tracks_with_net_and_players(self, self.net, self.players_boundaries, frame_start=23700, frame_end=25200)
        #train_predict.generate_features_df(self.tracks)
        self.predict_points(min_duration=0)

    def remove_static_tracks(self, min_frames=60, min_diff_allowed=0.005):
        static = {}
        for track in self.tracks:
            if len(track) > min_frames:
                static_idx = []

                start_static_fn = track.pifs[0].frame_number
                end_static_fn = 0
                counter = 0

                for i in range(1, len(track.pifs)):
                    if abs(track.pifs[i].y - track.pifs[i - 1].y) < min_diff_allowed:
                        end_static_fn = track.pifs[i].frame_number
                        counter += 1
                    else:
                        if counter > min_frames:
                            static_idx.append((start_static_fn, end_static_fn))
                        start_static_fn = track.pifs[i].frame_number
                        counter = 0
                static[track] = static_idx

        subtracks = []
        for track, static_moments in static.items():
            subtracks += Track.split_track(track, static_moments)
        for track in self.tracks:
            if track not in static.keys():
                subtracks.append(track)

        self.tracks = sorted(subtracks, key=lambda track: track.start())

    def predict_points(self, min_duration):
        all_pifs = []
        for track in self.tracks:
            for pif in track.pifs:
                all_pifs.append(pif)
        all_pifs = sorted(all_pifs, key=lambda pif: pif.frame_number)
        tracks_predicted = PredictPointModel.predict(all_pifs)

        pif_index = defaultdict(list)
        for pif in all_pifs:
            pif_index[pif.frame_number].append(pif)

        # Process tracks
        new_tracks = []

        for track in tracks_predicted:
            new_track = Track()
            for frame in range(track['start_frame'], track['end_frame'] + 1):
                if frame in pif_index:
                    for pif in pif_index[frame]:
                        new_track.add_pif(pif)
            new_tracks.append(new_track)

        self.tracks = new_tracks
        self.points = []
        for track in self.tracks:
            if len(track) >= min_duration:
                new_point = Point(track)
                self.points.append(new_point)







    def remerge_tracks(self, margin=20):
        track_list = sorted(self.tracks.copy(), key=lambda obj: obj.start())

        grouped = []
        current_group = [track_list[0]]

        for i in range(1, len(track_list)):
            if track_list[i].start() <= track_list[i - 1].end() + margin:
                current_group.append(track_list[i])
            else:
                grouped.append(current_group)
                current_group = [track_list[i]]

        grouped.append(current_group)

        tracks_grouped = []
        for group in grouped:
            if len(group) == 1:
                tracks_grouped += group
            else:
                merged_track = Track.join_tracks(group, group[0].tag)
                tracks_grouped.append(merged_track)

        self.tracks = tracks_grouped




    def get_player_position_over_time(self, tag, axis='y', start=0, end=float('inf')):
        player = []
        fn = []

        if end == float('inf'):
            frames = self.frames_controller.frame_list[start:]
        else:
            frames = self.frames_controller.get(start, end)
        for frame in frames:
            for p in frame.players():
                if p.tag == tag:
                    if axis == 'y':
                        player.append(1 - p.y)
                    elif axis == 'x':
                        player.append(p.x)

                    fn.append(frame.frame_number)

        return player, fn



    def track_to_points(self, min_duration = 180):
        points = []
        for track in self.tracks:
            if len(track) != 0 and track.pifs[-1].frame_number - track.pifs[0].frame_number >= min_duration:

                new_point = Point(track)
                points.append(new_point)



        return [point for point in points if point.duration() >= min_duration]


    def split_points_if_needed(self, points):
        new_points = []
        for point in points:
            shots_to_split = []
            for shot in point.shots:
                if shot.position == Position.BOTTOM and not shot.bottom_on_net:
                    mountains = Point.find_mountains(shot.pifs)
                    if len(mountains) > 2:
                        shots_to_split.append(shot)

            if len(shots_to_split) == 0:
                new_points.append(point)
            else:
                for splitted in Point.split_point_list(point, shots_to_split):
                    new_points.append(splitted)

        return new_points

    def clean_tracks(self):
        # Conditions
        #   - No balls moments
        #   - No short moments
        #   - No static ball moments
        #   - No slow ball moments (except start of point)
        #   - No mountains on bottom moments
        #   - No jump moments
        # ORDER MATTERS!

        start = 45000
        end = 46500

        vis = Visuals()
        #vis.plot_tracks(self.tracks, self.frames_controller.frame_list, frame_start=start, frame_end=end, fps=60, net=self.net)

        #self.no_balls_moments()
        #self.clean_short_tracks()
        #vis.plot_tracks(self.tracks, self.frames_controller.frame_list, frame_start=start, frame_end=end, fps=60, net=self.net)


        self.static_balls_moments(min_frames=60, min_diff_allowed = 0.0005)
        #vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, frame_start=start, frame_end=end, net=self.net)


        #self.bottom_mountains_moments(max_mountains=self.hyperparameters['max_bottom_mountains'])
        #self.top_mountains_moments(max_mountains=self.hyperparameters['max_top_mountains'], max_height=self.hyperparameters['max_height_top_mountains'])
        #vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, frame_start=start, frame_end=end, net=self.net)

        #self.jumps_moments(min_frames=self.hyperparameters['jumps_min_frames'], max_jump_allowed = self.hyperparameters['jumps_max_allowed'], num_jumps = self.hyperparameters['jumps_max_num'])
        #vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, frame_start=start, frame_end=end, net=self.net)

        #self.remerge_tracks(margin=5)

        #self.slow_ball_moments(min_frames=self.hyperparameters['slow_balls_min_frames'], min_velocity=self.hyperparameters['slow_balls_min_velocity'])
        #vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, frame_start=start, frame_end=end, net=self.net)

        #self.discontinuous_moments(min_frames=self.hyperparameters['disc_min_frames'], frames_disc=self.hyperparameters['disc_frames_disc'], max_jump_allowed = 0.15, min_occurrences=self.hyperparameters['disc_min_occurrences'])
        #vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, frame_start=start, frame_end=end, net=self.net)

    def jumps_moments(self, min_frames, max_jump_allowed, num_jumps):
        jumps = {}
        for track in self.tracks:
            if len(track) > 1:

                jump_pifs = []
                for i in range(1, len(track.pifs)):
                    if abs(track.pifs[i].y - track.pifs[i-1].y) > max_jump_allowed or abs(track.pifs[i].x - track.pifs[i-1].x) > max_jump_allowed:
                        jump_pifs.append(track.pifs[i])

                consecutives = self.get_flexible_segments(jump_pifs, min_frames)

                jumps[track] = []
                for c in consecutives:
                    if len(c) >= num_jumps:
                        jumps[track].append((c[0].frame_number, c[-1].frame_number))


        subtracks = []
        for track in self.tracks:
            if track in jumps.keys():
                subtracks += Track.split_track(track, jumps[track])
            else:
                subtracks.append(track)

        self.tracks = subtracks

    def discontinuous_moments(self, min_frames, frames_disc, max_jump_allowed, min_occurrences):
        disc = {}
        for track in self.tracks:
            if len(track) > 1:

                disc_pifs = []
                for i in range(1, len(track.pifs)):
                    if track.pifs[i].frame_number - track.pifs[i - 1].frame_number > frames_disc:
                        if track.pifs[i].y < 0.8 and track.pifs[i - 1].y < 0.8:
                            disc_pifs.append(track.pifs[i])
                    #elif abs(track.pifs[i].y - track.pifs[i-1].y) > max_jump_allowed:
                    #    disc_pifs.append(track.pifs[i])
                    #elif abs(track.pifs[i].x - track.pifs[i-1].x) > max_jump_allowed:
                    #    disc_pifs.append(track.pifs[i])



                consecutives = self.get_flexible_segments(disc_pifs, min_frames)

                disc[track] = []
                for c in consecutives:
                    if len(c) >= min_occurrences:
                        disc[track].append((c[0].frame_number, c[-1].frame_number))

        subtracks = []
        for track in self.tracks:
            if track in disc.keys() and disc[track]:
                subtracks += Track.split_track(track, disc[track])
            else:
                subtracks.append(track)

        self.tracks = subtracks

    def slow_ball_moments(self, min_frames, min_velocity):
        slow = {}
        for track in self.tracks:
            if len(track) > 1:
                slow_idx = []

                start_slow_fn = track.pifs[0].frame_number
                end_slow_fn = 0
                counter = 0

                for i in range(1, len(track.pifs)):
                    vel = PositionInFrame.velocity(track.pifs[i], track.pifs[i-1], scale=1000)

                    if vel < min_velocity:
                        end_slow_fn = track.pifs[i].frame_number
                        counter += 1
                    else:
                        if counter > min_frames:
                            slow_idx.append((start_slow_fn, end_slow_fn))
                        start_slow_fn = track.pifs[i].frame_number
                        counter = 0
                slow[track] = slow_idx

        subtracks = []
        for track, slow_moments in slow.items():
            subtracks += Track.split_track(track, slow_moments)

        self.tracks = subtracks

    def static_balls_moments(self, min_frames=60, min_diff_allowed = 0.0005):
        static = {}
        for track in self.tracks:
            if len(track) > min_frames:
                static_idx = []

                start_static_fn = track.pifs[0].frame_number
                end_static_fn = 0
                counter = 0

                for i in range(1, len(track.pifs)):
                    if abs(track.pifs[i].y - track.pifs[i-1].y) < min_diff_allowed:
                        end_static_fn = track.pifs[i].frame_number
                        counter += 1
                    else:
                        if counter > min_frames:
                            static_idx.append((start_static_fn, end_static_fn))
                        start_static_fn = track.pifs[i].frame_number
                        counter = 0
                static[track] = static_idx




        subtracks = []
        for track, static_moments in static.items():
            subtracks += Track.split_track(track, static_moments)


        self.tracks = subtracks


    def get_flexible_segments(self, numbers, limit=300):
        segments = []
        current_segment = []

        for i in range(len(numbers)):
            if not current_segment:
                current_segment.append(numbers[i])
            elif abs(numbers[i].frame_number - current_segment[-1].frame_number) < limit:
                current_segment.append(numbers[i])
            else:
                segments.append(current_segment)
                current_segment = [numbers[i]]

        if current_segment:
            segments.append(current_segment)

        return segments



    def low_variance_moments(self, window=120, min_variance_allowed = 0.001):
        low_var = {}
        for track in self.tracks:
            if len(track) > window:
                low_variance = []
                for i in range(len(track)-window):
                    if PositionInFrame.calculate_variance(track.pifs[i:i+window]) < min_variance_allowed:
                        low_variance.append((i, i+window))
                low_variance = self.merge_moments(low_variance)
                low_var[track] = low_variance



        subtracks = []
        for track, low_var_moments in low_var.items():
            subtracks += Track.split_track(track, low_var_moments)

        self.tracks = subtracks


    def top_mountains_moments(self, max_mountains, max_height):
        mountains = {}
        for track in self.tracks:
            if len(track) > 1:
                segments = []
                current_segment = []
                mountains_list = []

                for pif in track.pifs:
                    if pif.y < self.net.y - self.net.height/2:
                        current_segment.append(pif)
                    else:
                        if current_segment:  # only add non-empty segments
                            segments.append(current_segment)
                            current_segment = []

                # Add the last segment if it has any elements
                if current_segment:
                    segments.append(current_segment)

                for possible_mountain in segments:
                    if len(Point.find_mountains(possible_mountain, smooth=True, max_height=max_height)) > max_mountains:
                        mountains_list.append((possible_mountain[0].frame_number, possible_mountain[-1].frame_number))

                if len(mountains_list) > 0:
                    mountains[track] = mountains_list

        subtracks = []
        for track in self.tracks:
            if track in mountains:
                subtracks += Track.split_track(track, mountains[track])
            else:
                subtracks.append(track)

        self.tracks = subtracks

    def bottom_mountains_moments(self, max_mountains):
        mountains = {}
        for track in self.tracks:
            if len(track) > 1:
                segments = []
                current_segment = []
                mountains_list = []

                for pif in track.pifs:
                    if pif.y > self.net.y - self.net.height/2:
                        current_segment.append(pif)
                    else:
                        if current_segment:  # only add non-empty segments
                            segments.append(current_segment)
                            current_segment = []

                # Add the last segment if it has any elements
                if current_segment:
                    segments.append(current_segment)

                for possible_mountain in segments:
                    if len(Point.find_mountains(possible_mountain, smooth=True)) > max_mountains:
                        mountains_list.append((possible_mountain[0].frame_number, possible_mountain[-1].frame_number))

                if len(mountains_list) > 0:
                    mountains[track] = mountains_list

        subtracks = []
        for track in self.tracks:
            if track in mountains:
                subtracks += Track.split_track(track, mountains[track])
            else:
                subtracks.append(track)

        self.tracks = subtracks



    def low_uniformity_moments(self, min_frames=60, min_diff_allowed = 0.0005):

        low_unif = {}
        for track in self.tracks:
            if len(track) > min_frames:
                time_pos = []
                for pif in track.pifs:
                    time_pos.append(pif.y)
                unif = self.find_non_uniform_segments(time_pos, threshold=0.1)
                print(unif)

    def find_non_uniform_segments(self, time_positions, min_length=30, threshold=0.1):
        if len(time_positions) < 2:
            return "Not enough data to determine uniformity"

        non_uniform_segments = []
        current_segment = []

        differences = np.diff(time_positions)
        mean_diff = np.mean(differences)

        start_index = 0

        for i in range(1, len(time_positions)):
            current_segment.append(time_positions[i - 1])

            # Calculate the coefficient of variation (CV) for the current segment
            segment_diffs = np.diff(current_segment)
            if len(segment_diffs) > 1:
                cv = np.std(segment_diffs) / mean_diff

                if cv > threshold:
                    # Segment is non-uniform
                    if len(current_segment) >= min_length:
                        non_uniform_segments.append(current_segment[:])
                    current_segment = []  # Reset the current segment

        # Check the last segment
        current_segment.append(time_positions[-1])
        if len(current_segment) >= min_length:
            segment_diffs = np.diff(current_segment)
            cv = np.std(segment_diffs) / mean_diff
            if cv > threshold:
                non_uniform_segments.append(current_segment)

        return non_uniform_segments




    def clean_short_tracks(self, minimum=20):
        self.tracks = [track for track in self.tracks if len(track) >= minimum]
    def no_balls_moments(self, window=120):
        tracks = sorted(self.tracks.copy(), key=lambda obj: obj.start())
        no_balls = {}
        for track in tracks:
            if len(track) > 1:
                no_balls_idx = []
                last_fn = track.pifs[0].frame_number
                for pif in track.pifs:
                    if pif.frame_number - last_fn > window:
                        no_balls_idx.append((last_fn, pif.frame_number))
                    last_fn = pif.frame_number
                no_balls[track] = no_balls_idx

        subtracks = []
        for track, no_balls_moments in no_balls.items():
            subtracks += Track.split_track(track, no_balls_moments)

        self.tracks = subtracks








    def merge_moments(self, moments):
        merged_moments = []

        for interval in moments:
            # If the merged_intervals list is empty or the current interval does not overlap with the last one, append it
            if not merged_moments or merged_moments[-1][1] < interval[0]:
                merged_moments.append(interval)
            else:
                # If there is an overlap, merge the current interval with the last one in the merged_intervals list
                merged_moments[-1] = (merged_moments[-1][0], max(merged_moments[-1][1], interval[1]))
        return merged_moments

    def merge_tracks(self, margin=30):
            merged = []
            track_list = sorted(self.tracks.copy(), key=lambda obj: obj.start())
            pool = track_list.copy()

            while len(pool) > 0:
                to_merge = [pool[0]]
                for track in pool[1:]:
                    if track.start() >= to_merge[-1].end():
                        to_merge.append(track)
                merged_track = Track.join_tracks(to_merge, to_merge[0].tag)
                merged.append(merged_track)
                for track in to_merge:
                    pool.remove(track)
            self.tracks = merged









    def load_tracks(self, delete_statics = True, min_duration = 180):
        tracks = {}
        for i, frame in self.frames_controller.enumerate():
            for ball in frame.balls():
                if ball.tag not in tracks.keys():
                    tracks[ball.tag] = []
                tracks[ball.tag].append(PositionInFrame(ball.x, ball.y, frame.frame_number))

        tracks_loaded = []
        for tag, track in tracks.items():
            if tag is not None:
                new_track = Track(tag)
                for pif in track:
                    new_track.add_pif(pif)

                static = new_track.check_static_with_variance(minimum=60)
                if not static or static and not delete_statics:
                    tracks_loaded.append(new_track)
            else:
                for none_track in track:
                    new_track = Track(None)
                    new_track.add_pif(none_track)
                    new_track.static = False
                    tracks_loaded.append(new_track)

        return tracks_loaded


