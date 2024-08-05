import matplotlib
import numpy as np

from padelClipsPackage.Track import *
from padelClipsPackage.Visuals import Visuals


class PositionTrackerV2:
    def __init__(self, frames_controller, fps, net):

        self.fps = fps
        self.frames_controller = frames_controller
        self.net = net
        # Load phase: Equivalence track-tag
        self.tracks = self.load_tracks(delete_statics=True)
        # Merge phase: Join consecutive tracks
        self.tracks = self.merge_tracks()
        # Split phase: Detect split conditions, like empty seconds, multiple non-static balls...
        self.tracks = self.split_tracks()

    def split_tracks(self):
        # Conditions
        #   - No balls windows
        #   - Static ball windows
        #   - Multiple balls windows
        #   - Multiple inflexion points in same net position windows (Not for now)
        # ORDER MATTERS!

        vis = Visuals()
        vis.plot_tracks(self.tracks, self.frames_controller.frame_list, frame_end=3000, fps=60, net=self.net)

        self.no_balls_moments()
        self.clean_short_tracks()
        vis = Visuals()
        vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, net=self.net)


        vis = Visuals()
        vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, net=self.net)

        # Low variance moments, like start/end of points
        self.static_balls_moments(min_frames=120, min_diff_allowed = 0.025)
        vis = Visuals()
        vis.plot_tracks(self.tracks, self.frames_controller.frame_list, fps=60, net=self.net)

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


    def static_balls_moments(self, min_frames=60, min_diff_allowed = 0.0005):
        static = {}
        for track in self.tracks:
            if len(track) > 1:
                static_idx = []
                last_y = track.pifs[0].y
                start_static_fn = track.pifs[0].frame_number
                end_static_fn = 0
                counter = 0

                for pif in track.pifs[1:]:
                    if abs(pif.y - last_y) < min_diff_allowed:
                        last_y = pif.y
                        end_static_fn = pif.frame_number
                        counter += 1
                    else:
                        last_y = pif.y
                        if counter > min_frames:
                            static_idx.append((start_static_fn, end_static_fn))
                        start_static_fn = pif.frame_number
                        counter = 0
                static[track] = static_idx


        subtracks = []
        for track, static_moments in static.items():
            subtracks += Track.split_track(track, static_moments)

        self.tracks = subtracks



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

    def merge_tracks(self):
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
            return merged









    def load_tracks(self, delete_statics = True):
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




class PositionTracker:
    def __init__(self, frames_controller, fps, net):
        self.closed_tracks = []
        self.open_tracks = []
        self.fps = fps
        self.frames_controller = frames_controller
        self.net = net

        self.create_tracks()
        self.tag_frames()

        vis = Visuals()
        vis.plot_tracks(self.closed_tracks, self.frames_controller.frame_list, frame_end=7200, fps=60, net=self.net)

        self.playtime = self.generate_playtime()



        # reduced = False
        reduced = True
        while reduced:
            reduced = self.reduce_noise_v2()
            if reduced:
                self.playtime = self.merge_playtime()

        self.update_tags()

        self.detect_breaks_v2(100)
        # self.detect_breaks_under_net()
        # self.merge_tracks_too_close()

        self.playtime = self.generate_playtime()

        self.points = self.generate_points()

        Visuals.plot_points(self.closed_tracks, self.points, self.net, self.fps, end=7200)

        self.plot_tracks_with_net(self.closed_tracks, fps=self.fps)
        self.plot_tracks_with_net_and_players(self.closed_tracks)

    def get_tracks_between_frames(self, start, end, include_partial=True):
        result = []
        for track in self.closed_tracks:
            if len(track.pifs) > 0:
                if track.start() > start and track.end() < end:
                    result.append(track)
                elif include_partial and (track.has_frame(start) or track.has_frame(end)):
                    result.append(track)

        return result

    def update_tags(self):
        for (start, end), tag in self.playtime.items():
            for i in range(start, end + 1):
                self.frames_controller.get(i).tag = tag

    def detect_breaks_v2(self, window, max_min=0.15):

        points = []
        for frames, value in self.playtime.items():
            if value == 'point':
                points.append(frames)

        for point in points:
            tracks = self.get_tracks_between_frames(point[0], point[1], include_partial=True)
            tracks = [track for track in tracks if len(track.pifs) > 0]
            megatrack = []
            for track in tracks:
                pifs = track.pifs.copy()
                for pif in track.pifs:
                    if pif.frame_number < point[0] or pif.frame_number > point[1]:
                        pifs.remove(pif)

                megatrack += pifs

            buffer = []
            if len(megatrack) > 0:
                initial_pos = self.position_over_the_net(megatrack[0].y, self.net)
                for pif in megatrack:
                    pif_pos = self.position_over_the_net(pif.y, self.net)
                    if pif_pos == initial_pos:
                        buffer.append(pif)
                    else:
                        if len(buffer) > 0 and buffer[-1].frame_number - buffer[0].frame_number >= window:
                            bmax, bmin = PositionInFrame.calculate_max_min(buffer)
                            if bmax - bmin < max_min or initial_pos == 'under':
                                for i in range(buffer[0].frame_number, buffer[-1].frame_number + 1):
                                    self.frames_controller.get(i).tag = 'break'
                        buffer = []
                        initial_pos = self.position_over_the_net(pif.y, self.net)
                if len(buffer) > 0 and buffer[-1].frame_number - buffer[0].frame_number >= window:
                    bmax, bmin = PositionInFrame.calculate_max_min(buffer)
                    if bmax - bmin < max_min or initial_pos == 'under':
                        for i in range(buffer[0].frame_number, buffer[-1].frame_number + 1):
                            self.frames_controller.get(i).tag = 'break'

    def position_over_the_net(self, y, net):
        if y < net.y - net.height / 2:
            return 'over'
        elif y > net.y + net.height / 2:
            return 'under'
        else:
            return 'middle'

    def merge_playtime(self):
        merged_frames = {}
        last_range, last_tag = None, None

        for (start, end), tag in sorted(self.playtime.items()):
            if last_tag == tag and last_range[1] + 1 == start:
                # Extend the last range
                last_range = (last_range[0], end)
            else:
                if last_range:
                    merged_frames[last_range] = last_tag
                last_range, last_tag = (start, end), tag

        # Add the last range to the dictionary
        if last_range:
            merged_frames[last_range] = last_tag

        return merged_frames

    def generate_points(self):
        playtime = [point for point in self.playtime.keys() if self.playtime[point] == 'point']
        point_tracks = []
        for point in playtime:
            start = point[0]
            end = point[1]

            track = Track()
            for frame_n in range(start, end + 1):
                for old_track in self.closed_tracks:
                    for pif in old_track.pifs:
                        if pif.frame_number == frame_n:
                            track.add_pif(pif)
            point_tracks.append(track)
        return point_tracks

    def get_tracks_by_frame(self, frame_number):
        tracks = []
        for track in self.closed_tracks:
            if track.start() <= frame_number <= track.end():
                tracks.append(track)
        return tracks

    def generate_playtime(self):
        playtime = self.timeline_to_sections()
        return playtime

    def reduce_noise_v2(self, window=80, mess_window=25):
        reduced = False
        keys = list(self.playtime.keys())
        for i in range(1, len(self.playtime.keys()) - 1):
            if (self.playtime[keys[i]] != 'mess' or keys[i][1] - keys[i][0] < mess_window) and self.playtime[
                keys[i - 1]] == self.playtime[keys[i + 1]] and \
                    (self.playtime[keys[i]] != self.playtime[keys[i - 1]] and keys[i][1] - keys[i][0] < window) and \
                    (keys[i - 1][1] - keys[i - 1][0] > keys[i][1] - keys[i][0] or
                     keys[i + 1][1] - keys[i + 1][0] > keys[i][1] - keys[i][0]):
                self.playtime[keys[i]] = self.playtime[keys[i - 1]]
                reduced = True
        return reduced

    def timeline_to_sections(self):
        sections = {}
        start = 0
        current = None
        for frame_number, frame in self.frames_controller.enumerate():
            try:
                if current == None:
                    current = frame.tag

                if frame.tag != current:
                    end = frame_number - 1
                    sections[(start, end)] = current
                    start = frame_number
                    current = frame.tag
            except:
                print(frame)
        return sections

    def tag_frames(self):
        playtime = {}
        tracks_start_to_end = []
        for track in self.closed_tracks:
            first = track.start()
            end = track.end()
            if first is not None and end is not None:
                tracks_start_to_end.append((first, end))

        for start, end in tracks_start_to_end:
            for frame in range(start, end + 1):

                if frame in playtime:
                    playtime[frame] += 1
                else:
                    playtime[frame] = 1

            # Generate the final dictionary with True or False based on the count

        # Determine the full range of frames

        for i, frame in self.frames_controller.enumerate():
            # True if exactly one track covers the frame, False otherwise
            if i in playtime.keys():
                count = playtime.get(i, 0)
                if frame != None:
                    if count == 1:
                        frame.tag = 'point'
                    elif count == 0:
                        frame.tag = 'empty'
                    elif count > 1:
                        frame.tag = 'mess'
            elif frame != None:
                frame.tag = 'no_frames'

    def create_tracks(self):
        for i, frame in self.frames_controller.enumerate():
            if i % 100 == 0:
                print("Tracking balls from frame " + str(i) + "/" + str(len(self.frames_controller)), end='\r')
            self.track_ball(frame)
            self.close_tracks(frame.frame_number)


        clean = self.clean_tracks(self.closed_tracks)
        self.closed_tracks = clean



    def clean_tracks(self, tracks):
        clean = tracks.copy()
        self.remove_short_tracks(clean)
        self.remove_static_balls(clean, minimum_length=3)
        self.remove_short_tracks(clean)

        for track in clean.copy():
            if len(track.pifs) == 0:
                clean.remove(track)

        return clean

    def remove_static_balls(self, tracks, minimum_length=3, minimum_to_split=100):
        for track in tracks:
            track.pifs = self.filter_consecutive_objects_with_min_count(track.pifs, min_count=minimum_length)

        tracks_copy = tracks.copy()
        index = []
        for track in tracks_copy:
            for i, pif in enumerate(track.pifs):
                if i > 0 and pif.frame_number - track.pifs[i - 1].frame_number >= minimum_to_split:
                    index.append(i)

            index = sorted(set(index))
            # Add the start and end boundaries for slicing
            track_splitted = [track.pifs[start:end] for start, end in zip([0] + index, index + [len(track.pifs)])]
            if len(track_splitted) > 1:
                tracks.remove(track)
                for newtrack in track_splitted:
                    ntr = Track()
                    for pif in newtrack:
                        ntr.add_pif(pif)
                    tracks.append(ntr)

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

    def remove_short_tracks(self, tracks, minimum_length=3):
        to_remove = [track for track in tracks if
                     len(track.pifs) <= minimum_length and abs(track.max_min()[1] - track.max_min()[0]) < 0.1]
        for track in to_remove:
            tracks.remove(track)

    def close_tracks(self, frame_number, tolerance=5):
        to_close = [track for track in self.open_tracks if track.end() + tolerance < frame_number]
        for track in to_close:
            #track.check_static()
            self.closed_tracks.append(track)
            self.open_tracks.remove(track)

    def track_ball(self, frame, max_distance=0.3):

        balls = frame.balls()
        balls_to_track = balls.copy()

        distances = {}
        for ball in balls:
            # Añadir a tracks ya abiertos
            for track in self.open_tracks:
                distance = PositionInFrame.distance_to(PositionInFrame(ball.x, ball.y, None),
                                                       track.last_pif())
                # Distancia maxima para añadir candidato
                if max_distance > distance:
                    distances[(ball, track)] = distance

        # Se calcula la distancia de los candidatos, y se elige el mas cercano
        while len(balls_to_track) > 0 and len(distances.keys()) > 0:
            s_ball, s_track = self.get_shortest(distances)
            s_track.add_pif(PositionInFrame(s_ball.x, s_ball.y, frame.frame_number))
            balls_to_track.remove(s_ball)

        # NUEVOS Tracks
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

    def plot_tracks_with_net_and_players(self, tracks, frame_start=0, frame_end=float('inf')):
        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

        # Define a helper function to convert frame number to seconds
        def frame_to_seconds(frame_number):
            return (frame_number - frame_start) / self.fps

        # Format function for times on x-axis
        def format_seconds(seconds):
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        # Plot Y positions of ball and players on ax1
        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.pifs if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.pifs if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn) for fn in frame_numbers]
                ax1.plot(seconds, y_positions, marker='o',
                         label=f'Track from frame {track.pifs[0].frame_number}', color='green')

        player_a, fn_a = self.get_player_position_over_time('A', 'y', frame_start, frame_end)
        player_b, fn_b = self.get_player_position_over_time('B', 'y', frame_start, frame_end)
        ax1.plot([frame_to_seconds(fn) for fn in fn_a], player_a, marker='x', label='Player A', color='red')
        ax1.plot([frame_to_seconds(fn) for fn in fn_b], player_b, marker='x', label='Player B', color='yellow')
        player_c, fn_c = self.get_player_position_over_time('C', 'y', frame_start, frame_end)
        player_d, fn_d = self.get_player_position_over_time('D', 'y', frame_start, frame_end)
        ax1.plot([frame_to_seconds(fn) for fn in fn_c], player_c, marker='x', label='Player A', color='blue')
        ax1.plot([frame_to_seconds(fn) for fn in fn_d], player_d, marker='x', label='Player B', color='black')

        ax1.axhline(y=(1 - self.net.y) + self.net.height / 2, color='blue', label='Net (Sup)')
        ax1.axhline(y=(1 - self.net.y) - self.net.height / 2, color='blue', label='Net (Inf)')

        ax1.set_xlabel('Time (hh:mm:ss)')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Ball and Player Y Positions Over Time')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))
        ax1.grid(True)

        # Plot X positions of ball and players on ax2
        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.pifs if frame_start <= pif.frame_number <= frame_end]
            x_positions = [pif.x for pif in track.pifs if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn) for fn in frame_numbers]
                ax2.plot(seconds, x_positions, marker='o',
                         label=f'Track from frame {track.pifs[0].frame_number}', color='green')

        player_a, fn_a = self.get_player_position_over_time('A', 'x', frame_start, frame_end)
        player_b, fn_b = self.get_player_position_over_time('B', 'x', frame_start, frame_end)
        ax2.plot([frame_to_seconds(fn) for fn in fn_a], player_a, marker='x', label='Player A', color='red')
        ax2.plot([frame_to_seconds(fn) for fn in fn_b], player_b, marker='x', label='Player B', color='yellow')
        player_c, fn_c = self.get_player_position_over_time('C', 'x', frame_start, frame_end)
        player_d, fn_d = self.get_player_position_over_time('D', 'x', frame_start, frame_end)
        ax2.plot([frame_to_seconds(fn) for fn in fn_c], player_c, marker='x', label='Player C', color='blue')
        ax2.plot([frame_to_seconds(fn) for fn in fn_d], player_d, marker='x', label='Player D', color='black')

        ax2.set_xlabel('Time (hh:mm:ss)')
        ax2.set_ylabel('X Position')
        ax2.set_title('Ball and Player X Positions Over Time')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))
        ax2.grid(True)

        #plt.legend()

        plt.tight_layout()
        plt.show(block=True)

    def plot_tracks_with_net(self, tracks, frame_start=-1, frame_end=float('inf'), fps=30):
        if frame_end == float('inf'):
            frame_end = self.frames_controller.get(-1).frame_number

        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)  # Reintroduce ax3 for the second plot

        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.pifs if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.pifs if frame_start <= pif.frame_number <= frame_end]

            if frame_numbers:
                ax1.plot(frame_numbers, y_positions, marker='o',
                         label=f'Track starting at frame {track.pifs[0].frame_number}', color='green')

        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Y Position of Ball')
        ax1.set_title('Track of Ball Y Positions Over Time')
        ax1.set_xlim(left=frame_start, right=frame_end)
        ax1.grid(True)

        ax1.axhline(y=(1 - self.net.y) + self.net.height / 2, color='blue', label='Net (Sup)')
        ax1.axhline(y=(1 - self.net.y) - self.net.height / 2, color='blue', label='Net (Inf)')

        # Function to format seconds into HH:MM:SS
        def format_time(seconds):
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        # Create secondary x-axis for frame numbers
        ax2 = ax1.twiny()
        ax2.set_xlabel('Time in HH:MM:SS')
        frame_ticks = np.linspace(frame_start, frame_end, num=10)  # Create 10 ticks from start to end frame
        time_ticks = frame_ticks / fps
        time_labels = [format_time(t) for t in time_ticks]
        ax2.set_xlim(left=frame_start / fps, right=frame_end / fps)
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)
        ax2.grid(True)

        # Plot ax3 with whatever data is relevant (e.g., track coverage or some other data)
        # Example: Populate ax3 similarly, assuming 'colors' array or function is defined
        if hasattr(self, 'frames'):
            colors = [
                'green' if frame.tag == 'point' else 'red' if frame.tag == 'mess' else 'yellow' if frame.tag == 'break' else 'blue'
                for frame in self.frames]
            ax3.vlines(range(len(self.frames)), 0, 1, colors=colors)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Coverage')
        ax3.set_title('Track Coverage')
        ax3.grid(True)

        plt.tight_layout()
        plt.show(block=True)
