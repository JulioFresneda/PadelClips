from datetime import timedelta



from padelLynxPackage.Track import *
import matplotlib
from padelLynxPackage.Frame import *
from collections import defaultdict
from padelLynxPackage import aux
import numpy as np

class PositionTracker:
    def __init__(self, frames, fps, net):
        self.closed_tracks = []
        self.open_tracks = []
        self.fps = fps
        self.frames = frames

        self.net = net

        self.manage_tracks()

        #self.smooth_tracks()

        self.tag_frames()


        self.playtime = self.generate_playtime()

        #reduced = False
        reduced = True
        while reduced:
            reduced = self.reduce_noise_v2()
            if reduced:
                self.playtime = self.merge_playtime()


        self.update_tags()

        self.detect_breaks_v2(100)
        #self.detect_breaks_under_net()
        self.playtime = self.generate_playtime()

        self.points = self.generate_points()



        # self.plot_tracks(self.closed_tracks, frame_start=41550, frame_end=42600, print_globes=False)
        self.plot_tracks_with_net(self.points, frame_start=00)

        self.plot_tracks_with_net_and_players(self.points, frame_start=0)

    def smooth_tracks(self):

        for track in self.closed_tracks:
            if len(track.track) > 0:
                smooth = []
                for pif in track.track:
                    smooth.append((pif.x, pif.y))
                smooth = aux.apply_kalman_filter(smooth)
                for s, pif in zip(smooth, track.track):
                    pif.x = s[0]
                    pif.y = s[1]


    def get_tracks_between_frames(self, start, end, include_partial=True):
        result = []
        for track in self.closed_tracks:
            if len(track.track) > 0:
                if track.first_frame() > start and track.last_frame() < end:
                    result.append(track)
                elif include_partial and (track.has_frame(start) or track.has_frame(end)):
                    result.append(track)

        return result

    def update_tags(self):
        for (start, end), tag in self.playtime.items():
            for i in range(start, end + 1):
                self.frames[i].tag = tag



    def detect_breaks(self, window, max_min=0.15):

        points = []
        for frames, value in self.playtime.items():
            if value == 'point':
                points.append(frames)

        for point in points:
            tracks = self.get_tracks_between_frames(point[0], point[1], include_partial=True)
            tracks = [track for track in tracks if len(track.track) > 0]
            for track in tracks:
                pifs = track.track.copy()
                for pif in track.track:
                    if pif.frame_number < point[0] or pif.frame_number > point[1]:
                        pifs.remove(pif)
                buffer = []
                initial_pos = self.position_over_the_net(pifs[0].y, self.net)
                for pif in pifs:
                    pif_pos = self.position_over_the_net(pif.y, self.net)
                    if pif_pos == initial_pos:
                        buffer.append(pif)
                    else:
                        if len(buffer) > 0 and buffer[-1].frame_number - buffer[0].frame_number >= window:
                            bmax, bmin = PositionInFrame.calculate_max_min(buffer)
                            if bmax - bmin < max_min:
                                for i in range(buffer[0].frame_number, buffer[-1].frame_number+1):
                                    self.frames[i].tag = 'break'
                        buffer = []
                if len(buffer) > 0 and buffer[-1].frame_number - buffer[0].frame_number >= window:
                    bmax, bmin = PositionInFrame.calculate_max_min(buffer)
                    if bmax - bmin < max_min:
                        for i in range(buffer[0].frame_number, buffer[-1].frame_number+1):
                            self.frames[i].tag = 'break'


    def detect_breaks_under_net(self, min_max = 0.15):
        points = []
        for frames, value in self.playtime.items():
            if value == 'point':
                points.append(frames)

        for point in points:
            tracks = self.get_tracks_between_frames(point[0], point[1], include_partial=True)
            tracks = [track for track in tracks if len(track.track) > 0 and
                      track.position_in_net(self.net) == 'under' and track.get_direction_changes() >= 2
                      and PositionInFrame.calculate_max_min(track.track) > min_max]

            for track in tracks:
                pifs = track.track.copy()
                for pif in track.track:
                    if pif.frame_number < point[0] or pif.frame_number > point[1]:
                        pifs.remove(pif)

                for pif in pifs:
                    self.frames[pif.frame_number].tag = 'break'












    def detect_breaks_v2(self, window, max_min=0.15):

        points = []
        for frames, value in self.playtime.items():
            if value == 'point':
                points.append(frames)

        for point in points:
            tracks = self.get_tracks_between_frames(point[0], point[1], include_partial=True)
            tracks = [track for track in tracks if len(track.track) > 0]
            megatrack = []
            for track in tracks:
                pifs = track.track.copy()
                for pif in track.track:
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
                                for i in range(buffer[0].frame_number, buffer[-1].frame_number+1):
                                    self.frames[i].tag = 'break'
                        buffer = []
                        initial_pos = self.position_over_the_net(pif.y, self.net)
                if len(buffer) > 0 and buffer[-1].frame_number - buffer[0].frame_number >= window:
                    bmax, bmin = PositionInFrame.calculate_max_min(buffer)
                    if bmax - bmin < max_min or initial_pos == 'under':
                        for i in range(buffer[0].frame_number, buffer[-1].frame_number+1):
                            self.frames[i].tag = 'break'


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
            for frame_n in range(start, end+1):
                for old_track in self.closed_tracks:
                    for pif in old_track.track:
                        if pif.frame_number == frame_n:
                            track.add_pif(pif)
            point_tracks.append(track)
        return point_tracks


    def clean_playtime(self):
        keys = list(self.playtime.keys())
        for i in range(1, len(keys) - 1):
            if self.playtime[keys[i]] == 'no_frames':
                if self.playtime[keys[i - 1]] == 'point' and self.playtime[keys[i + 1]] == 'point':
                    self.playtime[keys[i]] = 'point'
                    for f in self.frames[keys[i][0]:keys[i][1]]:
                        if f != None:
                            f.tag = 'point'

    def get_tracks_by_frame(self, frame_number):
        tracks = []
        for track in self.closed_tracks:
            if track.first_frame() <= frame_number <= track.last_frame():
                tracks.append(track)
        return tracks

    def get_frame(self, frame_number):
        lowest = self.frames[0].frame_number
        if lowest <= frame_number < lowest + len(self.frames):
            return self.frames[frame_number - lowest]

    def generate_playtime(self):
        playtime = self.timeline_to_sections()
        return playtime

    def reduce_noise_v2(self, window=80, mess_window=25):
        reduced = False
        keys = list(self.playtime.keys())
        for i in range(1, len(self.playtime.keys()) - 1):
            if (self.playtime[keys[i]] != 'mess' or keys[i][1] - keys[i][0] < mess_window) and self.playtime[keys[i - 1]] == self.playtime[keys[i + 1]] and \
                    (self.playtime[keys[i]] != self.playtime[keys[i - 1]] and keys[i][1] - keys[i][0] < window) and \
                    (keys[i - 1][1] - keys[i - 1][0] > keys[i][1] - keys[i][0] or
                     keys[i + 1][1] - keys[i + 1][0] > keys[i][1] - keys[i][0]):
                self.playtime[keys[i]] = self.playtime[keys[i - 1]]
                reduced = True
        return reduced

    def reduce_noise(self, playtime, limit=30, min_point_between_empties=3):
        playtime_clean = playtime.copy()
        sections = self.timeline_to_sections(playtime)
        keys = list(sections.keys())

        for i, key in enumerate(keys[1:-1], start=1):
            before = sections[keys[i - 1]]
            before_length = abs(keys[i - 1][1] - keys[i - 1][0])
            after = sections[keys[i + 1]]
            after_length = abs(keys[i + 1][1] - keys[i + 1][0])
            current = sections[key]
            current_length = abs(key[0] - key[1])

            # if before == 'empty' and current == 'point' and current_length >= min_point_between_empties:
            #    for i in range(keys[i-1][0], keys[i-1][1] + 1):
            #        playtime_clean[i] = current
            if before == after and before != current and current_length <= limit and before_length > current_length and after_length > current_length:
                quality = False
                if current == 'point' and before == 'empty':
                    tracks_affected = self.get_tracks_by_frame(key[0])
                    if len(tracks_affected) == 1:
                        if tracks_affected[0].quality_track:
                            quality = True
                            for i in range(keys[i - 1][0], keys[i - 1][1] + 1):
                                playtime_clean[i] = current
                #    pass
                if not quality:
                    for i in range(key[0], key[1] + 1):
                        playtime_clean[i] = before

        return playtime_clean

    def timeline_to_sections(self):
        sections = {}
        start = 0
        current = None
        for frame_number, frame in enumerate(self.frames):
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
            first = track.first_frame()
            end = track.last_frame()
            if first != None and end != None:
                tracks_start_to_end.append((first, end))

        for start, end in tracks_start_to_end:
            for frame in range(start, end + 1):

                if frame in playtime:
                    playtime[frame] += 1
                else:
                    playtime[frame] = 1

            # Generate the final dictionary with True or False based on the count

        # Determine the full range of frames

        for i, frame in enumerate(self.frames):
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

    def manage_tracks(self):
        for i, frame in enumerate(self.frames):
            if i % 100 == 0:
                print("Tracking balls from frame " + str(i) + "/" + str(len(self.frames)), end='\r')
            self.track_ball(frame)
            self.close_tracks(frame.frame_number)

        clean = self.clean_tracks(self.closed_tracks)
        self.closed_tracks = clean



        #tracks_with_globe = self.detect_globes(self.closed_tracks)
        #self.closed_tracks = tracks_with_globe
        #self.globes = [track for track in self.closed_tracks if track.globe]

        # self.plot_tracks([track for track in self.closed_tracks if track.static is False])

    def detect_globes(self, tracks, max_distance=100):
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
                if s_frame > closest_frame and s_frame < e_frame and abs(s_frame - e_frame) < max_distance:
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
            globe.check_quality()

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
        self.remove_short_tracks(clean)
        self.remove_static_balls(clean, minimum_length=3)
        self.remove_short_tracks(clean)

        for track in clean.copy():
            if len(track.track) == 0:
                clean.remove(track)
        #self.remove_high_density_tracks(clean)

        # self.remove_short_tracks(tracks, minimum_length=3)
        # self.remove_shadow_tracks(tracks, margin=0)
        # self.keep_valuable_tracks(tracks, percentage=0.25)

        return clean

    def remove_static_balls(self, tracks, minimum_length=3, minimum_to_split = 100):
        for track in tracks:
            track.track = self.filter_consecutive_objects_with_min_count(track.track, min_count=minimum_length)

        tracks_copy = tracks.copy()
        index = []
        for track in tracks_copy:
            for i, pif in enumerate(track.track):
                if i > 0 and pif.frame_number - track.track[i-1].frame_number >= minimum_to_split:
                    index.append(i)

            index = sorted(set(index))
            # Add the start and end boundaries for slicing
            track_splitted = [track.track[start:end] for start, end in zip([0] + index, index + [len(track.track)])]
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

    def remove_static_balls_v2(self, tracks, minimum_length=3, minimum_to_split=100):
        tracks_copy = tracks.copy()
        for track in tracks_copy:
            clean_tracks, removed_pifs = self.filter_consecutive_objects_with_min_count_v2(track.track, min_count=minimum_length, min_split=minimum_to_split)
            if len(clean_tracks) > 0:
                tracks.remove(track)
                for ct in clean_tracks:
                    nt = Track()
                    for pif in ct:
                        nt.add_pif(pif)
                    tracks.append(nt)
            else:
                if track.check_static() and len(track.track) >= minimum_length:
                    tracks.remove(track)


    def filter_consecutive_objects_with_min_count_v2(self, objects, variance=0.005, min_count=3, min_split=10):
        filtered_objects = []
        current_group = []
        removed = []
        filtered_container = []

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
                        filtered_objects.extend(current_group.copy())
                    else:
                        if len(filtered_objects) > 0:
                            removed.append(current_group)
                            if len(filtered_container) > 0 and filtered_objects[0].frame_number - filtered_container[-1][-1].frame_number <= min_split:
                                filtered_container[-1].extend(filtered_objects.copy())
                            else:
                                filtered_container.append(filtered_objects.copy())

                        filtered_objects = []
                    # Start a new group
                    current_group = [objects[i]]

        # Check the last group collected
        if 0 < len(current_group) < min_count:
            filtered_objects.extend(current_group.copy())
        if min_count > len(filtered_objects) > 0:
            if len(filtered_container) > 0 and filtered_objects[0].frame_number - filtered_container[-1][-1].frame_number <= min_split:
                filtered_container[-1].extend(filtered_objects.copy())
            else:
                filtered_container.append(filtered_objects.copy())


        return filtered_container, removed

    def keep_valuable_tracks(self, tracks, percentage=0.5):
        max_distance = max(track.distance() for track in tracks)

        # Step 2: Set the threshold as a percentage of the maximum value
        threshold = max_distance * percentage

        # Step 3: Filter the list to remove objects below the threshold
        filtered_objects = [track for track in tracks if track.distance() < threshold]
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
        to_remove = [track for track in tracks if len(track.track) <= minimum_length and abs(track.max_min()[1]-track.max_min()[0]) < 0.1]
        for track in to_remove:
            tracks.remove(track)

    def remove_high_density_tracks(self, tracks, max_length=20, max_density = 0.06):
        to_remove = [track for track in tracks if len(track.track) <= max_length and track.density() > max_density]
        for track in to_remove:
            tracks.remove(track)

    def close_tracks(self, frame_number, tolerance=5):
        to_close = [track for track in self.open_tracks if track.last_frame() + tolerance < frame_number]
        for track in to_close:
            track.check_static()
            self.closed_tracks.append(track)
            self.open_tracks.remove(track)

    def track_ball(self, frame, max_distance=0.3):

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

    def get_player_position_over_time(self, tag, axis = 'y', start=0, end=float('inf')):
        player = []
        fn = []

        if end == float('inf'):
            frames = self.frames[start:]
        else:
            frames = self.frames[start:end]
        for frame in frames:
            for p in frame.players():
                if p.tag == tag:
                    if axis == 'y':
                        player.append(1 - p.y)
                    elif axis == 'x':
                        player.append(p.x)

                    fn.append(frame.frame_number)

        return player, fn

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timedelta

    import matplotlib.pyplot as plt
    import numpy as np

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
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn) for fn in frame_numbers]
                ax1.plot(seconds, y_positions, marker='o',
                         label=f'Track from frame {track.track[0].frame_number}', color='green')

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
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            x_positions = [pif.x for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn) for fn in frame_numbers]
                ax2.plot(seconds, x_positions, marker='o',
                         label=f'Track from frame {track.track[0].frame_number}', color='green')

        player_a, fn_a = self.get_player_position_over_time('A', 'x', frame_start, frame_end)
        player_b, fn_b = self.get_player_position_over_time('B', 'x', frame_start, frame_end)
        ax2.plot([frame_to_seconds(fn) for fn in fn_a], player_a, marker='x', label='Player A', color='red')
        ax2.plot([frame_to_seconds(fn) for fn in fn_b], player_b, marker='x', label='Player B', color='yellow')
        player_c, fn_c = self.get_player_position_over_time('C', 'x', frame_start, frame_end)
        player_d, fn_d = self.get_player_position_over_time('D', 'x', frame_start, frame_end)
        ax2.plot([frame_to_seconds(fn) for fn in fn_c], player_c, marker='x', label='Player A', color='blue')
        ax2.plot([frame_to_seconds(fn) for fn in fn_d], player_d, marker='x', label='Player B', color='black')

        ax2.set_xlabel('Time (hh:mm:ss)')
        ax2.set_ylabel('X Position')
        ax2.set_title('Ball and Player X Positions Over Time')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_tracks_with_net(self, tracks, frame_start=-1, frame_end=float('inf'), fps=30):
        if frame_end == float('inf'):
            frame_end = self.frames[-1].frame_number

        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)  # Reintroduce ax3 for the second plot

        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]

            if frame_numbers:
                ax1.plot(frame_numbers, y_positions, marker='o',
                         label=f'Track starting at frame {track.track[0].frame_number}', color='green')

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
        plt.show()

    def plot_tracks(self, tracks, frame_start=-1, frame_end=float('inf'),
                    print_globes=False):  # Adding frame_rate parameter
        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)  # Create two subplots sharing the x-axis

        # First plot (Tracks)
        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]

            if frame_numbers:
                if print_globes:
                    color = 'orange' if track.globe else 'blue'
                    ax1.plot(frame_numbers, y_positions, marker='o',
                             label=f'Track starting at frame {track.track[0].frame_number}', color=color)
                else:
                    if track.quality_track:
                        ax1.plot(frame_numbers, y_positions, marker='x',
                                 label=f'Track starting at frame {track.track[0].frame_number}')
                    else:
                        ax1.plot(frame_numbers, y_positions, marker='o',
                                 label=f'Track starting at frame {track.track[0].frame_number}')

        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Y Position of Ball')
        ax1.set_title('Track of Ball Y Positions Over Time')
        ax1.set_xlim(left=frame_start, right=frame_end)  # Set the x-axis limits
        # ax1.legend()
        ax1.grid(True)

        # Create secondary x-axis for frame numbers
        ax2 = ax1.twiny()
        ax2.set_xlabel('Time in Seconds')
        ax2.set_xlim(left=frame_start / self.fps, right=frame_end / self.fps)
        ax2.set_xticks(ax1.get_xticks() / self.fps)
        ax2.grid(True)

        # Second plot (Track Coverage)
        framecolors = []
        for frames, value in self.playtime.items():
            for i in range(frames[1] - frames[0] + 1):
                framecolors.append(value)
        colors = ['white' if frame == None else 'green' if frame == 'point' else 'red' if frame == 'mess' else 'blue'
                  for frame in framecolors]

        ax3.vlines(range(len(framecolors)), 0, 1, colors=colors)
        ax3.set_xlabel('Frame Number')
        ax3.set_yticks([])
        ax3.set_title('Track Coverage')
        ax3.grid(True)

        # plt.tight_layout()
        plt.show()
