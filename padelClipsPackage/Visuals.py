import matplotlib
import matplotlib.pyplot as plt




def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def frame_to_seconds(frame_number, frame_start, fps):
    return (frame_number - frame_start) / fps


class Visuals:

    @staticmethod
    def plot_frames(frames, fps, start=0, end=None):
        matplotlib.use('TkAgg')
        fig, ax = plt.subplots(figsize=(20, 10))

        for i, frame in enumerate(frames):
            if frame.frame_number >= start and (end is None or frame.frame_number <= end):
                frame_time = frame_to_seconds(frame.frame_number, 0, fps)
                for ball in frame.balls():
                    ax.plot(frame_time, 1 - ball.y, 'o', label=f'Ball at Frame {frame.frame_number}')

        ax.set_xlabel('Time (hh:mm:ss)')
        ax.set_ylabel('Y Position')
        ax.set_title('Balls Positions Over Time')
        ax.grid(True)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_points(tracks, points, net, fps, start=0, end=None):
        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, ax = plt.subplots(figsize=(20, 10))
        y_pos_frn = []
        y_pos_val = []

        for track in tracks:
            if track.first_frame() >= start and (end is None or track.last_frame() <= end):
                y_pos_frn += [pif.frame_number for pif in track.track]
                y_pos_val += [1 - pif.y for pif in track.track]
            #seconds = [frame_to_seconds(fn, 0) for fn in y_pos_frn]
        ax.plot(y_pos_frn, y_pos_val, marker='o',
                    label=f'All tracks', color='red')

        ax.axhline(y=(1 - net.y) + net.height / 2, color='blue', label='Net (Sup)')
        ax.axhline(y=(1 - net.y) - net.height / 2, color='blue', label='Net (Inf)')

        ax.set_xlabel('Time (hh:mm:ss)')
        ax.set_ylabel('Y Position')
        ax.set_title('Ball and Player Y Positions Over Time')
        #
        ax.grid(True)

        for point in points:
            if point.first_frame() >= start and (end is None or point.last_frame() <= end):

                highlight_track = [(fn, y) for fn, y in zip(y_pos_frn, y_pos_val) if point.first_frame() <= fn <= point.last_frame()]
                if highlight_track:
                    h_sec, h_val = zip(*highlight_track)
                    ax.plot(h_sec, h_val, marker='o', color='green')

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(frame_to_seconds(x,0,fps))))


        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    @staticmethod
    def plot_tracks_tagged(tracks, frames, net, fps, start=0, end=None):
        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, ax = plt.subplots(figsize=(20, 10))
        y_pos_frn = []
        y_pos_val = []

        for track in tracks:
            if track.first_frame() >= start and (end is None or track.last_frame() <= end):
                y_pos_frn += [pif.frame_number for pif in track.track]
                y_pos_val += [1 - pif.y for pif in track.track]
            # seconds = [frame_to_seconds(fn, 0) for fn in y_pos_frn]
        ax.plot(y_pos_frn, y_pos_val, marker='o',
                label=f'All tracks', color='red')

        ax.axhline(y=(1 - net.y) + net.height / 2, color='blue', label='Net (Sup)')
        ax.axhline(y=(1 - net.y) - net.height / 2, color='blue', label='Net (Inf)')

        ax.set_xlabel('Time (hh:mm:ss)')
        ax.set_ylabel('Y Position')
        ax.set_title('Ball and Player Y Positions Over Time')
        #
        ax.grid(True)

        for frame in frames:
            if frame.frame_number >= start and (end is None or frame.frame_number <= end):

                highlight_track = [(fn, y) for fn, y in zip(y_pos_frn, y_pos_val) if
                                   frame.frame_number <= fn <= frame.frame_number]
                if highlight_track:
                    h_sec, h_val = zip(*highlight_track)
                    ax.plot(h_sec, h_val, marker='o', color='green')

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(frame_to_seconds(x, 0, fps))))

        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    def plot_tracks(self, tracks, frames, frame_start=0, frame_end=float('inf'), fps=30, net=None):
        # Set the appropriate backend
        matplotlib.use('TkAgg')

        # Create two subplots (one above the other)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

        # Plot y and x positions on the first subplot
        self.plot_y_positions(ax1, tracks, frame_start, frame_end, fps, False, net)
        self.plot_x_positions(ax1, tracks, frame_start, frame_end, fps, False)

        # Plot vertical lines for tags on the second subplot
        self.plot_frame_tags(ax2, frames, frame_start, frame_end)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show(block=True)

    def plot_frame_tags(self, ax, frames, frame_start, frame_end):
        fps = 60
        frame_numbers = frames[frame_start:frame_end]
        seconds = [frame_to_seconds(fn.frame_number, frame_start, fps) for fn in frame_numbers]
        tags = [frame.tag for frame in frames if frame_start <= frame.frame_number <= frame_end]

        for sec, tag in zip(seconds, tags):
            color = 'green' if tag == 'point' else 'red'
            # Plot a vertical line at the corresponding second
            ax.axvline(x=sec, color=color, linestyle='-', linewidth=2)
            # Optionally, add a label for the first occurrence of each color
            if color == 'green' and not hasattr(self, 'green_label_added'):
                ax.axvline(x=sec, color=color, linestyle='-', linewidth=2, label='Point')
                setattr(self, 'green_label_added', True)
            elif color == 'red' and not hasattr(self, 'red_label_added'):
                ax.axvline(x=sec, color=color, linestyle='-', linewidth=2, label='Non-point')
                setattr(self, 'red_label_added', True)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Tag Indicator')
        ax.set_title('Frame Tags Over Time')
        ax.legend()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))





    def plot_y_positions(self, ax, tracks, frame_start, frame_end, fps, show_players, net):
        colors = ['green', 'yellow', 'red', 'blue', 'black']
        for i, track in enumerate(tracks):
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn, frame_start, fps) for fn in frame_numbers]
                ax.plot(seconds, y_positions, marker='o',
                         label=f'Track from frame {track.track[0].frame_number}', color=colors[i%5])

        if show_players:
            self.plot_players(ax, 'y', frame_start, frame_end)

        if net is not None:
            ax.axhline(y=(1 - net.y) + net.height / 2, color='blue', label='Net (Sup)')
            ax.axhline(y=(1 - net.y) - net.height / 2, color='blue', label='Net (Inf)')

        ax.set_xlabel('Time (hh:mm:ss)')
        ax.set_ylabel('Y Position')
        ax.set_title('Ball and Player Y Positions Over Time')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))
        ax.grid(True)
        return seconds

    def plot_x_positions(self, ax, tracks, frame_start, frame_end, fps, show_players):
        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            x_positions = [pif.x for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn, frame_start, fps) for fn in frame_numbers]
                ax.plot(seconds, x_positions, marker='o',
                         label=f'Track from frame {track.track[0].frame_number}', color='green')

        if show_players:
            self.plot_players(ax, 'x', frame_start, frame_end)

        ax.set_xlabel('Time (hh:mm:ss)')
        ax.set_ylabel('X Position')
        ax.set_title('Ball and Player X Positions Over Time')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))
        ax.grid(True)

    def plot_players(self, ax, axis, frame_start, frame_end):
        players = ['A', 'B', 'C', 'D']
        colors = ['red', 'yellow', 'blue', 'black']
        for player, color in zip(players, colors):
            player_pos, frame_nums = self.get_player_position_over_time(player, axis, frame_start, frame_end)
            ax.plot([frame_to_seconds(fn, frame_start) for fn in frame_nums], player_pos, marker='x', label=f'Player {player}', color=color)

    def plot_track_coverage(self, ax):
        colors = [
            'green' if frame.tag == 'point' else 'red' if frame.tag == 'mess' else 'yellow' if frame.tag == 'break' else 'blue'
            for frame in self.frames_controller.frame_list]
        ax.vlines(range(len(self.frames_controller)), 0, 1, colors=colors)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Coverage')
        ax.set_title('Track Coverage')
        ax.grid(True)

    def get_player_position_over_time(self, player, axis, frame_start, frame_end):
        # Placeholder method, replace with actual logic
        return [], []

# Usage example:
# plotter = PadelPlotter(fps=30, net=net_object, frames=frames_object)
# plotter.plot_tracks(tracks, frame_start=0, frame_end=1000, show_players=True, show_net=True)
