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

    def plot_points(self, tracks, points, net, fps):
        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

        y_pos_frn = []
        y_pos_val = []

        for track in tracks:
            y_pos_frn += [pif.frame_number for pif in track.track]
            y_pos_val += [1 - pif.y for pif in track.track]
            #seconds = [frame_to_seconds(fn, 0) for fn in y_pos_frn]
        axes[0].plot(y_pos_frn, y_pos_val, marker='o',
                    label=f'All tracks', color='red')

        axes[0].axhline(y=(1 - net.y) + net.height / 2, color='blue', label='Net (Sup)')
        axes[0].axhline(y=(1 - net.y) - net.height / 2, color='blue', label='Net (Inf)')

        axes[0].set_xlabel('Time (hh:mm:ss)')
        axes[0].set_ylabel('Y Position')
        axes[0].set_title('Ball and Player Y Positions Over Time')
        #
        axes[0].grid(True)

        for point in points:
            highlight_track = [(fn, y) for fn, y in zip(y_pos_frn, y_pos_val) if point.first_frame() <= fn <= point.last_frame()]
            if highlight_track:
                h_sec, h_val = zip(*highlight_track)
                axes[0].plot(h_sec, h_val, marker='o', color='green')

        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(frame_to_seconds(x,0,fps))))


        plt.legend()
        plt.tight_layout()
        plt.show(block=True)



    def plot_tracks(self, tracks, frame_start=0, frame_end=float('inf'), show_players=True, show_net=True):
        matplotlib.use('TkAgg')  # Use the appropriate backend
        fig, axes = plt.subplots(3 if show_players else 2, 1, figsize=(20, 10), sharex=True)

        self.plot_y_positions(axes[0], tracks, frame_start, frame_end, show_players, show_net)
        self.plot_x_positions(axes[1], tracks, frame_start, frame_end, show_players)

        if hasattr(self, 'frames'):
            self.plot_track_coverage(axes[2] if show_players else axes[1])

        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    def plot_y_positions(self, ax, tracks, frame_start, frame_end, show_players, show_net):
        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            y_positions = [1 - pif.y for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn, frame_start) for fn in frame_numbers]
                ax.plot(seconds, y_positions, marker='o',
                         label=f'Track from frame {track.track[0].frame_number}', color='green')

        if show_players:
            self.plot_players(ax, 'y', frame_start, frame_end)

        if show_net:
            ax.axhline(y=(1 - self.net.y) + self.net.height / 2, color='blue', label='Net (Sup)')
            ax.axhline(y=(1 - self.net.y) - self.net.height / 2, color='blue', label='Net (Inf)')

        ax.set_xlabel('Time (hh:mm:ss)')
        ax.set_ylabel('Y Position')
        ax.set_title('Ball and Player Y Positions Over Time')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: format_seconds(x)))
        ax.grid(True)

    def plot_x_positions(self, ax, tracks, frame_start, frame_end, show_players):
        for track in tracks:
            frame_numbers = [pif.frame_number for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            x_positions = [pif.x for pif in track.track if frame_start <= pif.frame_number <= frame_end]
            if frame_numbers:
                seconds = [frame_to_seconds(fn, frame_start) for fn in frame_numbers]
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
