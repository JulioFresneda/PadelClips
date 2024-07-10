from padelLynxPackage.Point import Point
import numpy as np
import matplotlib.pyplot as plt

# Match
# - Top 10 longest points - OK
# - Heatmap for each player
# - Average shots per point - OK
# - Tags: Globes, hot match, etc

# Teams
# - Top 10 points per team
# - How the team played

# Players
# - Top 10 points with more shots by player - OK
# - Km ran - OK
# - Top 3 points where player ran the most - OK
# - Type of player: Globes, smashes, etc
# - % of rights and reves - OK
# - % of shots to the net

# Trophies
# - Player that ran the most - OK
# - Player with most globes
# - Player that did more shots - OK
# - Player most freezed - OK
# - Player that goes more to the net - OK
# - Player that did more shots to the net
# - Negative trophies


class GameStats:
    def __init__(self, frames, points, net):
        self.frames = frames
        self.points = points
        self.net = net

    def print_game_stats(self):
        print("------- Match stats -------")

        top_3_longest_points = self.top_x_longest_points(3)
        print("\nTop 3 longest points: ")
        print(top_3_longest_points)

        avg_shots_per_point = int(self.average_shots_per_point())
        print("\nAverage shots per point: " + str(avg_shots_per_point))

        for player in Point.players:
            print("\n\n------- Player stats: " + player.tag + " -------")
            print("\nTop 3 points with more shots by player")
            top_shots_by_player = self.top_x_points_more_shots_by_player(3, player.tag)
            print(top_shots_by_player)

            print("\nMeters ran: " + str(self.meters_ran(player.tag)))
            print("\nTop 3 points where player ran the most")
            top = self.top_x_points_where_player_ran_the_most(3, player.tag)

            r, b = self.rights_and_backhands(player.tag)
            print("\nRights: " + str(r) + ", backhands: " + str(b) + ", backhands percentage: " + str(b/r*100) + "%")

        print("\n\n------- Trophies -------")
        print("\nPlayer that ran the most: " + self.player_that_ran_the_most().tag)
        shots = self.player_shots()

        print("\nPlayer that did more shots: " + shots[-1][0] + ", " + str(shots[-1][1]) + " shots")
        print("\nPlayer most freezed: " + shots[0][0] + ", " + str(shots[0][1]) + " shots")

        distances = self.player_distances_to_the_net()
        print("\nPlayer that went most to the net: " + distances[0][0])

        #self.players_heatmap()




    def top_x_longest_points(self, x):
        top = sorted(self.points, key=len, reverse=True)[:x]
        return top

    def average_shots_per_point(self):
        total_length = sum(len(obj) for obj in self.points)
        average_length = total_length / len(self.points)
        return average_length

    def top_x_points_more_shots_by_player(self, x, player_tag):
        top = sorted(self.points, key=lambda point: point.how_many_shots_by_player(player_tag), reverse=True)[:x]
        return top

    def meters_ran(self, player_tag, point=None):
        meters = 0.0
        last_pos = None

        if point is None:
            frames_to_count = self.frames
        else:
            try:
                frames_to_count = self.frames[point.first_frame():point.last_frame()]
            except:
                frames_to_count = []

        for frame in frames_to_count:
            player = frame.player(player_tag)
            if player is not None:
                if last_pos is not None:
                    meters += Point.euclidean_distance(last_pos[0], last_pos[1], player.x, player.y)
                last_pos = (player.x, player.y)
        return meters

    def top_x_points_where_player_ran_the_most(self, x, player_tag):
        top = sorted(self.points, key=lambda point: self.meters_ran(player_tag, point), reverse=True)[:x]
        return top

    def rights_and_backhands(self, player_tag):
        rights = 0
        backhands = 0
        for point in self.points:
            shots = [s for s in point.shots if s.tag == player_tag]
            for shot in shots:
                fn = shot.inflection.frame_number
                ball_x_pos = shot.inflection.x
                player = self.frames[fn].player(player_tag)
                if player is not None:
                    player_x_pos = player.x
                    if ball_x_pos > player_x_pos:
                        rights += 1
                    else:
                        backhands += 1

        return rights, backhands

    def player_that_ran_the_most(self):
        top = sorted(Point.players, key=lambda player: self.meters_ran(player.tag), reverse=True)[:1]
        return top[0]

    def player_shots(self):
        shots = {}
        for player in Point.players:
            shots[player.tag] = 0

        for point in self.points:
            for player_tag in shots.keys():
                shots[player_tag] += point.how_many_shots_by_player(player_tag)

        return sorted(shots.items(), key=lambda item: item[1])

    def player_distances_to_the_net(self):
        distances = {}
        for player in Point.players:
            distances[player.tag] = self.average_player_distance_to_the_net(player.tag)
        return sorted(distances.items(), key=lambda item: item[1])

    def average_player_distance_to_the_net(self, player_tag):
        total_distance = 0.0
        shots_to_divide = 0
        for point in self.points:
            for frame in point.point_frames():
                player_pos = frame.player(player_tag)
                total_distance += abs((player_pos.y + player_pos.height / 2) - (self.net.y + self.net.height / 2))
                shots_to_divide += 1
        return total_distance / shots_to_divide



    def players_heatmap(self):
        center = (self.net.x, self.net.y + self.net.height/2)
        coords = []
        for frame in self.frames:
            for player in frame.players():
                coords.append((player.x, player.y))
        self.generate_heatmap(coords, center)


    def adjust_coordinates(self, coords, center):
        """Adjust coordinates based on the center position."""
        # Center is expected to be a tuple (center_x, center_y)
        # Adjusting by subtracting center coordinates to re-center the data
        adjusted_coords = [(x - center[0], y - center[1]) for x, y in coords]
        return adjusted_coords

    def generate_heatmap(self, coords, center, resolution=100):
        """Generate a heatmap from player coordinates."""
        # Adjust coordinates
        coords = self.adjust_coordinates(coords, center)

        # Extract x and y coordinates
        x_coords, y_coords = zip(*coords)

        # Create a histogram with the adjusted coordinates
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=resolution, range=[[-0.5, 0.5], [-0.5, 0.5]],
                                                 density=True)

        # Plot heatmap
        plt.figure(figsize=(8, 5))
        plt.imshow(heatmap.T, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5], aspect='auto', cmap='hot')
        plt.colorbar(label='Density')
        plt.title('Padel Court Player Heatmap')
        plt.xlabel('Horizontal Position')
        plt.ylabel('Vertical Position')
        plt.grid(False)
        plt.show()