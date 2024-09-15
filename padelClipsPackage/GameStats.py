from padelClipsPackage.Point import Point, Shot
import numpy as np
import matplotlib.pyplot as plt

from padelClipsPackage.PositionInFrame import PositionInFrame
from padelClipsPackage.Shot import Category, Position


class GameStats:
    def __init__(self, game):
        self.frames_controller = game.frames_controller
        self.points = game.points
        self.net = game.net
        self.game = game

        self.medals = {
            "Nube": self.nube(),
            "Trueno": self.trueno(),
            "Destello": self.destello(),
            "Bastion": self.bastion(),
            "Vanguardia": self.vanguardia(),
            "Nevera": self.nevera(),
            "Iman": self.iman(),
            "Obsesivo": self.obsesivo(),
            "Pulpo": self.pulpo(),
            "Hipopotamo": self.hipopotamo()
        }
        self.medals_descriptions = {
            "Nube": ["Jugador mas globero.", "{{ result }} globos registrados.", "Un {{ result }} más que el segundo."],
            "Trueno": ["Jugador con mas smashes.", "{{ result }} smashes registrados.", "Un {{ result }} más que el segundo."],
            "Destello": ["Smash mas veloz del partido.", "Una velocidad de {{ result }} m/s.", "Un {{ result }} más que la media."],
            "Bastion": ["Jugador con mas tiempo en fondo de pista.", "{{ result }} minutos registrados.", "Un {{ result }} más que el segundo."],
            "Vanguardia": ["Jugador con mas tiempo en red.", "{{ result }} minutos registrados.", "Un {{ result }} más que el segundo."],
            "Nevera": ["Jugador que menos bolas ha recibido.", "{{ result }} de bolas registradas.", "Un {{ result }} menos que el segundo."],
            "Iman": ["Jugador que mas bolas ha recibido.", "{{ result }} de bolas registradas.", "Un {{ result }} más que el segundo."],
            "Obsesivo": ["Jugador que mas bolas ha lanzado a un mismo rival.", "{{ result }} de bolas registradas hacia {{ result2 }}.", "Un {{ result }} más que a su compañero."],
            "Pulpo": ["Jugador con mayor equilibrio en derecha y reves.", "{{ result }} de derechas.", "Un {{ result }} menos que el segundo."],
            "Hipopotamo": ["Jugador mas territorial.", "{{ result }} de campo abarcado.", "Un {{ result }} más que su compañero."]
        }

    #########################################################
    # WELCOME TO THE AWESOME GAME AWARDS AND GAME STATS!
    #
    # MEDALLAS (INDIVIDUALES)
    #   -   Nube        - Más globos
    #   -   Destello    - Smash más rápido
    #   -   Trueno      - Mayor número de smashes
    #
    #   -   Bastión     - Mas cerca de red
    #   -   Vanguardia  - Mas lejos de red
    #
    #   -   Nevera      - Menos bolas recibidas
    #   -   Iman        - Mas bolas recibidas
    #   -   Obsesivo    - Mas bolas al mismo rival
    #
    #   -   Cortarollos - Acaba puntos (no estoy seguro por imprecision)
    #
    #   -   Pulpo       - % izq-dcha 50/50
    #   -   Hipopotamo  - Jugador mas territorial (acaparador)
    #
    #   -   MVP         - Mejor jugador, basado en stats
    #
    # TROFEOS (A PAREJAS)
    #   -   Trofeo de agresividad  - Pareja mas cercana a la red
    #   -   Trofeo de agilidad     - Pareja con mas metros corridos
    #   -   Trofeo de polivalencia - Pareja donde los jugadores cambian mas de posicion
    #


    def get_medals(self, tag):
        medals = {}
        for name, winner in self.medals.items():
            if winner[0] == tag:
                medals[name] = winner[1]

        return medals


    def print_game_stats(self):
        print("------- WELCOME TO THE AWESOME GAME AWARDS AND GAME STATS! -------")
        print("\n---------- MEDALLAS ----------")
        print("----------> Nube")
        nube = self.nube()
        print(nube)

        print("----------> Trueno")
        trueno = self.trueno()
        print(trueno)

        print("----------> Destello")
        destello = self.destello()
        print(destello)

        print("----------> Bastion")
        bastion = self.bastion()
        print(bastion)

        print("----------> Vanguardia")
        vanguardia = self.vanguardia()
        print(vanguardia)

        print("----------> Nevera")
        nevera = self.nevera()
        print(nevera)

        print("----------> Iman")
        iman = self.iman()
        print(iman)

        print("----------> Obsesivo")
        obsesivo = self.obsesivo()
        print(obsesivo)

        print("----------> Pulpo")
        pulpo = self.pulpo()
        print(pulpo)

        print("----------> Hipopotamo")
        hipopotamo = self.hipopotamo()
        print(hipopotamo)

        print("----------> MVP")
        mpv = self.mvp()

    def mvp(self):
        print("Por programar")

    def hipopotamo(self):
        positions = {}
        for player in self.game.player_templates:
            positions[player.tag] = []
        for frame in self.game.frames_controller.frame_list:
            for player in frame.players():
                left = self.game.players_boundaries_horizontal[player.position]['left']
                right = self.game.players_boundaries_horizontal[player.position]['right']
                pos = self.scale_position(player.x, left, right)
                positions[player.tag].append(pos)

        variances = []
        for player, positions in positions.items():
            variances.append((player, np.std(positions)))

        return sorted(variances, key=lambda p: p[1])[-1]

    def pulpo(self):
        players_ordered = []
        for player in self.game.player_templates:
            r, l = self.rights_and_backhands(player.tag)
            players_ordered.append((player.tag, abs(l - r)))
        return sorted(players_ordered, key=lambda p: p[1])[0]

    def scale_position(self, x, a, b):
        return abs((x - a) / (b - a))

    def nevera(self):
        return self.get_player_most_shots()[-1]

    def iman(self):
        return self.get_player_most_shots()[0]

    def obsesivo(self):
        shots = self.get_shots()
        striker_receiver = {}
        for shot in shots:
            if shot.category is not Category.SERVE:
                if shot.striker is not None and shot.receiver is not None:
                    if (shot.striker.tag, shot.receiver.tag) not in striker_receiver.keys():
                        striker_receiver[(shot.striker.tag, shot.receiver.tag)] = 0
                    striker_receiver[(shot.striker.tag, shot.receiver.tag)] += 1
        count_shots_by_player = []
        for players, shots in striker_receiver.items():
            count_shots_by_player.append((players, shots))
        count_shots_by_player = sorted(count_shots_by_player, key=lambda x: x[1], reverse=True)
        winner = count_shots_by_player[0]
        return (winner[0][0], (winner[0][1], winner[1]))

    def bastion(self):
        distances = self.player_distances_to_the_net()
        return distances[-1]

    def vanguardia(self):
        distances = self.player_distances_to_the_net()
        return distances[0]

    def nube(self):
        return self.get_player_most_shots(Category.FULL_GLOBE)[0]

    def trueno(self):
        return self.get_player_most_shots(Category.SMASH)[0]

    def destello(self):
        shots = self.get_shots()
        fastest_shot = None
        fastest_vel = float('inf')

        for shot in shots:
            if shot.category is Category.SMASH:
                vel = PositionInFrame.speed_list(shot.pifs, scale=100)
                if fastest_vel > vel:
                    fastest_shot = shot
                    fastest_vel = vel
        return (shot.striker.tag, (fastest_shot, fastest_vel))

    def get_player_most_shots(self, category=None):
        shots = self.get_shots()
        shots_by_player = {}
        for shot in shots:
            if shot.category is category or category is None:
                if shot.striker is not None:
                    if shot.striker.tag not in shots_by_player.keys():
                        shots_by_player[shot.striker.tag] = []
                    shots_by_player[shot.striker.tag].append(shot)
        count_shots_by_player = []
        for player, shots in shots_by_player.items():
            count_shots_by_player.append((player, len(shots)))
        count_shots_by_player = sorted(count_shots_by_player, key=lambda x: x[1], reverse=True)
        return count_shots_by_player

    def categorize_player_shots(self):
        categories = {}
        for player in self.game.player_templates:
            r, l = self.rights_and_backhands(player.tag)
            r = int(r*100/(r+l))
            l = 100 - r
            categories[player.tag] = {Category.SMASH: 0, Category.FULL_GLOBE: 0, 'left': l, 'right': r}

        for shot in self.get_shots():
            if shot.striker is not None:
                if shot.category is Category.SMASH or shot.category is Category.FULL_GLOBE:
                    categories[shot.striker.tag][shot.category] += 1

        return categories




    def get_shots(self):
        shots = []
        for point in self.points:
            for shot in point.shots:
                shots.append(shot)
        return shots

    def top_x_longest_points(self, x):
        top = sorted(self.points, key=lambda p: p.duration(), reverse=True)
        if len(top) > x:
            top = top[:x]
        return top

    def top_x_more_shots(self, x):
        top = sorted(self.points, key=lambda p: len(p), reverse=True)
        if len(top) > x:
            top = top[:x]
        return top

    def top_x_points_more_meters_ran(self, x):
        top = sorted(self.points, key=lambda point: self.overall_meters_ran(point), reverse=True)
        if len(top) > x:
            top = top[:x]
        return top

    def average_shots_per_point(self):
        total_length = sum(len(obj.shots) for obj in self.points)
        average_length = total_length / len(self.points)
        return average_length

    def top_x_points_more_shots_by_player(self, x, player_tag):
        top = sorted(self.points, key=lambda point: point.how_many_shots_by_player(player_tag), reverse=True)[:x]
        return top

    def overall_meters_ran(self, point=None):
        meters = 0
        for p in Point.game.player_templates:
            meters += self.meters_ran(p.tag, point)
        return meters

    def meters_ran(self, player_tag, point=None):
        meters = 0.0
        last_pos = None

        if point is None:
            frames_to_count = self.frames_controller.frame_list
        else:
            try:
                frames_to_count = self.frames_controller.get(point.start(), point.end())
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
            shots = [s for s in point.shots if s.striker is not None and s.striker.tag == player_tag]
            for shot in shots:
                fn = shot.striker_pif.frame_number
                ball_x_pos = shot.striker_pif.x
                player = self.frames_controller.get(fn).player(player_tag)
                if player is not None:
                    player_x_pos = player.x
                    if ball_x_pos > player_x_pos:
                        rights += 1
                    else:
                        backhands += 1

        return rights, backhands

    def player_that_ran_the_most(self):
        top = sorted(Point.game.player_templates, key=lambda player: self.meters_ran(player.tag), reverse=True)[:1]
        return top[0]

    def total_shots(self):
        total = 0
        for point in self.points:
            total += len(point.shots)
        return total

    def player_shot_number(self, player_tag=None):
        shots = {}
        for player in Point.game.player_templates:
            shots[player.tag] = 0

        for point in self.points:
            for pt in shots.keys():
                shots[pt] += point.how_many_shots_by_player(pt)

        if player_tag is None:
            return sorted(shots.items(), key=lambda item: item[1])
        else:
            return shots[player_tag]

    def player_shots(self, player_tag, category=None):
        shots = []
        for point in self.points:
            for shot in point.shots:
                if shot.tag == player_tag and shot.category == category or category is None:
                    shots.append(shot)
        return shots

    def player_distances_to_the_net(self):
        distances = {}
        for player in self.game.player_templates:
            distances[player.tag] = self.average_player_distance_to_the_net(player.tag)
        return sorted(distances.items(), key=lambda item: item[1])

    def average_player_distance_to_the_net(self, player_tag):
        total_distance = 0.0
        shots_to_divide = 0
        for point in self.points:
            for frame in point.point_frames():
                player_pos = frame.player(player_tag)
                net = self.net.y + self.net.height / 2
                player = player_pos.y + player_pos.height / 2
                if net > player:
                    wall = self.game.players_boundaries_vertical[Position.TOP]
                else:
                    wall = self.game.players_boundaries_vertical[Position.BOTTOM]

                distance = self.scale_position(player, wall, net)

                total_distance += distance
                shots_to_divide += 1
        return total_distance / shots_to_divide

    def total_points(self):
        return len(self.points)

    def players_heatmap(self):
        center = (self.net.x, self.net.y + self.net.height / 2)
        coords = []
        for i, frame in self.frames_controller.enumerate():
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
