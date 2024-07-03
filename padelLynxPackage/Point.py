from padelLynxPackage import Track
from padelLynxPackage.PositionTracker import PositionTracker
import math

class Shot:
    def __init__(self, pifs, position = None):
        self.pifs = pifs
        self.position = position
        self.tag = None

        if self.position == 'over':
            self.inflection = min(self.pifs, key=lambda x: x.y)
        else:
            self.inflection = max(self.pifs, key=lambda x: x.y)

    def tag_shot(self, tag):
        self.tag = tag


class Point:

    net = None
    frames = None
    players = None
    def __init__(self, track: Track):
        self.track = track
        self.shots = self.track_to_shots()

        for shot in self.shots:
            shot.tag_shot(self.shortest_player(shot.inflection, shot.position))

        self.print_shots()


    def print_shots(self):
        for shot in self.shots:
            if len(shot.pifs) > 0:
                print("Player " + str(shot.tag) + ": " + str(shot.pifs[0].frame_number) + " -> " + str(shot.pifs[-1].frame_number))
        print('\n')




    def shortest_player(self, pif, position, max_distance=0.2):
        all_players = Point.frames[pif.frame_number].players()
        all_players = sorted(all_players, key=lambda player: player.y)
        if all_players != None:
            if position == 'over':
                if len(all_players) > 2:
                    players = all_players[:2]
                else:
                    players = all_players
            else:
                if len(all_players) > 2:
                    players = all_players[2:]
                else:
                    players = all_players




            if len(players) > 0:
                dis = self.euclidean_distance(players[0].x, players[0].y, pif.x, pif.y)
                tag = players[0].tag

                if len(players) > 1:
                    for p in players[1:]:
                        _dis = self.euclidean_distance(p.x, p.y, pif.x, pif.y)
                        if _dis < dis:
                            dis = _dis
                            tag = p.tag

                if dis <= max_distance:
                    return tag
                else:
                    return None
            return None
        return None

    def euclidean_distance(self, x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
    def track_to_shots(self):
        shots = []
        buffer = []
        initial_pos = self.position_over_the_net(self.track.track[0].y, Point.net)

        for pif in self.track.track:
            pif_pos = self.position_over_the_net(pif.y, Point.net)
            if pif_pos == initial_pos or initial_pos == 'middle' or pif_pos == 'middle' and initial_pos == 'under':
                buffer.append(pif)
                if initial_pos == 'middle':
                    initial_pos = self.position_over_the_net(pif.y, Point.net)

            else:
                shots.append(Shot(buffer, position=initial_pos))
                buffer = [pif]
                initial_pos = self.position_over_the_net(pif.y, Point.net)
                if initial_pos == 'middle':
                    initial_pos = 'under'
        if len(buffer) > 0:
            shots.append(Shot(buffer, position=initial_pos))
        return shots

    def position_over_the_net(self, y, net):
        net_upper_pos = net.y - net.height / 2
        net_lower_pos = net.y + net.height / 2
        if y < net_upper_pos:
            return 'over'
        elif y > net_lower_pos:
            return 'under'
        else:
            return 'middle'


