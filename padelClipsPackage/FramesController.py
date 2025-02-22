from padelClipsPackage.Frame import Frame
from padelClipsPackage.Object import Label, PlayerTemplate
from padelClipsPackage.aux import apply_kalman_filter
import time


class FramesController:
    def __init__(self, frame_list):
        self.frame_list = frame_list
        self.remove_extra_players()
        self.find_frame_templates()

    def remove_extra_players(self):
        for frame in self.frame_list:
            if len(frame.players()) > 4:
                to_remove = []
                new_players = sorted(frame.players(), key=lambda p: p.conf, reverse=True)[:4]

                for obj in frame.objects:
                    if obj.class_label is Label.PLAYER.value:
                        if obj not in new_players:
                            to_remove.append(obj)
                for obj in to_remove:
                    frame.objects.remove(obj)

    def __len__(self):
        return len(self.frame_list)

    def frame_tags(self):
        return [frame.tag for frame in self.frame_list]

    def get_player_positions(self, tag, start=0, end=None):
        if end is None:
            end = len(self.frame_list)

        frames = self.frame_list[start:end]
        pos = []
        for frame in frames:
            player = frame.player(tag)
            pos.append((player.x, player.y + player.height / 2))
        return pos

    def get(self, index, index_end=None):
        if index_end is None:
            return self.frame_list[index]
        else:
            return self.frame_list[index:index_end]

    def enumerate(self):
        return enumerate(self.frame_list)

    def get_template_players(self):
        return self.template_players

    def tag_frames(self, players, player_features):
        player_pos = {'A': [], 'B': [], 'C': [], 'D': []}
        player_idx = {'A': [], 'B': [], 'C': [], 'D': []}

        for i, frame in enumerate(self.frame_list):
            if i % 100 == 0:
                pass
                #print("Tagging frame " + str(i) + " out of " + str(len(self.frame_list)), end='\r')

            # Tag players
            self.tag_players_in_frame(frame, players, player_features)

            # Save position and idx to future
            for player in frame.players():
                player_pos[player.tag].append((player.x, player.y))
                player_idx[player.tag].append(i)

        return player_pos, player_idx

    def fill_missing_positions(self, index_positions, values, total_new_values):
        # Initialize new lists for the complete index and values
        new_index = list(range(total_new_values))
        new_values = []

        # Keep track of the current value
        current_value = values[0] if index_positions[0] > 0 else None
        value_idx = 0

        # Iterate over the new index list and fill in the values
        for idx in new_index:
            if idx in index_positions:
                current_value = values[value_idx]
                value_idx += 1
            new_values.append(current_value)

        return new_values

    def smooth_player_tags(self, player_pos, player_idx, number_of_frames):

        fixed_player_pos = {}
        for player_tag in player_pos.keys():
            fixed_player_pos[player_tag] = self.fill_missing_positions(player_idx[player_tag], player_pos[player_tag],
                                                                       number_of_frames)
        smoothed = {}
        for player_tag in player_pos.keys():
            print("Smoothing tag " + player_tag, end='\n')
            smoothed[player_tag] = apply_kalman_filter(fixed_player_pos[player_tag])

        for tag in smoothed.keys():
            for i, pos in enumerate(smoothed[tag]):
                if pos is None:
                    print(i)
        for i, frame in enumerate(self.frame_list):
            for player_tag in smoothed.keys():
                frame.update_player_position(player_tag, smoothed[player_tag][i][0], smoothed[player_tag][i][1])

    def tag_players_in_frame(self, frame: Frame, players, player_features):

        def get_player_features(tag):
            pf = player_features[str(int(tag))]
            return pf

        matches = []
        pairs = {}

        tags = {}
        for player in players:
            tags[player.tag] = player

        players_from_frame = []
        for obj in frame.players():
            players_from_frame.append(obj)

        for tag in tags.keys():
            for obj in players_from_frame:
                player_in_frame_ft = get_player_features(obj.tag)
                dist = PlayerTemplate.features_distance(tags[tag].template_features, player_in_frame_ft)
                pairs[(tag, obj.tag)] = dist

        while len(pairs.keys()) > 0:
            lowest_dist = float('inf')
            lowest_pair = (None, None)
            for (tag, idx), dist in pairs.items():
                if dist < lowest_dist:
                    lowest_dist = dist
                    lowest_pair = (tag, idx)
            matches.append(lowest_pair)
            tag = lowest_pair[0]
            idx = lowest_pair[1]
            tmp = list(pairs.keys()).copy()
            for pair in tmp:
                if tag == pair[0]:
                    pairs.pop(pair)
                elif idx == pair[1]:
                    pairs.pop(pair)

        for match in matches:
            frame.update_player_tag(match[1], match[0])

        for player in frame.players():
            if player.tag != 'A' and player.tag != 'B' and player.tag != 'C' and player.tag != 'D':
                frame.objects.remove(player)

    def find_frame_templates(self):
        best_frame_players = None
        best_frame_net = None
        best_avg_conf_players = 0.0
        best_avg_conf_net = 0.0


        label_number_net = Label.NET.value

        for i, frame in enumerate(self.frame_list):
            if i % 100 == 0:
                print("Looking for frame templates (NET): " + str(i) + "/" + str(len(self.frame_list)), end='\r')
            # Filter objects with class_label == 1
            net_objects = [obj for obj in frame.objects if obj.class_label == label_number_net]

            # Check if there are exactly four such objects
            if len(net_objects) == 1:
                # Calculate the average confidence of these objects
                avg_conf_net = net_objects[0].conf

                # Find the frame where this average deviation is minimized
                if best_avg_conf_net < avg_conf_net:
                    best_avg_conf_net = avg_conf_net
                    best_frame_net = frame

        net = [obj for obj in best_frame_net.objects if obj.class_label == Label.NET.value][0]
        for i, frame in enumerate(self.frame_list):
            if i % 100 == 0:
                print("Looking for frame templates (PLAYERS): " + str(i) + "/" + str(len(self.frame_list)), end='\r')
            # Filter objects with class_label == 4

            players = frame.players()
            sorted_players = sorted(players, key=lambda p: p.get_foot(), reverse=True)
            if len(sorted_players) == 4 and sorted_players[1].get_foot() > net.get_foot() > sorted_players[2].get_foot():
                # Calculate the average confidence of these objects
                avg_conf_player = sum(obj.conf for obj in sorted_players) / len(sorted_players)

                # Find the frame where this average deviation is minimized
                if best_avg_conf_players < avg_conf_player:
                    best_avg_conf_players = avg_conf_player
                    best_frame_players = frame

        self.template_players = best_frame_players
        self.template_net = best_frame_net
