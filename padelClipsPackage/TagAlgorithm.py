from statistics import mean

from padelClipsPackage.Object import PlayerTemplate
from padelClipsPackage.PositionInFrame import PositionInFrame
from padelClipsPackage.Shot import Position
from collections import Counter

class TagAlgorithm:
    def __init__(self, frames, origin_fn, template_players, player_features, net):
        self.frames = frames
        self.origin_fn = origin_fn
        self.template_players = template_players
        self.player_features = player_features
        self.net = net



    def tag(self):
        print("Tagging players...")
        self.tag_pos_level(Position.TOP)
        self.tag_pos_level(Position.BOTTOM)

    def tag_pos_level(self, position):
        self.tags = [p.tag for p in self.template_players if p.position is position]


        to_right = self.frames[self.origin_fn:]
        to_left = self.frames[:self.origin_fn][::-1]

        self.tag_direction_level(to_right, position)
        self.tag_direction_level(to_left, position)


    def tag_direction_level(self, frames, position):
        last_players = self.frames[self.origin_fn].players(position)
        last_seen = {}
        for lp in last_players:
            last_seen[lp.new_tag] = (lp, self.origin_fn)

        # Multiple detection same player
        for frame in frames:
            tags_free = self.tags.copy()
            players = frame.players(position)


            for lp in last_players:
                for p in players:
                    if lp.tag == p.tag:
                        p.new_tag = lp.new_tag
                        tags_free.remove(lp.new_tag)

            if len(tags_free) == 1:
                for p in players:
                    if p.new_tag is None:
                        p.new_tag = tags_free[0]

            # 2 out, 2 new
            elif len(tags_free) == 2:
                self.tag_algorithm_dists_deepf(frame, players, tags_free, last_seen)

            for p in players:
                last_seen[p.new_tag] = (p, frame.frame_number)







    def tag_algorithm_dists_deepf(self, frame, players, tags_free, last_seen):
        threshold = 0.5
        distances = {}
        for tag in tags_free.copy():
            # Last seen recently
            if frame.frame_number - last_seen[tag][1] < 180:
                for p in players:
                    dist_with_last_seen = PositionInFrame.distance_to(last_seen[tag][0], p)
                    if dist_with_last_seen < threshold:
                        distances[(p, tag)] = dist_with_last_seen

        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        used = []
        for pair in sorted_distances.keys():
            if pair[1] not in used and pair[0].tag not in used:
                pair[0].new_tag = pair[1]
                used.append(pair[1])
                used.append(pair[0].tag)
        for u in used:
            if u in tags_free:
                tags_free.remove(u)

        # If only one seen recently
        if len(tags_free) == 1:
            for p in players:
                if p.new_tag is None:
                    p.new_tag = tags_free[0]
        # None seen recently, use deepf
        elif len(tags_free) == 2:
            threshold = 100

            distances = {}

            for p in players:
                for tag in tags_free:
                    tplayer_deepf = [tp for tp in self.template_players if tp.tag == tag][0].template_features
                    player_deepf = self.player_features[str(int(p.tag))]

                    dist = PlayerTemplate.features_distance(tplayer_deepf, player_deepf)
                    distances[(p, tag)] = dist

            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            used = []
            for pair in sorted_distances.keys():
                if pair[1] not in used and pair[0].tag not in used:
                    pair[0].new_tag = pair[1]
                    used.append(pair[1])
                    used.append(pair[0].tag)







    def tag_old(self, frames):

        self.new_tags = ['A', 'B', 'C', 'D']
        last_players = self.start_frame.players()

        self.top_bottom = {}
        self.top_bottom[Position.TOP] = [p for p in self.template_players if p.position is Position.TOP]
        self.top_bottom[Position.BOTTOM] = [p for p in self.template_players if p.position is Position.BOTTOM]

        self.top_bottom_tags = {}
        self.top_bottom_tags[Position.TOP] = [p.tag for p in self.top_bottom[Position.TOP]]
        self.top_bottom_tags[Position.BOTTOM] = [p.tag for p in self.top_bottom[Position.BOTTOM]]


        last_seen = {}

        for frame in frames:
            # MATCH 1:1
            TagAlgorithm.tag_positions(last_players, frame.players(), self.net)

            matches = TagAlgorithm.match(last_players, frame.players())

            tags_used = []
            for match in matches:
                match[1].new_tag = match[0].new_tag
                tags_used.append(match[0].new_tag)

            new_players = [p for p in frame.players() if p.new_tag is None]
            candidates = [tag for tag in self.new_tags if tag not in tags_used]

            # MATCH 0:1 TOPS and BOTTOM
            for position in [Position.TOP, Position.BOTTOM]:
                new_players_ = [p for p in new_players if p.position is position]
                candidates_ = [tag for tag in candidates if tag in self.top_bottom_tags[position]]
                if len(new_players_) > 0:
                    self.match_new(new_players_, candidates_, last_seen, frame.frame_number)

            last_players = frame.players()
            for player in last_players:
                last_seen[player.new_tag] = (player, frame.frame_number)


    @staticmethod
    def tag_positions(left, right, net):
        for lp in left:
            for rp in right:
                if rp.tag == lp.tag:
                    rp.position = lp.position

        top = [p for p in right if p.position is Position.TOP]
        bottom = [p for p in right if p.position is Position.BOTTOM]
        none = [p for p in right if p.position is None]
        none = sorted(none, key=lambda p: p.get_foot(), reverse=True)

        for n in none:
            if len(bottom) == 2:
                n.position = Position.TOP
            elif len(top) == 2:
                n.position = Position.BOTTOM

            else:
                if n.get_foot() > net.get_foot() + 0.1:
                    n.position = Position.BOTTOM
                    bottom.append(n)
                elif n.get_foot() < net.get_foot() - 0.1:
                    n.position = Position.TOP
                    top.append(n)

        top = [p for p in right if p.position is Position.TOP]
        bottom = [p for p in right if p.position is Position.BOTTOM]
        none = [p for p in right if p.position is None]
        none = sorted(none, key=lambda p: p.get_foot(), reverse=True)

        for n in none:
            if len(bottom) == 2:
                n.position = Position.TOP
            elif len(top) == 2:
                n.position = Position.BOTTOM

            else:
                if n.get_foot() > net.get_foot():
                    n.position = Position.BOTTOM
                    bottom.append(n)
                else:
                    n.position = Position.TOP
                    top.append(n)


    def match_new(self, new_players, candidates, last_seen, fn, use_dist=120):
        def get_player_features(tag):
            pf = self.player_features[str(int(tag))]
            return pf

        tagged = []
        new_players_to_tag = new_players

        candidates_dist = [c for c in candidates if fn - last_seen[c][1] <= use_dist]
        candidates_deepf = [c for c in candidates if c not in candidates_dist]

        # First tag dist-related
        for candidate in candidates_dist:
            if len(new_players_to_tag) > 0:
                candidate_ls, candidate_ls_fn = last_seen[candidate]

                closest_player = new_players_to_tag[0]
                closest_dist = PositionInFrame.distance_to(candidate_ls, closest_player)

                for nplayer in new_players_to_tag:

                    dist = PositionInFrame.distance_to(nplayer, candidate_ls)

                    if dist < closest_dist:
                        closest_player = nplayer
                        closest_dist = dist


                closest_player.new_tag = candidate
                tagged.append(candidate)
                new_players_to_tag.remove(closest_player)



        # Then tag with deepf
        for nplayer in new_players_to_tag:
            closest_candidate = candidates_deepf[0]
            closest_dist = 100
            for candidate in candidates_deepf:

                tplayer = [tp for tp in self.template_players if tp.tag == candidate][0]
                player_in_frame_ft = get_player_features(nplayer.tag)

                dist = PlayerTemplate.features_distance(tplayer.template_features, player_in_frame_ft)

                if dist < closest_dist:
                    closest_candidate = candidate
                    closest_dist = dist

            nplayer.new_tag = closest_candidate
            tagged.append(closest_candidate)
            candidates_deepf.remove(closest_candidate)


    @staticmethod
    def match(left, right):
        left_ = [p for p in left]
        right_ = [p for p in right]

        matches = []
        # Same-tag matches
        for lplayer in left:
            for rplayer in right:
                if lplayer.tag == rplayer.tag:
                    matches.append((lplayer, rplayer))
                    left_.remove(lplayer)
                    right_.remove(rplayer)

        # Different-tag matches, based on distances
        distances = TagAlgorithm.distances(left_, right_)
        for pair, dist in distances.items():
            if pair[0] in left_ and pair[1] in right_:
                matches.append(pair)
                left_.remove(pair[0])
                right_.remove(pair[1])

        return matches

    @staticmethod
    def distances(left, right):
        distances = {}
        for lplayer in left:
            for rplayer in right:
                dist = PositionInFrame.distance_to(lplayer, rplayer)
                distances[(lplayer, rplayer)] = dist
        return dict(sorted(distances.items(), key=lambda item: item[1]))

    @staticmethod
    def tag_player_positions(frames, net):
        players_by_tag = {}
        for frame in frames:
            players = frame.players()
            for player in players:
                if player.tag not in players_by_tag.keys():
                    players_by_tag[player.tag] = []
                players_by_tag[player.tag].append(player)

        def get_mean(tag):
            players = players_by_tag[tag]
            foots = [p.get_foot() for p in players]
            return mean(foots)
        def get_lowest(tag):
            players = players_by_tag[tag]
            foots = [p.get_foot() for p in players]
            return max(foots)
        def get_highest(tag):
            players = players_by_tag[tag]
            foots = [p.get_foot() for p in players]
            return min(foots)

        players_by_tag_dist = {}
        for tag, players in players_by_tag.items():
            players_by_tag_dist[tag] = get_mean(tag)

        tagged = []

        def get_dist(tag):
            return players_by_tag_dist[tag]

        def tag_all_players(tag, position):
            if tag not in tagged:
                for player in players_by_tag[tag]:
                    player.position = position
                    tagged.append(tag)


        for frame in frames:
            if frame.frame_number%1000 == 0:
                print(f"Tagging positions: {frame.frame_number}/{len(frames)}", end='\r')
            players = frame.players()


            players = sorted(players, key=lambda p: players_by_tag_dist[p.tag])

            if len(players) == 4:
                for p in players[:2]:
                    tag_all_players(p.tag,Position.TOP)
                for p in players[2:]:
                    tag_all_players(p.tag,Position.BOTTOM)

            else:
                top = []
                bottom = []

                for p in players:
                    if get_dist(p.tag) < net.get_foot() and len(top) <= 2:
                        tag_all_players(p.tag,Position.TOP)
                        top.append(p)
                    else:
                        tag_all_players(p.tag,Position.BOTTOM)
                        bottom.append(p)

        fps = 0
        for frame in frames:
            for position in [Position.TOP, Position.BOTTOM]:
                players = sorted(frame.players(position), key=lambda p: p.conf)

                while len(players) > 2:
                    fp = players[0]
                    frame.objects.remove(fp)
                    players.remove(fp)
                    fps +=1
        print(f"Removed {fps} falses positives")











    @staticmethod
    def tag_player_positions_old(frames, net):
        for frame in frames:
            sorted_players = sorted(frame.players(), key=lambda p: p.get_foot())

            if len(sorted_players) == 4:
                for p in sorted_players[:2]:
                    p.position = Position.BOTTOM
                for p in sorted_players[2:]:
                    p.position = Position.TOP

            else:
                top = []
                bottom = []

                for p in sorted_players:
                    if p.get_foot() < net.get_foot() and len(top) <= 2:
                        p.position = Position.TOP
                        top.append(p)
                    else:
                        p.position = Position.BOTTOM
                        bottom.append(p)

        # Check uniformity
        players_by_tag = {}
        for frame in frames:

            players = frame.players()
            for player in players:
                if player.tag not in players_by_tag.keys():
                    players_by_tag[player.tag] = []
                players_by_tag[player.tag].append(player)


        for tag, playerlist in players_by_tag.items():
            pos = [p.position for p in playerlist]

