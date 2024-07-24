import math
import matplotlib.pyplot as plt
class PositionInFrame:
    def __init__(self, x, y, frame_number):
        self.x = x
        self.y = y
        self.frame_number = frame_number

    def nearest(self, positions_in_frame):
        nearest = PositionInFrame(float('inf'), float('inf'), -1)
        for pif in positions_in_frame:
            if PositionInFrame.distance_to(self, pif) < PositionInFrame.distance_to(self, nearest):
                nearest = pif
        return nearest

    @staticmethod
    def distance_to(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def get_direction(a, b):
        if a.y > b.y:
            return "down"
        else:
            return "up"


    @staticmethod
    def calculate_max_min(objects):
        max = -1.1
        min = 1.1
        for obj in objects:
            if obj.y < min:
                min = obj.y
            if obj.y > max:
                max = obj.y
        return max, min

    @staticmethod
    def calculate_variance(objects, only_vertical=True):
        distances = []
        n = len(objects)
        for i in range(n):
            for j in range(i + 1, n):  # Avoid computing the same distance twice
                if not only_vertical:
                    distance = PositionInFrame.distance_to(objects[i], objects[j])
                else:
                    distance = abs(objects[i].y - objects[j].y)
                distances.append(distance)

        # Calculate the mean of the distances
        mean_distance = sum(distances) / len(distances)

        # Calculate the variance
        variance = sum((x - mean_distance) ** 2 for x in distances) / len(distances)
        return variance

    def __str__(self):
        print("PIF " + str(self.x) + ", " + str(self.y) + " in " + str(self.frame_number))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"PIF({str(self.x)},{str(self.y)}) in {str(self.frame_number)}"


class Track:
    def __init__(self):
        self.track = []
        self.static = True
        self.globe = False
        self.variance_to_dynamic = 0.000005
        self.variance_distance = 5
        self.quality_track = False


    def max_min(self):
        max = -1.0
        min = 2.0

        for pif in self.track:
            if pif.y > max:
                max = pif.y
            if pif.y < min:
                min = pif.y
        return max, min

    def not_crossed_net(self, net, tolerance):

        tail = self.track if len(self.track) <= tolerance else self.track[:-tolerance]

        pos = tail[0].y
        crossed = False
        for pif in tail:
            if pos < net.y - net.height/2 and pif.y > net.y + net.height/2:
                crossed = True
            elif pos > net.y + net.height/2 and pif.y < net.y - net.height/2:
                crossed = True
        return not crossed

    def position_in_net(self, net):
        pos = []
        for pif in self.track:
            if pif.y < net.y - net.height / 2:
                pos.append('over')
            elif pif.y > net.y + net.height / 2:
                pos.append('under')
            else:
                pos.append('middle')
        if len(pos) == 0:
            return None
        elif len(pos) == 1:
            return pos[0]
        else:
            return 'cross'


    def get_direction_changes(self):
        directions = []
        if len(self.track) == 1:
            return 0
        else:
            for i in range(1, len(self.track)):
                directions.append(PositionInFrame.get_direction(self.track[i-1], self.track[i]))
            changes = 0
            init = directions[0]
            for dir in directions:
                if dir != init:
                    changes += 1
                    init = dir
            return changes





    def get_pif(self, frame_number):
        pass
    def distance(self):
        dist = 0.0
        for i, pif in enumerate(self.track):
            if i > 0:
                dist += PositionInFrame.distance_to(self.track[i - 1], pif)
        return dist

    def density(self):
        if len(self.track) > 0:
            density = self.distance() / len(self.track)
            return density
        else:
            return 0



    def purge_static_subtracks(self):
        if len(self.track) > self.variance_distance:
            subtracks = [self.track[i:i + self.variance_distance] for i in
                         range(0, len(self.track) - self.variance_distance)]

            st_to_remove = []

            for subtrack in subtracks:
                variance = PositionInFrame.calculate_variance(subtrack)
                if variance < self.variance_to_dynamic:
                    st_to_remove.append(subtrack)

            st_list = []
            for st_tr in st_to_remove:
                for pif in st_tr:
                    if pif not in st_list:
                        st_list.append(pif)

            subtracks = self.split_and_remove(self.track.copy(), st_list)
            return subtracks
        return []

    def split_and_remove(self, items, to_remove):
        result = []  # This will hold all the sublists
        current_sublist = []  # This holds the current sublist being constructed

        for item in items:
            if item in to_remove:
                if current_sublist:  # Only add the sublist if it's not empty
                    result.append(current_sublist)
                    current_sublist = []  # Reset for the next sublist
            else:
                current_sublist.append(item)  # Add item to the current sublist

        if current_sublist:  # Add the last sublist if not empty
            result.append(current_sublist)

        return result

    def check_static(self):
        if len(self.track) > 1:
            # if len(self.track) < self.variance_distance:
            variance = PositionInFrame.calculate_variance(self.track)
            if variance > self.variance_to_dynamic:
                self.static = False
        return self.static


    def last_frame(self):
        if len(self.track) == 0:
            return None
        return self.track[-1].frame_number

    def first_frame(self):
        if len(self.track) == 0:
            return None
        return self.track[0].frame_number

    def has_frame(self, frame_number):
        for pif in self.track:
            if pif.frame_number == frame_number:
                return True
        return False



    def last_pif(self):
        if len(self.track) == 0:
            return None
        return self.track[-1]

    def add_pif(self, pif, check_static=False):
        if len(self.track) == 0 or self.last_frame() < pif.frame_number:
            self.track.append(pif)
            if check_static:
                self.check_static()
            self.check_quality()
            return True
        else:
            return False

    def check_quality(self):
        max, min = self.max_min()
        if abs(max-min) > 0.3 and len(self.track) > 3:
            self.quality_track = True

    def __str__(self):
        print("Track with" + str(len(self.track)) + " objects")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Track({str(len(self.track))} objects)"

