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
    def __init__(self, tag = None):
        self.pifs = []
        self.static = True
        self.globe = False
        self.variance_to_dynamic = 0.0000025
        self.variance_distance = 5
        self.tag = tag






    def max_min(self):
        max = -1.0
        min = 2.0

        for pif in self.pifs:
            if pif.y > max:
                max = pif.y
            if pif.y < min:
                min = pif.y
        return max, min


    def check_static_with_variance(self, minimum = 1):
        if len(self.pifs) > minimum:
            # if len(self.track) < self.variance_distance:
            variance = PositionInFrame.calculate_variance(self.pifs)
            if variance > self.variance_to_dynamic:
                self.static = False
        else:
            self.static = False
        return self.static




    def end(self):
        if len(self.pifs) == 0:
            return -1
        return self.pifs[-1].frame_number

    def start(self):
        if len(self.pifs) == 0:
            return -1
        return self.pifs[0].frame_number

    def has_frame(self, frame_number):
        for pif in self.pifs:
            if pif.frame_number == frame_number:
                return True
        return False



    def last_pif(self):
        if len(self.pifs) == 0:
            return None
        return self.pifs[-1]

    def add_pif(self, pif, check_static=False):
        if len(self.pifs) == 0 or self.end() < pif.frame_number:
            self.pifs.append(pif)
            if check_static:
                self.check_static_with_variance()
            return True
        else:
            return False


    @staticmethod
    def join_tracks(tracks, tag=None):
        new_track = Track(tag)
        for track in tracks:
            for pif in track.pifs:
                new_track.add_pif(pif)
        return new_track

    @staticmethod
    def split_track(track, idx_list):

        subtrack_start = 0

        subtracks = []
        for i in range(len(idx_list)):
            subtrack = Track(tag=str(track.tag) + "_" + str(i))
            for pif in track.pifs:
                if pif.frame_number >= subtrack_start and pif.frame_number < idx_list[i][0]:
                    subtrack.add_pif(pif)
            subtracks.append(subtrack)
            subtrack_start = idx_list[i][1]

        subtrack = Track(tag=str(track.tag) + "_" + str(len(idx_list)))
        for pif in track.pifs:
            if pif.frame_number >= subtrack_start:
                subtrack.add_pif(pif)
        subtracks.append(subtrack)

        return subtracks











    def __len__(self):
        return len(self.pifs)
    def __str__(self):
        print("Track from " + str(self.start()) + " to " + str(self.end()) + " with " + str(len(self.pifs)) + " objects")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Track(from {str(self.start())} to {str(self.end())}, {str(len(self.pifs))} objects)"

