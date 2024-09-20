import math


class PositionInFrame:
    def __init__(self, x, y, frame_number):
        self.x = x
        self.y = y
        self.frame_number = frame_number

    @staticmethod
    def vx(p1, p2):
        return (p2.x - p1.x) / (p2.frame_number - p1.frame_number)

    @staticmethod
    def vy(p1, p2):
        return (p2.y - p1.y) / (p2.frame_number - p1.frame_number)

    @staticmethod
    def speed(p1, p2):
        vx = PositionInFrame.vx(p1, p2)
        vy = PositionInFrame.vy(p1, p2)
        return math.sqrt(vx ** 2 + vy ** 2)

    @staticmethod
    def speed_list(pifs, scale=10):
        speeds = []
        for p1, p2 in zip(pifs[:len(pifs)-1], pifs[1:]):
            speeds.append(PositionInFrame.speed(p1, p2)*scale)

        return sum(speeds)/len(speeds)


    @staticmethod
    def ax(p1, p2, p3):
        vx1 = PositionInFrame.vx(p1, p2)
        vx2 = PositionInFrame.vx(p2, p3)
        return (vx2 - vx1) / (p3.frame_number - p2.frame_number)

    @staticmethod
    def ay(p1, p2, p3):
        vy1 = PositionInFrame.vy(p1, p2)
        vy2 = PositionInFrame.vy(p2, p3)
        return (vy2 - vy1) / (p3.frame_number - p2.frame_number)

    @staticmethod
    def acc(p1, p2, p3):
        ax = PositionInFrame.ax(p1, p2, p3)
        ay = PositionInFrame.ay(p1, p2, p3)
        return math.sqrt(ax ** 2 + ay ** 2)

    @staticmethod
    def angle(p1, p2):
        vx = PositionInFrame.vx(p1, p2)
        vy = PositionInFrame.vy(p1, p2)
        return math.atan2(vy, vx)

    @staticmethod
    def jx(p1, p2, p3, p4):
        ax1 = PositionInFrame.ax(p1, p2, p3)
        ax2 = PositionInFrame.ax(p2, p3, p4)
        return (ax2 - ax1) / (p4.frame_number - p3.frame_number)

    @staticmethod
    def jy(p1, p2, p3, p4):
        ay1 = PositionInFrame.ay(p1, p2, p3)
        ay2 = PositionInFrame.ay(p2, p3, p4)
        return (ay2 - ay1) / (p4.frame_number - p3.frame_number)

    @staticmethod
    def jerk(p1, p2, p3, p4):
        jx = PositionInFrame.jx(p1, p2, p3, p4)
        jy = PositionInFrame.jy(p1, p2, p3, p4)
        return math.sqrt(jx ** 2 + jy ** 2)

    @staticmethod
    def ti(p1, p2):
        return p2.frame_number - p1.frame_number


    @staticmethod
    def velocity(pif_a, pif_b, scale = 1):
        distance = PositionInFrame.distance_to(pif_a, pif_b)
        time = abs(pif_b.frame_number - pif_a.frame_number)

        return distance / time * scale

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
