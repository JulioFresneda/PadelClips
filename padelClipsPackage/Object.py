from enum import Enum

from scipy.spatial.distance import cosine


class Label(Enum):
    BALL = 0
    PLAYER = 1
    NET = 2

class PlayerPosition(Enum):
    OVER_RIGHT = 0
    OVER_LEFT = 1
    UNDER_RIGHT = 2
    UNDER_LEFT = 3


class Object:


    def __init__(self, class_label, x_center, y_center, width, height, conf, tag = None):
        self.class_label = class_label
        self.class_label_name = Label(class_label).name
        self.x = x_center
        self.y = y_center
        self.width = width
        self.height = height
        self.size = width*height
        self.conf = conf
        self.tag = tag



    def set_tag(self, tag):
        self.tag = tag

    def __str__(self):
        print("Object " + self.class_label_name)

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Object({self.class_label_name})"

    def get_position(self):
        return (self.x, self.y, self.width, self.height)




class Player(Object):
    def __init__(self, class_label, x_center, y_center, width, height, conf, tag=None):
        # Initialize the base class (Object)
        super().__init__(class_label, x_center, y_center, width, height, conf, tag)

    def tag_position(self, position: PlayerPosition):
        self.position = position


    @staticmethod
    def position_players(players):
        # Sort players by y coordinate in descending order (from top to bottom)
        if len(players) == 4:
            sorted_by_y = sorted(players, key=lambda player: player.y, reverse=False)

            # The top half of the list will be "Over", the bottom half will be "Under"
            top_players = sorted_by_y[:2]
            bottom_players = sorted_by_y[2:]

            # Sort top and bottom halves by x coordinate (from left to right)
            top_left, top_right = sorted(top_players, key=lambda player: player.x)
            bottom_left, bottom_right = sorted(bottom_players, key=lambda player: player.x)

            top_left.tag_position(PlayerPosition.OVER_LEFT)
            top_right.tag_position(PlayerPosition.OVER_RIGHT)
            bottom_left.tag_position(PlayerPosition.UNDER_LEFT)
            bottom_right.tag_position(PlayerPosition.UNDER_RIGHT)






class PlayerTemplate:
    def __init__(self, tag, template_features, frame_number, object):
        self.tag = tag
        self.template_features = template_features
        self.frame_number = frame_number
        self.player_object = object
        self.position_oocam = None

    def set_position_oocam(self, x, y):
        self.position_oocam = (x, y)

    def __str__(self):
        print("Player " + str(self.tag))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Player({self.tag})"

    @staticmethod
    def features_distance(features_a, features_b):
        dist = cosine(features_a, features_b)
        return dist


