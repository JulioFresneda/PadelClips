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


    def get_foot(self):
        return self.y + self.height/2

    def set_tag(self, tag):
        self.tag = tag

    def __str__(self):
        print("Object " + self.class_label_name)

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Object({self.class_label_name})"

    def get_position(self):
        return (self.x, self.y, self.width, self.height)


class Net(Object):
    def __init__(self, class_label, x_center, y_center, width, height, conf, tag=None):
        # Initialize the base class (Object)
        super().__init__(class_label, x_center, y_center, width, height, conf)

    def get_position(self):
        return self.y + self.height/2

    def __str__(self):
        print("Net")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Net"

class Player(Object):
    def __init__(self, class_label, x_center, y_center, width, height, conf, tag=None):
        # Initialize the base class (Object)
        self.position = None
        self.new_tag = None
        super().__init__(class_label, x_center, y_center, width, height, conf, tag)



    def __str__(self):
        print("Player " + self.tag)

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Player({self.tag})"

    def __copy__(self):
        np = Player(self.class_label, self.x, self.y, self.width, self.height, self.conf, self.tag)
        np.position = self.position
        return np

    def copy(self):
        return self.__copy__()





class PlayerTemplate:
    def __init__(self, tag, template_features, frame_number, object):
        self.tag = tag
        self.template_features = template_features
        self.frame_number = frame_number
        self.player_object = object
        self.position = self.player_object.position


    def __str__(self):
        print("Player " + str(self.tag))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Player({self.tag})"

    @staticmethod
    def features_distance(features_a, features_b):
        dist = cosine(features_a, features_b)
        return dist


