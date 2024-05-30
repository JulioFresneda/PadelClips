from enum import Enum
import os, re
from padelLynxPackage.aux import *


class Label(Enum):
    BALL = 0
    PLAYER = 1
    RACKET = 2
class Object:


    def __init__(self, class_label, x_center, y_center, width, height, conf):
        self.class_label = class_label
        self.class_label_name = Label(class_label).name
        self.x = x_center
        self.y = y_center
        self.width = width
        self.height = height
        self.size = width*height
        self.conf = conf

    def set_tag(self, tag):
        self.tag = tag

    def __str__(self):
        print("Object " + self.class_label_name)

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Object({self.class_label_name})"

    def get_yolo(self):
        return (self.x, self.y, self.width, self.height)


class Frame:
    def __init__(self, frame_number, frame_path, yolo):
        self.frame_number = frame_number
        self.frame_path = frame_path
        self.objects = yolo

    def players(self):
        return [obj for obj in self.objects if obj.class_label == 1]


    @staticmethod
    def load_frames(frames_path, yolo_path):
        frames = []
        frames_n = []
        yolo = []
        number_pattern = re.compile(r'\d+')

        for filename in os.listdir(frames_path):
            # Check if the file ends with .jpg
            if filename.endswith(".jpg"):
                # Construct the new filename with .txt extension
                new_filename = "segment_" + filename[:-4] + ".txt"
                # You can now use this new filename to create a text file or rename, etc.
                # For demonstration, let's just print the new filename
                frames.append(os.path.join(frames_path, filename))
                yolo.append(os.path.join(yolo_path, new_filename))

                match = number_pattern.search(filename)
                if match:
                    number = match.group()
                else:
                    number = -1
                frames_n.append(int(filename[:-4]))

        frames_loaded = []

        for frame, fnumber, yolo_info in zip(frames, frames_n, yolo):
            frames_loaded.append(Frame(fnumber, frame, Frame.read_yolo_txt(yolo_info)))

        frames_loaded = sorted(frames_loaded, key=lambda x: x.frame_number)

        return frames_loaded

    @staticmethod
    def read_yolo_txt(file_path):
        """
        Reads a YOLO format .txt file and returns a list of detections.

        Each detection is a tuple: (class_label, x_center, y_center, width, height)

        Parameters:
            file_path (str): Path to the YOLO format .txt file.

        Returns:
            List[Tuple[int, float, float, float, float]]: List of detections.
        """
        detections = []

        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_label = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                conf = float(parts[5])
                if conf > 0.5:
                    detections.append(Object(class_label, x_center, y_center, width, height, conf))

        return detections



    def __str__(self):
        return "Frame " + str(self.frame_number)

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Frame({self.frame_number})"

    def second(self, fps):
        return self.frame_number / fps


class Vector:
    class Direction(Enum):
        UP = 1
        DOWN = 2

    def __init__(self, position_a: Object, second_a: float, position_b: Object, second_b: float):
        self.x = position_b.x - position_a.x
        self.y = position_b.x - position_a.x
        self.time = second_b - second_a

        self.vertical_direction = self.Direction.UP if self.y > 0 else self.Direction.DOWN

