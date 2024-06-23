from enum import Enum
import os, re
from padelLynxPackage.aux import *
import pandas as pd
from collections import defaultdict

class Label(Enum):
    BALL = 0
    PLAYER = 1
    NET = 2
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


class Frame:
    def __init__(self, frame_number, frame_path, yolo = []):
        self.frame_number = frame_number
        self.frame_path = frame_path
        self.objects = yolo


    def players(self):
        return [obj for obj in self.objects if obj.class_label == 1]

    def balls(self):
        return [obj for obj in self.objects if obj.class_label == 0]

    def update_player_tag(self, old, new):
        for obj in self.objects:
            if obj.tag == old:
                obj.tag = new


    @staticmethod
    def merge_frame_list(list_a, list_b):

        bigger = (list_a if len(list_a) > len(list_b) else list_b).copy()
        smaller = list_a if len(list_a) <= len(list_b) else list_b

        for i, small_frame in enumerate(smaller):
            bigger[i].objects += small_frame.objects

        return bigger

    @staticmethod
    def load_from_excel(ball_excel_path, players_excel_path, mapping = {'ball':{0:Label.BALL}, 'players':{0:Label.NET, 1:Label.PLAYER}}):
        df_ball = pd.read_excel(ball_excel_path)
        df_players = pd.read_excel(players_excel_path)


        frames = []

        frame_info = defaultdict(list)
        for _, row in df_ball.iterrows():
            frame_info[int(row['frame'])].append(Object(mapping['ball'][row['class']].value, row['x'], row['y'], row['w'], row['h'], row['conf']))

        for _, row in df_players.iterrows():
            frame_info[int(row['frame'])].append(Object(mapping['players'][row['class']].value, row['x'], row['y'], row['w'], row['h'], row['conf'], tag=row['id']))



        # Determine the range of frame numbers
        max_frame = int(max(frame_info.keys()))

        for frame_number in range(max_frame + 1):
            if frame_number%100 == 0:
                print("Loading frame " + str(frame_number) + "/" + str(max_frame), end='\r')
            if frame_number in frame_info:
                frame = Frame(frame_number, None)
                obj = []
                for object in frame_info[frame_number]:
                    obj.append(object)
                frame.objects = obj.copy()

                frames.append(frame)

            else:
                frames.append(Frame(frame_number, None))



        return frames




    @staticmethod
    def load_frames(yolo_path, frames_path = None, mapping = {0:Label.BALL}):

        yolo_files = {}
        frames = []
        frame_img_path = []

        number_pattern = re.compile(r'\d+')

        for filename in os.listdir(yolo_path):
            # Check if the file ends with .jpg

            #frame_name = filename[:-4] + ".jpg"
            # You can now use this new filename to create a text file or rename, etc.
            # For demonstration, let's just print the new filename
            if frames_path != None:
                #frames.append(os.path.join(frames_path, frame_name))
                pass
            else:
                frame_img_path.append(None)



            match = number_pattern.search(filename.split('_')[-1])
            if match:
                number = match.group()
            else:
                number = -1

            yolo_files[int(number)] = os.path.join(yolo_path, filename)



        frame_numbers = list(yolo_files.keys())
        max_frames = max(frame_numbers)

        for i in range(max_frames+1):
            if i in frame_numbers:
                frames.append(Frame(i, None, Frame.read_yolo_txt(yolo_files[i], mapping)))
            else:
                frames.append(Frame(i,None,[]))


        return frames

    @staticmethod
    def read_yolo_txt(file_path, mapping):
        """
        Reads a YOLO format .txt file and returns a list of detections.

        Each detection is a tuple: (class_label, x_center, y_center, width, height)

        Parameters:
            file_path (str): Path to the YOLO format .txt file.

        Returns:
            List[Tuple[int, float, float, float, float]]: List of detections.
        """
        detections = []
        try:

            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_label = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    conf = float(parts[5])
                    if conf > 0.7:
                        cl = mapping[class_label].value

                        detections.append(Object(cl, x_center, y_center, width, height, conf))
        except:
            pass
        return detections


    def add_object(self, object: Object):
        self.objects.append(object)



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

