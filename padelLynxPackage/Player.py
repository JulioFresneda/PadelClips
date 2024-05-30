import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine


class PlayerFeatures:
    resnet = models.resnet50(pretrained=True)
    resnet.eval()  # Set the model to evaluation mode

    # Remove the final classification layer to use ResNet as a feature extractor
    model = nn.Sequential(*list(resnet.children())[:-1])

    # Define a transformation to preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, frame_path, player_yolo):
        self.player_image, self.color_histogram, self.deep_features = self.gen_player_features(frame_path, player_yolo, False)

    def __str__(self):
        print("PlayerFeature object")

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"PlayerFeature()"
    def get_player_features(self):
        return self.player_image, self.color_histogram, self.deep_features

    def extract_color_histogram(self, bins=(8, 8, 8)):
        """
        Extracts a color histogram from the image.

        Parameters:
        image (numpy.ndarray): The input image from which to extract the color histogram.
        bins (tuple): The number of bins for the histogram in each color channel (default is 8 for each channel).

        Returns:
        numpy.ndarray: The normalized color histogram as a feature vector.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(self.player_image, cv2.COLOR_BGR2HSV)

        # Compute the histogram for the HSV image
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def extract_deep_features(self):
        """
        Extracts deep features from the image using a pre-trained CNN model.

        Parameters:
        image (numpy.ndarray): The input image from which to extract deep features.
        model (torch.nn.Module): The pre-trained CNN model for feature extraction.
        preprocess (torchvision.transforms.Compose): The transformation to preprocess the input image.

        Returns:
        numpy.ndarray: The deep features extracted from the image.
        """
        # Convert the image from a NumPy array to a PIL Image
        image = Image.fromarray(cv2.cvtColor(self.player_image, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        input_tensor = self.preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

        # Extract features using the CNN model
        with torch.no_grad():
            features = self.model(input_tensor)

        # Flatten the features to create a feature vector
        features = features.flatten().numpy()

        return features

    def gen_player_features(self, frame_path, player_yolo, debug=False):
        """
        Extracts features for each player in the given frame.

        Parameters:
        frame_path (str): The path to the frame image.
        players (list): A list of player bounding boxes in YOLO format (x_center, y_center, width, height), values between 0 and 1.

        Returns:
        dict: A dictionary where keys are player indices and values are tuples of (color_histogram, deep_features, player_image).
        """
        # Load the frame image
        frame = cv2.imread(frame_path)
        frame_height, frame_width = frame.shape[:2]

        player_features = {}

        x_center, y_center, width, height = player_yolo
        # Convert YOLO format to bounding box coordinates
        x_center = int(x_center * frame_width)
        y_center = int(y_center * frame_height)
        width = int(width * frame_width)
        height = int(height * frame_height)

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Crop the player image from the frame
        self.player_image = frame[y_min:y_max, x_min:x_max]

        if(debug):
            plt.imshow(cv2.cvtColor(self.player_image, cv2.COLOR_BGR2RGB))  # Convert color from BGR to RGB
            plt.title("Average Frame")
            plt.axis("off")  # Turn off axis numbers and ticks
            plt.show()

        color_histogram = self.extract_color_histogram()
        deep_features = self.extract_deep_features()

        return self.player_image, color_histogram, deep_features

    @staticmethod
    def get_distance(features_a, features_b):
        color_hist1, deep_feat1, img1 = features_a.color_histogram, features_a.deep_features, features_a.player_image
        color_hist2, deep_feat2, img2 = features_b.color_histogram, features_b.deep_features, features_b.player_image
        # Calculate the distance between the features
        color_dist = cosine(color_hist1, color_hist2)
        deep_dist = cosine(deep_feat1, deep_feat2)
        # total_dist = color_dist + deep_dist
        total_dist = deep_dist

        return total_dist




class Player:
    def __init__(self, tag, player_features: PlayerFeatures):
        self.tag = tag
        self.player_features = player_features

    def __str__(self):
        print("Player " + str(self.tag))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Player({self.tag})"


