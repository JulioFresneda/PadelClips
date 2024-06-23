import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine











class Player:
    def __init__(self, tag, template_features):
        self.tag = tag
        self.template_features = template_features

    def __str__(self):
        print("Player " + str(self.tag))

    def __repr__(self):  # This makes it easier to see the result when printing the list
        return f"Player({self.tag})"

    @staticmethod
    def features_distance(features_a, features_b):
        dist = cosine(features_a, features_b)
        return dist


