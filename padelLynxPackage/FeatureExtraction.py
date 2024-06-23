import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import os
import csv
class FeatureExtractor:
    def __init__(self):


        resnet = models.resnet50(pretrained=True)
        resnet.eval()  # Set the model to evaluation mode

        # Remove the final classification layer to use ResNet as a feature extractor
        self.model = nn.Sequential(*list(resnet.children())[:-1])

        # Define a transformation to preprocess the input image
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_deep_features(self, image_portion):
        """
        Extracts deep features from the image using a pre-trained CNN model.

        Parameters:
        image_portion (numpy.ndarray): The input image portion from which to extract deep features.

        Returns:
        numpy.ndarray: The deep features extracted from the image.
        """
        # Convert the image from a NumPy array to a PIL Image
        image = Image.fromarray(cv2.cvtColor(image_portion, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        input_tensor = self.preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

        # Extract features using the CNN model
        with torch.no_grad():
            features = self.model(input_tensor)

        # Flatten the features to create a feature vector
        features = features.flatten().numpy()

        return features


# Example Usage
def process_image(tensor, image, cls):
    feature_extractor = FeatureExtractor()
    features_list = []

    for i, ((cx, cy, width, height), is_useful) in enumerate(zip(tensor, cls)):
        if is_useful != 1:
            continue
        cx, cy, width, height = map(int, (cx, cy, width, height))

        # Calculate the top-left corner of the portion
        x = cx - width // 2
        y = cy - height // 2

        # Ensure coordinates are within the image boundaries
        x = max(0, x)
        y = max(0, y)
        width = min(image.shape[1] - x, width)
        height = min(image.shape[0] - y, height)

        # Debugging output
        #print(f"Processing portion {i}: x={x}, y={y}, width={width}, height={height}")

        # Extract the portion of the image
        image_portion = image[y:y + height, x:x + width]

        # Save the portion as a JPG file
        #portion_image_path = os.path.join(output_folder, f"portion_{i}.jpg")
        #cv2.imwrite(portion_image_path, image_portion)

        # Extract deep features from the portion
        features = feature_extractor.extract_deep_features(image_portion)
        features_list.append(features)



    return features_list