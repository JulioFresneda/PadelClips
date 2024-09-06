import json

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.python.keras.models import load_model
import tensorflow.keras

from padelClipsPackage.PositionInFrame import PositionInFrame
import pandas as pd
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.ndimage import uniform_filter1d

class PredictPointModel:
    model_path = '/home/juliofgx/PycharmProjects/PadelClips/model_set1.keras'
    def __init__(self, tracks=None, true_labels=None):
        self.true_labels = true_labels
        self.pifs = []
        for track in tracks:
            self.pifs += track.pifs

        self.pifs = sorted(self.pifs, key=lambda pif: pif.frame_number)




    @staticmethod
    def predict(pifs):
        loaded_model = tf.keras.models.load_model(PredictPointModel.model_path)

        sequence_length = 180
        df = PredictPointModel.pifs_to_df(pifs)

        # Replace inf and -inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optionally, replace NaN with a specific value (e.g., 0 or the mean of the column)
        df.fillna(0, inplace=True)  # or df.fillna(df.mean(), inplace=True)

        # Extract the features and label from your DataFrame
        features = df[['vx', 'vy', 'speed', 'ax', 'ay', 'acc', 'angle', 'jx', 'jy', 'jerk', 'ti']].values
        labels = df['label'].values
        frame_numbers = df['fn'].values  # Extract the frame numbers

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X = []
        y = []
        frame_seqs = []

        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(labels[i + sequence_length - 1])  # Label is based on the last time step in the sequence
            frame_seqs.append(frame_numbers[i + sequence_length - 1])  # Track the corresponding frame number

        X = np.array(X)

        predictions = loaded_model.predict(X)

        # Convert predictions to binary labels
        predictions_binary = (predictions > 0.5).astype(int)
        xd = predictions_binary

        # Apply smoothing
        predictions_binary = uniform_filter1d(predictions_binary.flatten(), size=5, mode='nearest')
        predictions_binary = (predictions_binary > 0.5).astype(int)

        playtime_segments = []
        in_playtime = False
        start_frame = None

        for i in range(len(predictions_binary)):
            if predictions_binary[i] == 1 and not in_playtime:
                # Start of a new playtime segment
                in_playtime = True
                start_frame = frame_seqs[i]  # Use the frame number from the sequence
            elif predictions_binary[i] == 0 and in_playtime:
                # End of the current playtime segment
                in_playtime = False
                end_frame = frame_seqs[i]  # Use the frame number from the sequence
                playtime_segments.append((start_frame, end_frame))

        # Handle case where playtime continues until the last frame
        if in_playtime:
            end_frame = frame_seqs[-1]
            playtime_segments.append((start_frame, end_frame))

        # Display the start and end frames of each playtime segment
        playtime_segments = PredictPointModel.merge_segments(playtime_segments)

        # Ensure all start_frame and end_frame values are standard Python integers
        playtime_segments_dict = [{'start_frame': int(start), 'end_frame': int(end)} for start, end in
                                  playtime_segments]

        # Define the file path where you want to save the JSON file
        file_path = '/home/juliofgx/PycharmProjects/PadelClips/playtime_segments.json'

        # Save the list of dictionaries to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(playtime_segments_dict, json_file, indent=4)

        print(f"Playtime segments saved to {file_path}")

        return playtime_segments_dict

    @staticmethod
    def merge_segments(segments, threshold=60):
        # Sort segments by the start frame
        segments.sort()

        # Initialize the list to hold the merged segments
        merged_segments = []

        # Start with the first segment
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            # Check if the segments are closer than the threshold
            if start - current_end <= threshold:
                # Merge the segments
                current_end = max(current_end, end)
            else:
                # Append the current segment to the list
                merged_segments.append((current_start, current_end))
                # Start a new segment
                current_start, current_end = start, end

        # Don't forget to add the last segment
        merged_segments.append((current_start, current_end))

        return merged_segments

    @staticmethod
    def learn(features_path):
        sequence_length = 180
        df = pd.read_excel(features_path)

        # Replace inf and -inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Optionally, replace NaN with a specific value (e.g., 0 or the mean of the column)
        df.fillna(0, inplace=True)  # or df.fillna(df.mean(), inplace=True)


        # Extract the features and label from your DataFrame
        features = df[['vx', 'vy', 'speed', 'ax', 'ay', 'acc', 'angle', 'jx', 'jy', 'jerk', 'ti']].values
        labels = df['label'].values

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X = []
        y = []

        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(labels[i + sequence_length - 1])  # Label is based on the last time step in the sequence

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.4)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.2f}")

        model.save('/home/juliofgx/PycharmProjects/PadelClips/model_set1.keras')

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Convert predictions to binary labels
        predictions = (predictions > 0.5).astype(int)

        # Generate the confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # Optionally, plot the confusion matrix using a heatmap
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    @staticmethod
    def pifs_to_df(pifs):
        # Preallocate the DataFrame
        columns = ['fn', 'vx', 'vy', 'speed', 'ax', 'ay', 'acc', 'angle', 'jx', 'jy', 'jerk', 'ti', 'label']
        df = pd.DataFrame(index=range(len(pifs)), columns=columns)

        # Fill the DataFrame with features
        for i in range(len(pifs)):
            print(f"PIFs to df: {i}/{len(pifs)}", end='\r')
            df.at[i, 'fn'] = pifs[i].frame_number
            if i >= 1:
                df.at[i, 'vx'] = PositionInFrame.vx(pifs[i - 1], pifs[i])
                df.at[i, 'vy'] = PositionInFrame.vy(pifs[i - 1], pifs[i])
                df.at[i, 'speed'] = PositionInFrame.speed(pifs[i - 1], pifs[i])
                df.at[i, 'angle'] = PositionInFrame.angle(pifs[i - 1], pifs[i])
                df.at[i, 'ti'] = PositionInFrame.ti(pifs[i - 1], pifs[i])
            if i >= 2:
                df.at[i, 'ax'] = PositionInFrame.ax(pifs[i - 2], pifs[i - 1], pifs[i])
                df.at[i, 'ay'] = PositionInFrame.ay(pifs[i - 2], pifs[i - 1], pifs[i])
                df.at[i, 'acc'] = PositionInFrame.acc(pifs[i - 2], pifs[i - 1], pifs[i])
            if i >= 3:
                df.at[i, 'jx'] = PositionInFrame.jx(pifs[i - 3], pifs[i - 2], pifs[i - 1], pifs[i])
                df.at[i, 'jy'] = PositionInFrame.jy(pifs[i - 3], pifs[i - 2], pifs[i - 1], pifs[i])
                df.at[i, 'jerk'] = PositionInFrame.jerk(pifs[i - 3], pifs[i - 2], pifs[i - 1], pifs[i])

        return df

    def initialize_df(self, output_path):
        columns = ['vx', 'vy', 'speed', 'ax', 'ay', 'acc', 'angle', 'jx', 'jy', 'jerk', 'ti', 'label']
        df = pd.DataFrame(columns=columns)
        pifs = self.pifs

        # Fill the DataFrame with features
        for i in range(len(pifs)):
            label_true = 0
            for tl in self.true_labels:
                if tl[0] <= pifs[i].frame_number <= tl[1]:
                    label_true = 1
            if i >= 3:
                row = {
                    'vx': PositionInFrame.vx(pifs[i - 1], pifs[i]),
                    'vy': PositionInFrame.vy(pifs[i - 1], pifs[i]),
                    'speed': PositionInFrame.speed(pifs[i - 1], pifs[i]),
                    'ax': PositionInFrame.ax(pifs[i - 2], pifs[i - 1], pifs[i]),
                    'ay': PositionInFrame.ay(pifs[i - 2], pifs[i - 1], pifs[i]),
                    'acc': PositionInFrame.acc(pifs[i - 2], pifs[i - 1], pifs[i]),
                    'angle': PositionInFrame.angle(pifs[i - 1], pifs[i]),
                    'jx': PositionInFrame.jx(pifs[i - 3], pifs[i - 2], pifs[i - 1], pifs[i]),
                    'jy': PositionInFrame.jy(pifs[i - 3], pifs[i - 2], pifs[i - 1], pifs[i]),
                    'jerk': PositionInFrame.jerk(pifs[i - 3], pifs[i - 2], pifs[i - 1], pifs[i]),
                    'ti': PositionInFrame.ti(pifs[i - 1], pifs[i])
                }
            elif i >= 2:
                row = {
                    'vx': PositionInFrame.vx(pifs[i - 1], pifs[i]),
                    'vy': PositionInFrame.vy(pifs[i - 1], pifs[i]),
                    'speed': PositionInFrame.speed(pifs[i - 1], pifs[i]),
                    'ax': np.nan,  # Not enough data to calculate acceleration
                    'ay': np.nan,
                    'acc': np.nan,
                    'angle': PositionInFrame.angle(pifs[i - 1], pifs[i]),
                    'jx': np.nan,  # Not enough data to calculate jerk
                    'jy': np.nan,
                    'jerk': np.nan,
                    'ti': PositionInFrame.ti(pifs[i - 1], pifs[i])
                }
            elif i >= 1:
                row = {
                    'vx': PositionInFrame.vx(pifs[i - 1], pifs[i]),
                    'vy': PositionInFrame.vy(pifs[i - 1], pifs[i]),
                    'speed': PositionInFrame.speed(pifs[i - 1], pifs[i]),
                    'ax': np.nan,
                    'ay': np.nan,
                    'acc': np.nan,
                    'angle': PositionInFrame.angle(pifs[i - 1], pifs[i]),
                    'jx': np.nan,
                    'jy': np.nan,
                    'jerk': np.nan,
                    'ti': PositionInFrame.ti(pifs[i - 1], pifs[i])
                }
            else:
                row = {
                    'vx': np.nan,
                    'vy': np.nan,
                    'speed': np.nan,
                    'ax': np.nan,
                    'ay': np.nan,
                    'acc': np.nan,
                    'angle': np.nan,
                    'jx': np.nan,
                    'jy': np.nan,
                    'jerk': np.nan,
                    'ti': np.nan
                }
            row['label'] = label_true
            row_df = pd.DataFrame([row])
            df = pd.concat([df, row_df], ignore_index=True)

        df.to_excel(output_path)



#PredictPointModel.learn('/home/juliofgx/PycharmProjects/PadelClips/padelClipsTraining/predict/1set_ft.xlsx')