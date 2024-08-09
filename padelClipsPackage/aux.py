import cv2
from IPython.display import Image  # for displaying images
import random
import shutil
import numpy as np

from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw
import os
from pykalman import KalmanFilter
from PIL import Image

def extract_frame_from_video(video_path, frame_number, output_image_path):
    """
    Extracts a frame from the video at the given frame number and saves it as an image.

    :param video_path: Path to the input video file
    :param frame_number: The frame number to extract
    :param output_image_path: Path to save the extracted frame image
    """
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    # Set the frame position
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = video_capture.read()

    if success:
        # Save the frame as an image file
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} extracted and saved to {output_image_path}")
    else:
        print(f"Error: Could not read frame {frame_number}")

    # Release the video capture object
    video_capture.release()
def crop_and_save_image(input_image_path, output_image_path, x, y, height, width):
    """
    Crops the image based on the given x, y (center), height, and width values (between 0 and 1) and saves the cropped image.

    :param input_image_path: Path to the input image
    :param output_image_path: Path to save the cropped image
    :param x: X coordinate (between 0 and 1) of the center of the crop
    :param y: Y coordinate (between 0 and 1) of the center of the crop
    :param height: Height (between 0 and 1) of the crop
    :param width: Width (between 0 and 1) of the crop
    """
    # Open the input image
    image = Image.open(input_image_path)
    image_width, image_height = image.size

    # Calculate the center in pixels
    center_x = x * image_width
    center_y = y * image_height

    # Calculate the crop box in pixels
    left = center_x - (width * image_width) / 2
    upper = center_y - (height * image_height) / 2
    right = center_x + (width * image_width) / 2
    lower = center_y + (height * image_height) / 2

    # Perform the crop
    cropped_image = image.crop((left, upper, right, lower))

    # Save the cropped image
    cropped_image.save(output_image_path)
def video_to_frames(video_path, output_folder, start = 0, limit = None, steps = 10, real_count=False):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Initialize frame count
    frame_count = 0
    exported_count = 0

    # Read and save frames
    while limit == None or frame_count <= limit:

        ret, frame = cap.read()
        if not ret:
            break

            # Save frame as image
        if real_count:
            name = frame_count
        else:
            name = exported_count

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06}.jpg")
        #frame_filename = os.path.join(output_folder, f"{frame_count+1}.jpg")
        if (frame_count % steps == 0):
            cv2.imwrite(frame_filename, frame)
            exported_count += 1
        if (frame_count % 100 == 0):
            print("Frame " + str(frame_count), end= '\r')
        frame_count += 1


    # Release the video capture object
    cap.release()

    print(f"Extracted {frame_count} frames from the video.")

# Example usage
#video_path = 'dataset/padel2/padel2.mp4'
#output_folder = 'dataset/padel2/images/'
#video_to_frames(video_path, output_folder)


def apply_kalman_filter(positions):
    initial_state = positions[0]
    observation_covariance = np.eye(2)  # Assuming small error in observation
    transition_covariance = np.eye(2) * 0.03  # Assuming players generally move slightly
    transition_matrix = np.eye(2)

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix,
    )

    kalman_positions, _ = kf.smooth(positions)
    return kalman_positions


def plot_bounding_box(image, annotation_list, class_id_to_name_mapping, savepath):

    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (transformed_annotations[:, 3] / 2)
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (transformed_annotations[:, 4] / 2)
    transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
    transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)))

        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])


    plt.imshow(np.array(image))
    plt.savefig(savepath)


def plot_random_frame(frames_path, save_path):

    # Get any random annotation file
    annotations = [os.path.join(frames_path, x) for x in os.listdir(frames_path) if x[-3:] == "txt"]
    annotation_file = random.choice(annotations)
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x] for x in annotation_list]

    # Get the corresponding image file
    image_file = annotation_file.replace("annotations", "images").replace("txt", "PNG")
    assert os.path.exists(image_file)

    # Load the image
    image = Image.open(image_file)

    # Plot the Bounding Box
    plot_bounding_box(image, annotation_list, {0:"ball", 1:"player"}, save_path)



def split_train_test_valid(folder_path):
    # Read images and annotations
    images = [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x[-3:] == "PNG" or x[-3:] == "jpg"]
    annotations = [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x[-3:] == "txt"]

    images.sort()
    annotations.sort()

    # Split the dataset into train-valid-test splits
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    os.mkdir(os.path.join(folder_path, "images"))

    os.mkdir(os.path.join(folder_path, "images/train"))
    os.mkdir(os.path.join(folder_path, "images/test"))
    os.mkdir(os.path.join(folder_path, "images/val"))

    os.mkdir(os.path.join(folder_path, "labels"))

    os.mkdir(os.path.join(folder_path, "labels/train"))
    os.mkdir(os.path.join(folder_path, "labels/test"))
    os.mkdir(os.path.join(folder_path, "labels/val"))

    move_files_to_folder(train_images, os.path.join(folder_path,'images/train'))
    move_files_to_folder(val_images, os.path.join(folder_path,'images/val/'))
    move_files_to_folder(test_images, os.path.join(folder_path,'images/test/'))
    move_files_to_folder(train_annotations, os.path.join(folder_path,'labels/train/'))
    move_files_to_folder(val_annotations, os.path.join(folder_path,'labels/val/'))
    move_files_to_folder(test_annotations, os.path.join(folder_path,'labels/test/'))

#Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders


def get_video_fps(video_path):
    """
    Get the frames per second (FPS) of the video.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        float: Frames per second (FPS) of the video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps





def read_yolo_files(folder_path):
    all_detections = []
    filenames = [f for f in os.listdir(folder_path)]

    # Sort filenames based on the frame number extracted from the filename
    filenames.sort(key=lambda f: int(re.search(r'\d+', f[len("segment_"):-4]).group()))


    for filename in filenames:
        frame_number = int(re.search(r'\d+', filename[len("segment_"):-4]).group())
        file_path = os.path.join(folder_path, filename)
        detections = read_yolo_txt(file_path)

        ball_position = None
        players_position = []

        for detection in detections:
            if detection.class_label == 0:
                ball_position = detection
            else:
                players_position.append(detection)

        frame = Frame(frame_number, ball_position, players_position)

        all_detections.append(frame)

    return all_detections


def cut_video(video, start, end, output):
    video = VideoFileClip(video)

    # Set the FPS (frames per second) of your video to correctly calculate timecodes
    fps = video.fps

    # Define your start and end frames
    start_frame = start  # replace x with your start frame number
    end_frame = end  # replace y with your end frame number

    # Calculate start and end times in seconds
    start_time = start_frame / fps
    end_time = end_frame / fps

    # Cut the video between the start and end times
    cut_video = video.subclip(start_time, end_time)

    # Write the result to a file
    cut_video.write_videofile(output, codec='libx264')


def frame_to_seconds(frame_number, fps):
    return (frame_number) / fps


# Format function for times on x-axis
def format_seconds(frame_number, fps):
    seconds = frame_to_seconds(frame_number, fps)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"