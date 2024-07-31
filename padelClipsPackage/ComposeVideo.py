import os
import subprocess
import re
import sys
import json
import subprocess

from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.fx.fadein import fadein
from moviepy.video.io.VideoFileClip import VideoFileClip

from padelClipsPackage import Game

from pptx import Presentation

from padelClipsPackage.Point import Shot
from padelClipsPackage.aux import extract_frame_from_video, crop_and_save_image


class ComposeVideo:
    def __init__(self, game: Game, slides_dir, video_path):
        self.game = game
        self.slides_dir = slides_dir
        self.video_path = video_path

        self.cover = os.path.join(slides_dir, "cover.png")
        self.game_resume = os.path.join(slides_dir, "game_resume.png")
        self.top3player = os.path.join(slides_dir, "top3player.png")

        self.game_stats = os.path.join(slides_dir, "game_stats.pptx")
        self.player_stats = os.path.join(slides_dir, "player_stats.pptx")
        self.player_stats_png = {}
        self.player_images = {}
        self.get_player_images()

        self.cook_pptx()
        self.compose_with_videos()

    def compose_with_videos(self):
        #self.compose_game_resume()
        #for player in self.game.get_players():
        #    self.compose_top3_by_player(player.tag)

        self.create_video_sequence()

    def compose_top3_by_player(self, tag):
        points = self.game.gameStats.top_x_points_more_shots_by_player(3, tag)
        points_json = points_to_json(points)
        json_points_to_video(points_json, self.video_path, os.path.join(self.slides_dir, "top3_" + tag + ".mp4"))

    def compose_game_resume(self):
        points = self.game.gameStats.top_x_longest_points(20)
        points_json = points_to_json(points)
        json_points_to_video(points_json, self.video_path, os.path.join(self.slides_dir, "resume.mp4"))

    def cook_pptx(self):
        self.cook_pptx_global()
        for player in self.game.get_players():
            self.cook_pptx_player(player.tag)

    def cook_pptx_player(self, tag):
        player_stats_pptx = Presentation(self.player_stats)

        shots = self.game.gameStats.player_shot_number(tag)
        self.replace_placeholder_text(player_stats_pptx, "{{ shots }}", str(shots))

        meters_ran = int(self.game.gameStats.meters_ran(tag))
        self.replace_placeholder_text(player_stats_pptx, "{{ meters_run }}", str(meters_ran) + "m")

        globes = len(self.game.gameStats.player_shots(tag, Shot.Category.GLOBE))
        self.replace_placeholder_text(player_stats_pptx, "{{ globes }}", str(globes) + "m")

        self.replace_placeholder_text(player_stats_pptx, "{{ player }}", "Player " + tag )

        self.replace_image_in_slide(player_stats_pptx, self.player_images[tag])

        player_stats_pptx.save(os.path.join(self.slides_dir,'player_' + tag + '_cooked.pptx'))
        self.player_stats_png[tag] = os.path.join(self.slides_dir, 'player_' + tag + '_cooked.png')
        self.convert_pptx_to_png(os.path.join(self.slides_dir, 'player_' + tag + '_cooked.pptx'), self.slides_dir)


    def cook_pptx_global(self):
        game_stats_pptx = Presentation(self.game_stats)

        points = self.game.gameStats.total_points()
        self.replace_placeholder_text(game_stats_pptx, "{{ points }}", str(points))

        shots = self.game.gameStats.total_shots()
        self.replace_placeholder_text(game_stats_pptx, "{{ shots }}", str(shots))

        meters_ran = int(self.game.gameStats.overall_meters_ran())
        self.replace_placeholder_text(game_stats_pptx, "{{ meters_run }}", str(meters_ran) + "m")

        avg_shots = int(self.game.gameStats.average_shots_per_point())
        self.replace_placeholder_text(game_stats_pptx, "{{ avg_shots_point }}", str(avg_shots))

        game_stats_pptx.save(os.path.join(self.slides_dir,'game_stats_cooked.pptx'))
        self.game_stats_png = os.path.join(self.slides_dir, 'game_stats_cooked.png')
        self.convert_pptx_to_png(os.path.join(self.slides_dir, 'game_stats_cooked.pptx'), self.slides_dir)





    def replace_placeholder_text(self, prs, placeholder, replacement):
        # Iterate through each slide
        for slide in prs.slides:
            # Iterate through each shape within the slide
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                text_frame = shape.text_frame
                # Iterate through each paragraph in the text frame
                for paragraph in text_frame.paragraphs:
                    for run in paragraph.runs:
                        if placeholder in run.text:
                            # Replace the placeholder with the replacement text
                            run.text = run.text.replace(placeholder, replacement)

    def get_player_images(self):
        players = self.game.get_players()
        fn = players[0].frame_number
        frame_path = os.path.join(self.slides_dir, "frame_players.png")
        extract_frame_from_video(self.video_path, fn, frame_path)
        for tplayer in self.game.get_players():
            player = tplayer.player_object
            player_path = os.path.join(self.slides_dir,player.tag + ".png")
            crop_and_save_image(frame_path, player_path, player.x, player.y, player.height,
                                player.width)
            self.player_images[player.tag] = player_path


    def replace_image_in_slide(self, prs, image_path, old_image_index=0):
        """
        Replace an image in a PowerPoint slide.

        Args:
        pptx_path (str): Path to the PowerPoint file.
        slide_index (int): Index of the slide with the image (0-based).
        image_path (str): Path to the new image to insert.
        old_image_index (int): Index of the image to replace within the selected slide (0-based).
        """

        # Access the specific slide
        slide = prs.slides[0]

        # Find the image shape to replace
        image_shapes = [shape for shape in slide.shapes if shape.shape_type == 13]
        if not image_shapes or len(image_shapes) <= old_image_index:
            raise Exception("Image shape not found or index out of range.")

        # Get the target image shape based on the index
        old_image_shape = image_shapes[old_image_index]

        # Get size and position of the old image
        left = old_image_shape.left
        top = old_image_shape.top
        width = old_image_shape.width
        height = old_image_shape.height

        # Remove the old image
        sp = old_image_shape._element
        sp.getparent().remove(sp)

        # Add the new image with the same size and position
        slide.shapes.add_picture(image_path, left, top, width, height)



    def convert_pptx_to_png(self, pptx_path, output_dir):
        """
        Converts a PowerPoint PPTX file to PNG images using LibreOffice.

        Args:
        pptx_path (str): The path to the PowerPoint file.
        output_dir (str): The directory where the PNG images will be saved.
        """
        try:
            # Command to convert PPTX to PNG using LibreOffice
            cmd = [
                'libreoffice', '--headless', '--convert-to', 'png',
                '--outdir', output_dir, pptx_path
            ]

            # Run the command
            subprocess.run(cmd, check=True)
            print("Conversion successful, files saved to:", output_dir)
        except subprocess.CalledProcessError as e:
            print("An error occurred while converting the file:", e)
        except Exception as e:
            print("An error occurred:", e)

    def create_video_sequence(self, image_duration=5):
        # Define the file names of images and videos
        files = [
            "cover.png", "game_stats_cooked.png", "player_A_cooked.png",
            "top3player.png", "top3_A.mp4", "player_B_cooked.png",
            "top3player.png", "top3_B.mp4", "player_C_cooked.png",
            "top3player.png", "top3_C.mp4", "player_D_cooked.png",
            "top3player.png", "top3_D.mp4", "game_resume.png", "resume.mp4"
        ]

        # Prepare the list of clips
        clips = []

        # Loop through the files and create corresponding clips
        for filename in files:
            filepath = os.path.join(self.slides_dir, filename)
            if filename.endswith('.png'):
                # Create a clip from an image
                clip = ImageClip(filepath, duration=image_duration)
                clip = clip.set_duration(image_duration).resize(newsize=(1920, 1080))  # Resize to ensure uniformity
                clips.append(clip)
            elif filename.endswith('.mp4'):
                # Load a video file clip
                clip = VideoFileClip(filepath)
                clips.append(clip)

        # Concatenate all clips
        final_clip = concatenate_videoclips(clips, method="compose")

        # Write the result to a file
        final_clip.write_videofile("final_output.mp4", codec="libx264", fps=30)


def shots_to_json(shots, export=False, filename=""):
    json_shots = {}
    for i, shot in enumerate(shots):
        json_shots[i+1] = [shot.first_frame(), shot.last_frame()]

    if export:
        with open(filename, 'w') as f:
            json.dump(json_shots, f, indent=4)

    return json_shots


def points_to_json(points, export=False, filename=""):
    json_points = {}
    for i, point in enumerate(points):
        json_points[i+1] = [point.first_frame(), point.last_frame()]

    if export:
        with open(filename, 'w') as f:
            json.dump(json_points, f, indent=4)

    return json_points


def extract_clip(input_path, start_frame, end_frame, output_path, key, fps=30):
    """Extract clips and add text overlay using ffmpeg."""
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    drawtext = "drawtext=text='" + str(key) + "':x=w-tw-10:y=h-th-10:fontsize=720:fontcolor=white"
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_time), '-t', str(duration),  # Efficient seeking
        '-i', input_path,
        '-vf', drawtext,  # Text overlay
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'aac',  # Audio codec
        output_path
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Error handling
    if process.returncode != 0:
        print("Error occurred while executing FFmpeg with clip " + str(key) + ":")
        print(process.stderr)
        raise RuntimeError("FFmpeg failed with an error. See output above for more details.")

    print("FFmpeg process completed successfully with clip " + str(key))
def merge_clips(clips, output_path):
    """Merge video clips into a single video using ffmpeg."""
    with open('filelist.txt', 'w') as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")
    command = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', 'filelist.txt', '-c:v', 'libx264', '-c:a', 'aac', output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove('filelist.txt')


def concatenate_clips_with_transition(file_names, output_filename, transition_duration=1):
    # Convert transition_duration to float to avoid type issues
    transition_duration = float(transition_duration)

    # List to hold video clips with transitions
    clips_with_transitions = []

    # Load clips and apply fade in and fade out transitions
    for filename in file_names:
        clip = VideoFileClip(filename)
        clip = fadein(clip, transition_duration).fadeout(transition_duration)
        clips_with_transitions.append(clip)

    # Concatenate clips with transitions
    final_clip = concatenate_videoclips(clips_with_transitions, method="compose")

    # Output file
    final_clip.write_videofile(output_filename, codec="libx264", fps=24)

    return output_filename


def json_points_to_video(points_json, input_video_path, output_video_path, from_path = None, margin=0):

    # Load the JSON data from a file
    if from_path is not None:
        with open(from_path, 'r') as file:
            points = json.load(file)
    else:
        points = points_json

    temp_clips = []

    for key, frames in points.items():
        print("Extracting point " + str(key), end='\r')
        temp_clip_path = f"//making/temp_clip_{key}.mp4"
        start = frames[0] - margin
        if start < 0:
            start = 0
        end = frames[1] + margin


        extract_clip(input_video_path, start, end, temp_clip_path, key)
        temp_clips.append(temp_clip_path)

    print("Merging clips...")
    concatenate_clips_with_transition(temp_clips, output_video_path)

    # Optionally, clean up temporary clips
    for clip in temp_clips:
        os.remove(clip)
        pass

