import io
import os
import subprocess
import re
import sys
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from spire.presentation import FileFormat
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from padelClipsPackage import Game

from bs4 import BeautifulSoup

from padelClipsPackage.Point import Shot
from padelClipsPackage.Shot import Category, Position
from padelClipsPackage.aux import extract_frame_from_video, crop_and_save_image

import base64
import ssl
import hashlib

from pptx import Presentation
from PIL import Image
import os
import base64

from pptx import Presentation
from pptx.util import Inches

import aspose.slides as slides


class ComposeVideo:
    def __init__(self, game: Game, making_path, resources_path, video_path, output_path):
        self.player_images = {}
        self.game = game
        self.gameStats = game.gameStats
        self.making_path = making_path
        self.resources_path = resources_path
        self.video_path = video_path
        self.output_path = output_path

        self.get_player_images()
        self.generate_graphs()

        for player in self.game.players:
            input_file = self.resources_path + '/player_1.html'
            self.generate_html(player, input_file)

            input_file = self.resources_path + '/player_2.html'
            self.generate_html(player, input_file)

        self.compose_video()


    def compose_video(self):
        points = self.game.gameStats.top_x_longest_points(10)
        self.make_clips(points, margin=60)



    def make_clips(self, points, margin):

        temp_clips = []

        for i in range(len(points), 0, -1):
            print("Extracting point " + str(i), end='\r')
            temp_clip_path = f"{self.making_path}/temp_clip_{i}.mp4"
            start = points[i - 1].start() - margin
            end = points[i - 1].end() + margin
            overlay_path = f"{self.resources_path}/points/{i}.mp4"
            if not os.path.exists(temp_clip_path):
                extract_clip(self.video_path, start, end, temp_clip_path, overlay_path)
            temp_clips.append(temp_clip_path)

        print("Merging clips...")

        image_filenames = []
        for player in self.game.players:
            image_filenames.append(f"{self.making_path}/player_1_{player.tag}.png")
            image_filenames.append(f"{self.making_path}/player_2_{player.tag}.png")

        self.concatenate_clips_with_transition(temp_clips, self.output_path, image_filenames)

        # Optionally, clean up temporary clips
        for clip in temp_clips:
            os.remove(clip)
            pass

    def concatenate_clips_with_transition(self, file_names, output_filename, image_filenames, transition_duration=1):
        # Convert transition_duration to float to avoid type issues
        transition_duration = float(transition_duration)

        # List to hold video clips with transitions
        clips_with_transitions = []

        # Load clips and apply fade in and fade out transitions
        start_filename = f"{self.resources_path}/start.mp4"
        clip = VideoFileClip(start_filename)
        clip = clip.resize(newsize=(3840, 2160))
        clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
        clips_with_transitions.append(clip)

        for image_filename in image_filenames:
            image_clip = ImageClip(image_filename, duration=5)  # 5 seconds duration for each image
            clips_with_transitions.append(image_clip)

        for filename in file_names:
            clip = VideoFileClip(filename)
            clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
            clips_with_transitions.append(clip)



        # Concatenate clips with transitions
        final_clip = concatenate_videoclips(clips_with_transitions, method="compose")

        # Output file
        final_clip.write_videofile(output_filename, codec="libx264", fps=60)

        return output_filename

    def generate_html(self, player, input_file):
         # Replace with your actual file path

        with open(input_file, 'r', encoding='utf-8') as file:
            html_content = file.read()


        custom_names = {'A':'Macho alfalfa', 'B':"El puto", 'C':"Primo de Raul", 'D':"Heterocurioso"}


        # Replace the placeholders with actual player data
        html_content = html_content.replace('{{ tag }}', player.tag)
        html_content = html_content.replace('{{ keyname }}', custom_names[player.tag])
        html_content = html_content.replace('player.png', self.making_path + f'/{player.tag}.png')

        position = "Abajo" if player.player_object.position is Position.BOTTOM else "Arriba"
        html_content = html_content.replace('{{ position }}', position)
        teammate = None
        for p in self.game.players:
         if p.tag != player.tag and p.player_object.position is player.player_object.position:
             teammate = p.tag
        html_content = html_content.replace('{{ tag_teammate }}', teammate)

        if '1.html' in input_file:
            self.generate_html_1(html_content, player)
        else:
            self.generate_html_2(html_content, player)

    def generate_html_2(self, html_content, player):
        html_content = html_content.replace('{{ mrun }}', str(int(self.game.gameStats.meters_ran(player.tag))) + ' metros')

        output_file = self.making_path + '/player_2_' + player.tag + '.html'  # Replace with your desired output file path

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)

        png_output = self.making_path + '/player_2_' + player.tag + '.png'
        self.html_to_png(output_file, png_output)

    def generate_html_1(self, html_content, player):
        html_content = html_content.replace('{{ nshots }}', str(self.game.gameStats.player_shot_number(player.tag)))

        # Replace the image placeholders with the corresponding player-specific images

        html_content = html_content.replace('heatmap.png', self.making_path + f'/heatmap_{player.tag}.png')
        html_content = html_content.replace('shots_chart.png', self.making_path + f'/shots_chart_{player.tag}.png')

        # Write the modified HTML content back to a file (optional)
        output_file = self.making_path + '/player_1_' + player.tag + '.html'  # Replace with your desired output file path

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)

        png_output = self.making_path + '/player_1_' + player.tag + '.png'
        self.html_to_png(output_file, png_output)

    def html_to_png(self, html_file_path, output_png_path, resolution="3840x2160"):
        """
        Convert an HTML file to a PNG image using wkhtmltoimage with 4K resolution.

        :param html_file_path: Path to the input HTML file.
        :param output_png_path: Path to the output PNG file.
        :param resolution: The resolution of the output image. Default is "3840x2160" (4K).
        """
        try:
            # Call wkhtmltoimage to convert HTML to PNG
            subprocess.run(
                [
                    'wkhtmltoimage',
                    '--enable-local-file-access',
                    '--width', resolution.split('x')[0],
                    '--height', resolution.split('x')[1],
                    html_file_path,
                    output_png_path
                ],
                check=True
            )
            print(f"Successfully converted {html_file_path} to {output_png_path} in 4K resolution.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting HTML to PNG: {e}")

    def generate_graphs(self):
        for player in self.game.players:
            stats = self.gameStats.categorize_player_shots()
            self.generate_shots_graph(stats[player.tag], player.tag)

            pos_data = self.game.frames_controller.get_player_positions(player.tag)
            pos_data = self.scale_position(pos_data, player.player_object.position)
            self.generate_heatmap(pos_data, player.tag)

    def scale_position(self, pos_data, player_position):

        up = self.game.players_boundaries_vertical[Position.TOP]
        bottom = self.game.players_boundaries_vertical[Position.BOTTOM]
        left = self.game.players_boundaries_horizontal[player_position]['left']
        right = self.game.players_boundaries_horizontal[player_position]['right']

        def scale_position(data, a, b):
            return abs((data - a) / (b - a))

        scaled = []
        for pos in pos_data:
            x = scale_position(pos[0], left, right)
            y = scale_position(2 * (1 - pos[1]), up, bottom)
            scaled.append((x, y))
        return scaled

    def generate_heatmap(self, data, tag):

        # Extract x and y coordinates
        x_values, y_values = zip(*data)

        # Create a 2D histogram (heatmap) from the x and y values
        heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=50, range=[[0, 1], [0, 1]])

        # Create the figure with a 1:2 aspect ratio (taller than wide) without the legend
        fig, ax = plt.subplots(figsize=(5, 10))

        # Plot the heatmap
        cax = ax.imshow(heatmap.T, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm', aspect='auto')

        # Add continuous horizontal and vertical lines in the middle with even wider lines
        ax.axhline(y=0.5, color='white', linestyle='-', linewidth=8)
        ax.axvline(x=0.5, color='white', linestyle='-', linewidth=8)

        # Remove labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set the coordinate range from 0 to 1 for both axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.savefig(self.making_path + '/heatmap_' + tag + '.png', dpi=300, bbox_inches='tight')

    def generate_shots_graph(self, stats, tag):

        # Data for the spider chart
        labels = ["DERECHA", "SMASH", "IZQUIERDA", "VOLEA BAJA"]
        stats = [stats['right'] / 10, stats[Category.SMASH], stats['left'] / 10, stats[Category.LOW_VOLLEY]]
        for i in range(4):
            labels[i] = labels[i] + " (" + str(int(10 * stats[i])) + ")"

        # Number of variables we're plotting.
        num_vars = len(labels)

        # Split the circle into even parts and save the angles
        # so we know where to put each axis.
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop"
        # and append the start to the end.
        stats += stats[:1]
        angles += angles[:1]

        # Create the figure and polar subplot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Draw the outline of our data.
        ax.fill(angles, stats, color='#174779', alpha=0.25)
        ax.plot(angles, stats, color='#174779', linewidth=2)

        # Fill in the areas with the specified color
        ax.fill(angles, stats, color='#7FB6FF', alpha=0.6)

        # Draw one filled circle per point
        ax.scatter(angles, stats, color='#F8FBFF', s=100)

        # Fix the labels
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='#174779', size=20)

        # Set the background color
        ax.set_facecolor('#F8FBFF')

        fig.savefig(self.making_path + '/shots_chart_' + tag + '.png', dpi=300, bbox_inches='tight')

    def get_player_images(self):
        players = self.game.get_players()
        fn = players[0].frame_number
        frame_path = os.path.join(self.making_path, "frame_players.png")
        extract_frame_from_video(self.video_path, fn, frame_path)
        for tplayer in self.game.get_players():
            player = tplayer.player_object
            player_path = os.path.join(self.making_path, tplayer.tag + ".png")
            crop_and_save_image(frame_path, player_path, player.x, player.y, player.height,
                                player.width)
            self.player_images[tplayer.tag] = player_path




def shots_to_json(shots, export=False, filename=""):
    json_shots = {}
    for i, shot in enumerate(shots):
        json_shots[i + 1] = [shot.start(), shot.end()]

    if export:
        with open(filename, 'w') as f:
            json.dump(json_shots, f, indent=4)

    return json_shots


def points_to_json(points, export=False, filename=""):
    json_points = []
    for i, point in enumerate(points):
        json_points.append([point.start(), point.end()])

    if export:
        with open(filename, 'w') as f:
            json.dump(json_points, f, indent=4)

    return json_points


def extract_clip(input_path, start_frame, end_frame, output_path, overlay_clip_path, fps=60):
    """Extract clips and overlay a video in the bottom right corner using ffmpeg."""
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    # Command to overlay the video clip
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_time), '-t', str(duration),  # Efficient seeking
        '-i', input_path,  # Input video
        '-i', overlay_clip_path,  # Overlay video
        '-filter_complex', "[1:v]scale=iw*1:ih*1[ovrl];[0:v][ovrl]overlay=W-w-40:H-h-20:enable='lte(t,3)'",        # Scale and position overlay
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'aac',  # Audio codec
        output_path
    ]

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Error handling
    if process.returncode != 0:
        print("Error occurred while executing FFmpeg:")
        print(process.stderr)
        raise RuntimeError("FFmpeg failed with an error. See output above for more details.")

    print("FFmpeg process completed successfully.")


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


def concatenate_clips_with_transition(self, file_names, output_filename, transition_duration=1):
    # Convert transition_duration to float to avoid type issues
    transition_duration = float(transition_duration)

    # List to hold video clips with transitions
    clips_with_transitions = []

    # Load clips and apply fade in and fade out transitions
    start_filename = f"{self.resources_path}/resources/start.mp4"
    clip = VideoFileClip(start_filename)
    clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
    clips_with_transitions.append(clip)

    for filename in file_names:
        clip = VideoFileClip(filename)
        clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
        clips_with_transitions.append(clip)

    # Concatenate clips with transitions
    final_clip = concatenate_videoclips(clips_with_transitions, method="compose")

    # Output file
    final_clip.write_videofile(output_filename, codec="libx264", fps=60, threads=8, progress_bar = False)

    return output_filename


def merge_pairs(pairs):
    if not pairs:
        return []

    # Start with the first pair
    merged_pairs = [pairs[0]]

    for i in range(1, len(pairs)):
        last_merged_pair = merged_pairs[-1]
        current_pair = pairs[i]

        # If the end of the last merged pair overlaps with or touches the start of the current pair
        if last_merged_pair[1] >= current_pair[0]:
            # Merge the two pairs
            merged_pairs[-1] = (last_merged_pair[0], max(last_merged_pair[1], current_pair[1]))
        else:
            # Otherwise, add the current pair to the result as it is
            merged_pairs.append(current_pair)

    return merged_pairs


def json_points_to_video(points, input_video_path, output_video_path, from_path=None, margin=30):
    temp_clips = []

    for i in range(10, 0, -1):
        print("Extracting point " + str(i), end='\r')
        temp_clip_path = f"/home/juliofgx/PycharmProjects/PadelClips/making/temp_clip_{i}.mp4"
        start = points[i - 1][0] - margin
        end = points[i - 1][1] + margin
        overlay_path = f"/home/juliofgx/PycharmProjects/PadelClips/resources/points/{i}.mp4"
        if not os.path.exists(temp_clip_path):
            extract_clip(input_video_path, start, end, temp_clip_path, overlay_path)
        temp_clips.append(temp_clip_path)

    print("Merging clips...")
    concatenate_clips_with_transition(temp_clips, output_video_path)

    # Optionally, clean up temporary clips
    for clip in temp_clips:
        os.remove(clip)
        pass
