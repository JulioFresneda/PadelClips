import io
import os
import pickle
import random
import subprocess
import re
import sys
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, transfx, vfx, AudioFileClip, afx

from padelClipsPackage import Game

from bs4 import BeautifulSoup

from padelClipsPackage.Point import Shot
from padelClipsPackage.Shot import Category, Position
from padelClipsPackage.aux import extract_frame_from_video, crop_and_save_image

import base64
import ssl
import hashlib


from PIL import Image
import os
import base64

def slide_in(clip, direction="left", duration=1):
    W, H = clip.size
    if direction == "left":
        return clip.set_position(lambda t: (min(0, -W + t/duration * W), 'center'))
    elif direction == "right":
        return clip.set_position(lambda t: (max(0, W - t/duration * W), 'center'))
    elif direction == "top":
        return clip.set_position(lambda t: ('center', min(0, -H + t/duration * H)))
    elif direction == "bottom":
        return clip.set_position(lambda t: ('center', max(0, H - t/duration * H)))
    return clip


class ComposeVideo:
    def __init__(self, game: Game, making_path, resources_path, video_path, output_path, player_images={}):
        self.player_images = player_images
        self.game = game
        self.gameStats = game.gameStats if self.game is not None else None
        self.making_path = making_path
        self.resources_path = resources_path
        self.video_path = video_path
        self.output_folder = output_path

        self.players = sorted(self.game.get_players(), key=lambda p: p.player_object.y)
        self.players[:2] = sorted(self.players[:2], key=lambda p: p.player_object.x)
        self.players[2:] = sorted(self.players[2:], key=lambda p: p.player_object.x)

        self.generate_alias()

        self.paths = []

        if player_images == {}:
            self.get_player_images()
        self.generate_players_in_field()
        self.generate_heatmaps()
        self.generate_medals()

        self.paths.append(f"{self.resources_path}/images/top10.png")



        self.compose_video()

    def generate_alias(self):
        aliases = [
            "Rayo Veloz",
            "Pantera Negra",
            "Maestro Cancha",
            "Titan Invencible",
            "Tornado Rojo",
            "Serpiente Agil",
            "Rey de las Voleas",
            "Guerrero de la Raqueta",
            "Maquina Imparable",
            "Gladiador Feroz",
            "Risas Epicas",
            "Titan Raquetas",
            "Payaso Astuto",
            "Halcon Rapido",
            "Samurai Fuerte",
            "Bomba Cancha",
            "Dragon Fiero",
            "Ninja Silencioso",
            "Kraken Poderoso",
            "Bestia Fiera",
            "Maestro Risueño",
            "Explosivo Gigante",
            "Estrella Brillante",
            "Comico Valiente",
            "Fenix Renacido",
            "Leon Valiente",
            "Fantasma Agil",
            "Espartano Alegre",
            "Sombra Sigilosa"
        ]

        #alias = random.sample(aliases, 4)
        alias = ['Julio', 'Joa', 'Raul', 'Pedro']
        self.alias = {}
        for i, p in enumerate(self.players):
            self.alias[p.tag] = alias[i]

    def generate_player_data(self, tag, index):
        html_path = self.resources_path + '/player_data_template.html'
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        html_content = html_content.replace("CHART_PATH", self.making_path + f'/shots_chart_{tag}.png')
        html_content = html_content.replace("HEATMAP_PATH", self.making_path + f'/heatmap_{tag}.png')

        stats = self.gameStats.categorize_player_shots()[tag]
        stats = [stats['right'], stats[Category.SMASH], stats['left'], stats[Category.FULL_GLOBE]]

        html_content = html_content.replace("{{ dcha }}", str(stats[0]) + '%')
        html_content = html_content.replace("{{ izq }}", str(stats[2]) + '%')
        html_content = html_content.replace("{{ smashes }}", str(stats[1]))
        html_content = html_content.replace("{{ globos }}", str(stats[3]))

        plats = {2: "images/plat_orange.png", 1: "images/plat_yellow.png", 3: "images/plat_pink.png",
                 4: "images/plat_red.png"}
        colors = {1: '#cfdb46', 2: '#db9a46', 3: '#db46cc', 4: '#db4646'}
        html_content = html_content.replace("PLAT_PATH", f"{self.resources_path}/{plats[index + 1]}")
        html_content = html_content.replace("BORDER_COLOR", colors[index + 1])
        html_content = html_content.replace("PLAYER_PATH", self.player_images[tag])

        self.html_content_to_png(html_content, f'player_data_{tag}.png')

    def compose_video(self):
        points = self.game.gameStats.top_x_longest_points(10)
        self.make_clips(points, margin=60)

        points = self.game.gameStats.top_x_minutes(5)
        self.make_resume(points, margin=60)



    def make_resume(self, points, margin):
        temp_clips = []

        for i in range(len(points)):
            print("Extracting point " + str(i), end='\r')
            temp_clip_path = f"{self.making_path}/temp_clip_{i}.mp4"
            start = points[i].start() - margin
            end = points[i].end() + margin

            if not os.path.exists(temp_clip_path):
                extract_clip_wo_overlap(self.video_path, start, end, temp_clip_path)
            temp_clips.append(temp_clip_path)

        print("Merging clips...")

        target_resolution = (1920, 1080)

        # List to hold video clips with transitions
        clips = []

        # Load the start clip and apply fade-in and fade-out
        start_clip = VideoFileClip(f"{self.resources_path}/start.mp4").resize(target_resolution)
        start_clip = start_clip.fx(vfx.fadein, 1).fx(vfx.fadeout, 1)
        clips.append(start_clip)

        # Load image clips, apply crossfade transition, and add to the list


        # Load additional video files, apply fade-in/out, and add to the list
        for filename in temp_clips:
            clip = VideoFileClip(filename).resize(target_resolution)
            clip = clip.fx(vfx.fadein, 1).fx(vfx.fadeout, 1)
            clips.append(clip)

        end_clip = VideoFileClip(f"{self.resources_path}/end.mp4").resize(target_resolution)
        end_clip = end_clip.fx(vfx.fadein, 1).fx(vfx.fadeout, 1)
        clips.append(end_clip)

        # Create a composite video from all the clips with transitions
        final_clip = concatenate_videoclips(clips, method='compose')

        audio = AudioFileClip(f"{self.resources_path}/audio.mp3")
        audio = afx.audio_loop(audio, duration=final_clip.duration)
        final_clip = final_clip.set_audio(audio)

        # Write the final video output
        final_clip.write_videofile(f"{self.output_folder}/PadelClipsResumen.mp4", codec="libx264", fps=30)



        # Optionally, clean up temporary clips
        for clip in temp_clips:
            os.remove(clip)
            pass

    def make_clips(self, points, margin):

        temp_clips = []

        for i in range(len(points), 0, -1):
            print("Extracting point " + str(i), end='\r')
            temp_clip_path = f"{self.making_path}/temp_clip_{i}.mp4"
            start = points[i - 1].start() - margin
            end = points[i - 1].end() + margin
            overlay_path = f"{self.resources_path}/images/points/{i}.png"
            if not os.path.exists(temp_clip_path):
                extract_clip(self.video_path, start, end, temp_clip_path, overlay_path)
            temp_clips.append(temp_clip_path)

        print("Merging clips...")


        self.concatenate_clips_with_transition(temp_clips, self.output_folder)

        # Optionally, clean up temporary clips
        for clip in temp_clips:
            os.remove(clip)
            pass

    def left_to_right_crossfade(self, clip, duration):
        """
        Creates a left-to-right crossfade for a given clip.
        The clip gradually fades in from left to right during the given duration.
        """
        # Create a mask that moves from left to right
        mask = ImageClip(np.tile(np.linspace(0, 1, clip.w), (clip.h, 1)).T, ismask=True)
        mask = mask.set_duration(duration).resize(clip.size).set_pos('center')

        # Apply the mask to the clip
        return clip.set_mask(mask).set_duration(duration)
    def concatenate_clips_with_transition(self, file_names, output_folder, transition_duration=1):
        target_resolution = (1920, 1080)

        # List to hold video clips with transitions
        clips = []

        # Load the start clip and apply fade-in and fade-out
        start_clip = VideoFileClip(f"{self.resources_path}/start.mp4").resize(target_resolution)
        start_clip = start_clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
        clips.append(start_clip)

        # Load image clips, apply crossfade transition, and add to the list
        start = start_clip.duration
        for i, image_filename in enumerate(self.paths):
            if 'medal' in image_filename and '2' in image_filename:
                duration = 2
            elif 'medal' in image_filename and '3' in image_filename:
                duration = 3
            else:
                duration = 3
            tr_duration = 1 if 'medal' in image_filename and '2' else transition_duration
            image_clip = ImageClip(image_filename, duration=duration).resize(target_resolution)
            image_clip = image_clip.set_start(start).crossfadein(tr_duration)
            clips.append(image_clip)
            start += duration - tr_duration

        # Load additional video files, apply fade-in/out, and add to the list
        for filename in file_names:
            clip = VideoFileClip(filename).resize(target_resolution)
            clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
            clip = clip.set_start(start)
            clips.append(clip)
            start += clip.duration - transition_duration

        end_clip = VideoFileClip(f"{self.resources_path}/end.mp4").resize(target_resolution)
        end_clip = end_clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration).set_start(start)
        clips.append(end_clip)

        # Create a composite video from all the clips with transitions
        final_clip = CompositeVideoClip(clips)

        audio = AudioFileClip(f"{self.resources_path}/audio.mp3")
        audio = afx.audio_loop(audio, duration=final_clip.duration)
        final_clip = final_clip.set_audio(audio)

        # Write the final video output
        final_clip.write_videofile(f"{output_folder}/PadelClipsTop10.mp4", codec="libx264", fps=30)



    def generate_medals(self):
        medals = self.gameStats.medals

        for medal_name, stats in medals.items():
            print(f"Making png for medal {medal_name}", end='\r')
            tag = stats[0]
            index = None
            for i, player in enumerate(self.players, start=1):
                if player.tag == tag:
                    index = i

            self.generate_medal_html(medal_name, stats[0], index, stats[1])


    def generate_medal_html(self, medal_name, tag, index, stats):
        html_path = self.resources_path + '/html/medals.html'
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        html_content = html_content.replace(f"bastion.png", f"{medal_name.lower()}.png")
        html_content = html_content.replace("{{ title }}", medal_name)
        html_content = html_content.replace("<div class='stats'><h1>{{ stats }}</h1></div>", "<div class='stats'><h1></h1></div>")

        html_content = html_content.replace("PLAT_PATH", f"{self.resources_path}/images/plat_{index}.png")
        html_content = html_content.replace("PLAYER_PATH", f"{self.making_path}/{tag}.png")
        colors = {1: '#cfdb46', 2: '#db9a46', 3: '#db46cc', 4: '#db4646'}
        html_content = html_content.replace("BORDER_COLOR", colors[index])

        path = self.html_content_to_png(html_content, f"medal_{medal_name}_1.png")
        self.paths.append(path)

        html_content = html_content.replace(f"medals_1.png", f"medals_2.png")




        def switch_visibility(html_content):
            html_content = html_content.replace("visibility: hidden", "visibility: tmp")
            html_content = html_content.replace("visibility: visible", "visibility: hidden")
            html_content = html_content.replace("visibility: tmp", "visibility: visible")
            return html_content

        html_content = switch_visibility(html_content)
        html_content = html_content.replace(medal_name, self.alias[tag])
        stats_str = self.gameStats.get_medal_description(medal_name, stats, self.alias)
        html_content = html_content.replace("<div class='stats'><h1></h1></div>", f"<div class='stats'><h1>{stats_str}</h1></div>")

        path = self.html_content_to_png(html_content, f"medal_{medal_name}_2.png")
        self.paths.append(path)
        #html_content = switch_visibility(html_content)
        path = self.html_content_to_png(html_content, f"medal_{medal_name}_3.png")
        self.paths.append(path)




    def generate_html_medals(self, player):
        filepaths = []
        medals = self.game.gameStats.get_medals(player.tag)
        input_file = self.resources_path + '/medals.html'
        for medal, result in medals.items():
            print(f"Making png for medal {medal}", end='\r')
            html_content = self.generate_html_common(player, input_file)
            html_content = html_content.replace('medals/bastion.png', f"../resources/medals/{medal.lower()}.png")

            titles = self.game.gameStats.medals_descriptions[medal]
            title = titles[0].replace("{{ result }}", str(result))
            subtitle = titles[1]
            html_content = html_content.replace('{{ title }}', title)
            html_content = html_content.replace('{{ subtitle }}', subtitle)

            output_file = self.making_path + f'/medal_{player.tag}_{medal.lower()}.html'  # Replace with your desired output file path

            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(html_content)

            png_output = self.making_path + f'/medal_{player.tag}_{medal.lower()}.png'
            filepaths.append(png_output)
            self.html_to_png(output_file, png_output)
        return filepaths

    def generate_players_in_field(self):
        input_file = self.resources_path + '/html/players_in_field_template.html'
        with open(input_file, 'r', encoding='utf-8') as file:
            html_content = file.read()

        image_lines = []

        for i, player in enumerate(self.players):
            player_path = self.player_images[player.tag]
            #html_content = html_content.replace('player.png', self.making_path + f'/{player.tag}.png')

            line = f'<img src="../images/player.png" alt="Player Image" class="player player-{str(i + 1)}">'
            newline = line.replace("../images/player.png", player_path)

            html_content = html_content.replace(line, newline)
            image_lines.append(newline)

        # Find the image with specific class or source

        paths = []

        for j in range(4, 0, -1):
            html_content_tmp = html_content.replace("players_in_field_0.png", f"players_in_field_{j}.png")
            html_content_tmp = html_content_tmp.replace("JUGADORES", list(self.alias.values())[j - 1])
            path = self.html_content_to_png(html_content_tmp, f'players_in_field_{str(j)}.png')
            paths.append(path)
            html_content = html_content.replace(image_lines[j - 1], '')

        path = self.html_content_to_png(html_content, f'players_in_field_0.png')
        paths.append(path)
        self.paths += paths[::-1]

    def generate_html_player_stats(self, player, input_file):
        html_content = self.generate_html_common(player, input_file)
        if '1.html' in input_file:
            self.generate_html_1(html_content, player)
        else:
            self.generate_html_2(html_content, player)

    def generate_html_common(self, player, input_file):

        with open(input_file, 'r', encoding='utf-8') as file:
            html_content = file.read()

        custom_names = {'A': 'Macho alfalfa', 'B': "El puto", 'C': "Primo de Raul", 'D': "Heterocurioso"}

        # Replace the placeholders with actual player data
        html_content = html_content.replace('{{ tag }}', player.tag)
        html_content = html_content.replace('{{ keyname }}', custom_names[player.tag])
        html_content = html_content.replace('player.png', self.making_path + f'/{player.tag}.png')

        position = "Abajo" if player.player_object.position is Position.BOTTOM else "Arriba"
        html_content = html_content.replace('{{ position }}', position)
        teammate = None
        for p in self.game.player_templates:
            if p.tag != player.tag and p.player_object.position is player.player_object.position:
                teammate = p.tag
        html_content = html_content.replace('{{ tag_teammate }}', teammate)
        return html_content

    def html_content_to_png(self, html_content, png_name):
        output_file = self.making_path + '/tmp.html'  # Replace with your desired output file path

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(html_content)

        png_output = self.making_path + f'/{png_name}'
        self.html_to_png(output_file, png_output)

        return png_output

    def generate_html_2(self, html_content, player):
        html_content = html_content.replace('{{ mrun }}',
                                            str(int(self.game.gameStats.meters_ran(player.tag))) + ' metros')

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
            #print(f"Successfully converted {html_file_path} to {output_png_path} in 4K resolution.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while converting HTML to PNG: {e}")

    def generate_heatmaps(self):
        self.heatmaps_paths = {}
        self.hm_data = {}
        for i, player in enumerate(self.players, start=1):
            print(f"Making heatmaps for player {player.tag}", end='\r')
            pos_data = self.game.get_player_positions_scaled(player.tag)


            self.heatmaps_paths[player.tag] = self.generate_heatmap(pos_data, player.tag)
            self.hm_data[player.tag] = self.generate_heatmap_data(player.position, pos_data)

            self.generate_heatmap_html(player.tag, i)

    def generate_heatmap_html(self, tag, index):
        self.generate_heatmap_html_base(tag, index)
        self.generate_heatmap_html_details(tag, index, self.hm_data[tag])

    def generate_heatmap_html_details(self, tag, index, data):
        html_path = f"{self.resources_path}/html/heatmap/player_heatmap_details_{index}.html"

        for level in range(2, 5):
            with open(html_path, 'r', encoding='utf-8') as file:
                html_det = file.read()

            html_det = html_det.replace(f"heatmap_{index}_4.png", f"heatmap_{index}_{level}.png")

            html_det = self.replace_common_player_data(html_det, tag, index)

            if level < 4:
                html_det = html_det.replace("{{ data3 }}", "")
            else:
                html_det = html_det.replace("{{ data3 }}", str(data['lado']) + '% fuera')
            if level < 3:
                html_det = html_det.replace("{{ data2 }}", "")
            else:
                html_det = html_det.replace("{{ data2 }}", str(data['net']) + '% en red')

            html_det = html_det.replace("{{ data1 }}", str(data['fondo']) + '% en fondo')

            path = self.html_content_to_png(html_det, f"heatmap_det_{index}_{level}.png")
            self.paths.append(path)

    def replace_common_player_data(self, html_content, tag, index):
        #html_content = html_content.replace("HEATMAP_PATH", self.heatmaps_paths[tag])
        html_content = html_content.replace("heatmap_1_1.png", f"heatmap_{str(index)}_1.png")

        html_content = html_content.replace("{{ player_name }}", self.alias[tag])
        html_content = html_content.replace("PLAT_PATH", f"{self.resources_path}/images/plat_{index}.png")
        html_content = html_content.replace("PLAYER_PATH", f"{self.making_path}/{tag}.png")

        colors = {1: '#cfdb46', 2: '#db9a46', 3: '#db46cc', 4: '#db4646'}
        html_content = html_content.replace("BORDER_COLOR", colors[index])

        return html_content

    def generate_heatmap_html_base(self, tag, index):
        html_path = f"{self.resources_path}/html/heatmap/player_heatmap_base.html"
        with open(html_path, 'r', encoding='utf-8') as file:
            html_base = file.read()

        html_base = self.replace_common_player_data(html_base, tag, index)


        if index < 3:
            html_base = html_base.replace("{{ position_down }}", "")
            if index == 1:
                html_base = html_base.replace("{{ position_up }}", "Reves")
            else:
                html_base = html_base.replace("{{ position_up }}", "Drive")
        else:
            html_base = html_base.replace("{{ position_up }}", "")
            if index == 1:
                html_base = html_base.replace("{{ position_down }}", "Reves")
            else:
                html_base = html_base.replace("{{ position_down }}", "Drive")

        path = self.html_content_to_png(html_base, f'heatmap_base_{index}.png')
        self.paths.append(path)

    def generate_heatmap_data(self, position, coordenates):
        def calculate_percentage_near_point(coordinates, target, threshold=0.15, vertical=True):
            # Count how many times the object is near the target point vertically
            count_near = 0
            total_frames = len(coordinates)

            for x, y in coordinates:
                # Check if the vertical distance is within the threshold
                if vertical:
                    if abs(y - target) <= threshold:
                        count_near += 1
                else:
                    if abs(x - target) <= threshold:
                        count_near += 1

            # Calculate percentage
            percentage_near = (count_near / total_frames) * 100
            return int(percentage_near)

        categories = {}
        if position is Position.TOP:
            #fondo = self.game.players_boundaries_vertical[Position.TOP]
            fondo = 0
        else:
            #fondo = self.game.players_boundaries_vertical[Position.BOTTOM]
            fondo = 1
        categories['fondo'] = calculate_percentage_near_point(coordenates, fondo)
        categories['net'] = calculate_percentage_near_point(coordenates, 0.5)

        mean_x = sum([pos[0] for pos in coordenates]) / len(coordenates)
        if mean_x < 0.5:
            categories['lado'] = calculate_percentage_near_point(coordenates, 0.75, threshold=0.25, vertical=False)
        else:
            categories['lado'] = calculate_percentage_near_point(coordenates, 0.25, threshold=0.25, vertical=False)

        return categories



    def generate_heatmap(self, data, tag):

        # Extract x and y coordinates
        x_values, y_values = zip(*data)

        # Create a 2D histogram (heatmap) from the x and y values
        heatmap, xedges, yedges = np.histogram2d(x_values, y_values, bins=50, range=[[0, 1], [0, 2]])

        # Mask the heatmap where there is no data (i.e., where values are 0)
        heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)

        # Create the figure with a 1:2 aspect ratio (taller than wide) without the legend
        fig, ax = plt.subplots(figsize=(5, 10))

        ax.axis('off')

        # Create a colormap and set 'bad' (masked) data to be fully transparent
        cmap = plt.cm.get_cmap('coolwarm')
        cmap.set_bad(color=(0, 0, 0, 0))  # RGBA for fully transparent

        # Plot the heatmap with transparency where no data exists
        cax = ax.imshow(heatmap_masked.T, extent=[0, 1, 0, 2], origin='lower', cmap=cmap, aspect='auto', alpha=0.6,
                        zorder=1)

        # Remove labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set the coordinate range from 0 to 1 for the x-axis and 0 to 2 for the y-axis
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)

        savename = f"{self.making_path}/heatmap_{tag}.png"

        # Save the figure with a transparent background
        plt.savefig(savename, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)

        return savename

    def generate_shots_graph(self, stats, tag):

        # Data for the spider chart
        labels = [" ", " ", " ", " "]
        #labels = []
        stats = [stats['right'], stats[Category.SMASH], stats['left'], stats[Category.FULL_GLOBE]]
        #stats = []
        #for i in range(4):
        #    labels[i] = labels[i] + " (" + str(int(10 * stats[i])) + ")"

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

        # Set transparent background for the entire figure, but keep the axes' face color
        fig.patch.set_alpha(0)

        fig.savefig(self.making_path + '/shots_chart_' + tag + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

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


def extract_clip(input_path, start_frame, end_frame, output_path, overlay_clip_path=None, fps=60):
    """Extract clips and overlay a video in the bottom right corner using ffmpeg."""
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    # Command to overlay the video clip
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_time), '-t', str(duration),  # Efficient seeking
        '-i', input_path,  # Input video
        '-i', overlay_clip_path,  # Loop the image input
        '-filter_complex',
        "[0:v]scale=1920:1080,setsar=1[v0];[1:v]scale=iw*0.25:ih*0.25[ovrl];[v0][ovrl]overlay=W-w-60:H-h-60",
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'aac',  # Audio codec
        '-shortest',
        output_path
    ]

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Error handling
    if process.returncode != 0:
        print("Error occurred while executing FFmpeg:")
        print(process.stderr)
        raise RuntimeError("FFmpeg failed with an error. See output above for more details.")

    print("FFmpeg process completed successfully.")



def extract_clip_wo_overlap(input_path, start_frame, end_frame, output_path, fps=60):
    """Extract clips and overlay a video in the bottom right corner using ffmpeg."""
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    # Command to overlay the video clip
    command = [
        'ffmpeg', '-y',
        '-ss', str(start_time), '-t', str(duration),  # Efficient seeking
        '-i', input_path,  # Input video
        '-filter_complex',
        "[0:v]scale=1920:1080",
        '-c:v', 'libx264',  # Video codec
        '-c:a', 'aac',  # Audio codec
        '-shortest',
        output_path
    ]

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Error handling
    if process.returncode != 0:
        print("Error occurred while executing FFmpeg:")
        print(process.stderr)
        raise RuntimeError("FFmpeg failed with an error. See output above for more details.")

    print("FFmpeg process completed successfully.")


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
    final_clip.write_videofile(output_filename, codec="libx264", fps=60, threads=8, progress_bar=False)

    return output_filename


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
