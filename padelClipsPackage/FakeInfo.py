import pickle
from functools import reduce

import numpy as np



with open('/home/juliofgx/PycharmProjects/PadelClips/pos.pkl', 'rb') as file:
    pos_data = pickle.load(file)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image


def generate_heatmap(data, tag):
    # Extract x and y coordinates
    x_values, y_values = zip(*data)

    # Create a 2D histogram (heatmap) from the x and y values
    heatmap, xedges, yedges = np.histogram2d([], [], bins=50, range=[[0, 1], [0, 2]])

    # Mask the heatmap where there is no data (i.e., where values are 0)
    heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)

    # Create the figure with a 1:2 aspect ratio (taller than wide) without the legend
    fig, ax = plt.subplots(figsize=(5, 10))

    bg_img = Image.open(f"/home/juliofgx/PycharmProjects/PadelClips/resources/images/heatmap.png")
    ax.imshow(bg_img, extent=[0, 1, 0, 2], aspect='auto', zorder=0)

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

    # Save the figure with a transparent background
    plt.savefig('heatmap_' + tag + '.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)




def scale_position(pos_data):

    top = 0.34
    bottom = 1.07
    net = 0.51
    left = 0.23
    right = 0.95

    def scale_x(x, start, end):
        return (x-start)/(end-start)

    def scale_y(y, top, bottom):
        return (y-top)/(bottom-top)

    scaled = []
    for pos in pos_data:
        x = scale_x(pos[0], left, right)
        y = scale_y(pos[1], top, bottom)
        scaled.append((x, y))
    return scaled


pos_data = scale_position(pos_data)[10000:11000]










print(f"Max x: {str(max(pos_data, key=lambda x:x[0]))}")
print(f"Min x: {str(min(pos_data, key=lambda x:x[0]))}")

xtotal = sum([x[0] for x in pos_data])
print(f"Mean x: {str(xtotal/len(pos_data))}")


print(f"Max y: {str(max(pos_data, key=lambda x:x[1]))}")
print(f"Min y: {str(min(pos_data, key=lambda x:x[1]))}")

ytotal = sum([y[1] for y in pos_data])
print(f"Mean y: {str(ytotal/len(pos_data))}")

generate_heatmap(pos_data, 'D')