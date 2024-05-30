import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.ion()
matplotlib.use('TkAgg')
matplotlib.interactive(False)

# Load the image
image_path = '/home/juliofgx/PycharmProjects/padelLynx/dataset/padel2/segment1/images/test/frame_000978.PNG'
img = cv2.imread(image_path)

# Convert image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for white color and create a mask
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 25, 255])
mask = cv2.inRange(hsv, lower_white, upper_white)

# Apply the mask to get the white regions
masked_img = cv2.bitwise_and(img, img, mask=mask)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)

lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.show()