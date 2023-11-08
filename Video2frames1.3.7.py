# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:31:56 2023

@author: nicko
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 02:05:34 2023

@author: skippy
"""

import cv2
import os
import datetime
import re
# Get the current date and time


# Replace 'input_video.mp4' with the path to your input video file


# Set 1 if calibration checkboard, 0 if object and aruco markers 
calibrate = 0
#                        change user or other file directory details HERE
input_video_path = 'C:/Users/your username/visual-hull/videos/your video name'
device_name = 'SonyA7'

if calibrate == False:
    object_name = 'ball'
else:
    checkerboard_size = '9x16'



# Get the current date and time
current_date = datetime.date.today()
current_time = datetime.datetime.now().strftime('%H:%M:%S')

# Combine the date and time components
filename = f"{current_date}"

# Define a regex pattern to remove characters not allowed in Windows filenames
# Windows disallows characters like \ / : * ? " < > |
safe_filename = re.sub(r'[\/:*?"<>|]', '_', filename)

print(f"Safely formatted filename: {safe_filename}")

# Replace 'output_folder' with the directory where you want to save the frames
output_folder = f'Calibrate_output_frames{filename}'

if calibrate == 0:
    output_folder = f"object_{device_name}_{safe_filename}{object_name}"
else:
    output_folder = f"calibration_{device_name}_{safe_filename}{checkerboard_size}"


# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Initialize a frame counter
frame_counter = 2
frameSkip = 10
totalframes = 0

# Loop through each frame in the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Break the loop if we have reached the end of the video
    if not ret:
        break
    if frame_counter % frameSkip == 0:
    
        # Save the frame as an image
        frame_filename = os.path.join(output_folder, f'frame_{frame_counter:04d}.png')
        cv2.imwrite(frame_filename, frame)
        totalframes+=1
    # Increment the frame counter
    frame_counter += 1

# Release the video capture object
cap.release()

print(f"Extracted {totalframes} frames to '{output_folder}'")

# Optionally, you can also show the extracted frames
# for i in range(frame_counter):
#     frame_filename = os.path.join(output_folder, f'frame_{i:04d}.png')
#     frame = cv2.imread(frame_filename)
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
