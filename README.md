## Image Processing and 3D Modeling with Python using Visual Hull

This Python script is designed for image processing and 3D modeling tasks. It uses video to capture a set of images, detect Aruco markers, extracts contours, performs camera calibration, and create a 3D reconstruction of the desired object.

## Prerequisites

Before running this script, make sure you have the following libraries installed:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- gmsh (`gmsh`)
- sys (`sys`)
- logging (`logging`)

You can install these libraries using `pip` or any other package manager.

## Joose Usage 13/10

1. Clone this repository to your local machine.
2. Specific directions for Vis_Hull_2.0 and Checker_Calib_1.3.7 are included in english within the document, and within the first ~65 lines.  Although the Vis_Hull already contains the correct output from the Checker_Calib file.
3. Ensure that you have a set of images in the specified folder path, and the image filenames match the specified pattern.
4. Run the scripts and check for printed results.

## General Usage

1. Clone this repository to your local machine or download the script.
2. Modify the script's configuration parameters according to your specific requirements, ex. folder paths, the camera calibration parameters are default tuned to the '3_10oneplus(object_filmcan_photo)', and scaling factors are true for all of the included object images 9/10/23.
3. Ensure that you have a set of images in the specified folder path, and the image filenames match the specified pattern.
4. Run the script using Python:

   ```bash
   python script.py
The script will process each image, detect Aruco markers, extract contours, perform camera calibration, and generate 3D models.
Configuration Parameters
folder_path: The path to the folder containing the input images.
imagename_pattern: The pattern for matching image filenames.
Camera calibration parameters (CMTX and DIST): These parameters define the camera's intrinsic matrix and distortion coefficients.
centerpoint: The coordinates of the centerpoint in the world coordinate system.
aruco_points: Dictionary defining Aruco marker points in the world coordinate system.
Other parameters related to image processing and modeling.
Output
The script generates 3D models based on the detected contours and Aruco markers. These models are written to Gmsh format files with the .msh extension.

Troubleshooting
If you encounter any issues or errors while running the script, please refer to the error messages and ensure that the required libraries are correctly installed.
License
This script is provided under the MIT License. Feel free to modify and use it for your projects.
