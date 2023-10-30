# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:35:07 2023

@author: nicko
"""

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import glob
import os
import gmsh as msh
import sys
import logging
import csv
'''
*PLEASE CHECK single quotes for directions throughout the initial
part of the doc*

Parameters (do not change these for now)
'''

msh.initialize(sys.argv)
projection_list = [] 
aruco_scale = 60 #56.76 # size of aruco markers 
LC = 10 # resolution for GMSH vertices - only relevant for boolean
SCALING = 1  # iamge pixel dimension multiplier
debug_mode = 0


'''
These are the only two variables that need to be changed with the 
current repo. Add the folder path with 'object' and 'oneplus' to 
the path, imagename should be a shared initial portion of a string
followed by the star. (ie IMG*)
'''
folder_path = 'C:/Users/nicko/Object_images_2023-10-1721_26_44/'
imagename_pattern = 'frame*'



'''
This is the CMTX and DIST that are obtained from the calib.py file in
the repo, currently they are set to work with all github files 
including the name'oneplus'.  To recreate, go to the checker calib file.

'''

# Current CMTX and DIST found using 9x6 checkerboard.
# CMTX = np.array([738.4391110895575, 0.0, 391.10649709357614, 0.0,
#                  737.6804692382966, 516.2104937135692, 0.0, 0.0, 1.0], 
#                 dtype='float32').reshape(3, 3)
# DIST = np.array([-0.028550282679872235, -0.01788452651293315, 
#                  -0.0002072655617462208, 0.0006906692231601624, 
#                  0.5017724257813853], dtype='float32')
CMTX = np.array([1892.6743149817203, 0.0, 840.0250608153973, 0.0, 1892.7045637145145, 648.2388916667707, 0.0, 0.0, 1.0], dtype='float32').reshape(3, 3)
DIST = np.array([0, 0, 0, 0, 0], dtype='float32')


'''
These are the the real world coordinates for aruco corner locations 
in mm.  These are set to work with all github files including the 
name 'object'
'''
aruco_points = {
    # 39: np.array([[32.5,-32.5,-2.5],[32.5,32.5,-2.5],[32.5,-32.5,-62.5],
    #               [32.5,32.5,-62.5]], dtype='float32'),
    # 40: np.array([[32.5,32.5,-2.5],[-32.5,32.5,-2.5],[32.5,32.5,-62.5],
    #               [-32.5,32.5,-62.5]], dtype='float32'),
    # 41: np.array([[-32.5,32.5,-2.5],[-32.5,-32.5,-2.5],[-32.5,-32.5,-62.5],
    #               [-32.5,32.5,-62.5]], dtype='float32'),
    # 42: np.array([[-32.5,-32.5,-2.5],[32.5,-32.5,-2.5],[32.5,-32.5,-62.5],
    #               [-32.5,-32.5,-62.5]], dtype='float32'),
    
    20: np.array([[-5.02,47.44,1.06],[-47.44,5.02,1.06],[-77.44,35.02,-41.37],
                  [-35.02,77.44,-41.37]],dtype='float32'),
    
    21: np.array([[-47.44,-5.02,1.06],[-5.02,-47.44,1.06],[-35.02,-77.44,-41.37],
                  [-77.44,-35.02,-41.37]],dtype='float32'),

    22: np.array([[5.02,-47.44,1.06],[47.44,-5.02,1.06],[77.44,-35.02,-41.37],
                  [35.02,-77.44,-41.37]],dtype='float32'),

    23: np.array([[47.44,5.02,1.06],[5.02,47.44,1.06],[35.02,77.44,-41.37],[77.44,35.02,-41.37]],dtype='float32'),


    }


'''
Functions
'''

# msh.option.setNumber("General.AbortOnError", 1)

# darkness caLCulator
def calculate_darkness(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], 0, 255, -1)
    masked_image = cv.bitwise_and(image, image, mask=mask)
    darkness_value = np.mean(masked_image[mask > 0]) - 25
    return darkness_value


# resizing function
def resize_image(img, SCALING):
    width = int(img.shape[1] * SCALING)
    height = int(img.shape[0] * SCALING)
    dim = (width, height)
    image = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return image


# Detects aruco corners, IDS, and darkness values
def detect_markers(image):
    # Parameters for detection
    dictionary= cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_50)
    parameters = cv.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = 1
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    
    # Detect Aruco markers in the image
    corners, ids, _ = detector.detectMarkers(image)
    # aruco_corner_darkness = []
    # for i in range(len(corners)):
    #     # CaLCulate the darkness value of the black portion of the marker
    #     darkness = calculate_darkness(image, corners[i][0].astype(int))
    #     aruco_corner_darkness.append(darkness)
    #     if debug_mode == 1:
    #         print(f"Marker {ids[i][0]} Darkness: {darkness:.2f}")  
    return corners, ids #darkness, aruco_corner_darkness


# Detects outlines of hook (uses independent resizing)
def find_contours(imagename,debug_mode):
    print("Finding Contours")
    print()
    img = cv.imread(imagename, 0)
    img = resize_image(img,SCALING)
    img = cv.GaussianBlur(img, (5, 5), 5)
    # img = cv.medianBlur(img,15)
    width = int(img.shape[1] )
    height = int(img.shape[0] )
    
    #currently thresholding twice 
    # thresholded = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                    cv.THRESH_BINARY, 21, -5)
    ret, thresholded = cv.threshold(img, 0, 255,
                                    cv.THRESH_BINARY + cv.THRESH_OTSU)
   
    # Inversion needed for contour detection to work
    thresholded = cv.bitwise_not(thresholded)
    if debug_mode == 1:
        cv.imshow('Thresholded', thresholded)
        cv.waitKey(0)
        cv.destroyAllWindows()
    contours, hierarchy = cv.findContours(image=thresholded, 
                            mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    # print(contours)
    centroid_list = []
    for contour in contours:
        moments = cv.moments(contour)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        distance_from_center = np.sqrt((cX - width/2)**2 + (cY - height/2)**2)
        # print(distance_from_center,'centroid distance from center')
        centroid_list.append(distance_from_center)
        
        
    mincentroid = min(centroid_list)
    x = centroid_list.index(mincentroid)
    # print("cenbtroid list", centroid_list)
    # print(x, "X")
    if debug_mode ==  2:
        print(min(centroid_list),f"Minimum centroid distance to center (ID: {x})")
    coords = list(zip(contours[x][:, 0][:, 0], contours[x][:, 0][:, 1]))

    
    # Show the image with contours - optional
    if debug_mode == 1:
        img_with_contours = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(img_with_contours, [contours[x]], -1, (0, 255, 0), 2)
        cv.imshow('Central Contour Highlight', img_with_contours)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return coords

# Combines aruco location data 
def multi_pose_estimator(ids):
    # Initialize lists to store matched image points and correspondi0ng object points
    matched_image_points = []
    matched_object_points = []

    # Define the IDS to search for
    key_list = []
    for key in aruco_points:
        key_list.append(key)

    # create image points and object point arrays based on aruco_id values
    for i, value in enumerate(key_list):
        if value in ids:
            
            # Get the index of the ids array that matches the aruco_id list value
            marker_index = np.where(ids == value)[0][0]

            # extract the corners value corresponding to the ids array and aruco_id value
            marker_corners = corners[marker_index][0]
            # print(ids[marker_index])
            # print(marker_corners )
            # Get the 3D coordinates from the aruco_points dictionary treating the value as the key
            aruco_object_points = aruco_points[value]
            # print (value)
            # print(aruco_object_points)
            
            # Add matched points to the lists
            matched_image_points.append(marker_corners)
            matched_object_points.append(aruco_object_points)
            
    #convert to float32 arrays. float 64 is optional for object points
    matched_image_points = np.array(matched_image_points, dtype=np.float32)
    matched_object_points = np.array(matched_object_points, dtype=np.float32)

    #reshape to 2d arrays, points should still be in the correct order.  THIS  FOrotation_matrx IS 
    #REQUIRED for solvePNP to work correctly and process all objects into the alg
    shaper = (matched_image_points.shape[0])*4
    reshaped_array_img = np.reshape(matched_image_points , (shaper, 2))
    reshaped_array_obj = np.reshape(matched_object_points , (shaper, 3))
    return reshaped_array_img, reshaped_array_obj


# Solves perspective and point
def pnp_solver():
    # Perform camera pose estimation using solvePnP
    success, rotation_vector, translation_vect = cv.solvePnP(
        reshaped_array_obj , reshaped_array_img , CMTX, DIST)
    if debug_mode == 3:
        print("Rotation Vector:")
        print(rotation_vector)
        print("Translation Vector:")
        print(translation_vect)
        print("Success Y/N:",success)
    
    # CaLCulate rotation matrix using the rotation vector from solvePnp
    rotation_matrx = cv.Rodrigues(rotation_vector)[0]
    
    #CaLCulates camera position from rotation matrix and translation vector
    camera_position_pnp = np.array(-np.matrix(rotation_matrx).T * np.matrix(translation_vect))
    
    #creates Camera projection matrix from multiplying camera calibration matrix by the first two columns 
    #of the rotation matrix and 1st column of translation vector
    H = CMTX @ np.column_stack((rotation_matrx[:, 0], rotation_matrx[:, 1], translation_vect[:, 0]))
    return H,camera_position_pnp,rotation_matrx,translation_vect,rotation_vector


# Transforms image plane to world coordinate sys  
def dimensional_transforms_contours(contour_points):
   
    # Convert 2D points to 3D rays
    unit_vector_list=[]
    world_coordinates_list = []

    
    for image_coordinates in contour_points:
    
        u, v = image_coordinates
    
        # Homogenize the pixel coordinate to a 3d array
        homogenized_coordinates = np.array([u, v, 1]).T.reshape((3, 1)) #ALL
    
        # Transform 3d pixel to camera coordinate frame w/ inv of camera matrix
        camera_frame_coordinates = np.linalg.inv(CMTX) @ \
            homogenized_coordinates # 1_ and 2_

        # Transform pixel camera coordinate to World coordinate frame 
        world_frame_coordinates = -rotation_matrx.T.dot(translation_vect) + \
            (rotation_matrx.T@camera_frame_coordinates) 
        world_coordinates_list.append(world_frame_coordinates)
        # Transform camera origin in World coordinate frame
        camera_0 = np.array([0,0,0]).T; camera_0.shape = (3,1)
        cam_world = -rotation_matrx.T.dot(translation_vect) + rotation_matrx.T\
            @ camera_0
        
        vector = world_frame_coordinates - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        
        unit_vector_list.append(unit_vector)
        
        
    return unit_vector_list,cam_world


# STL creation script begins
def polyhedron_obj_converter(world_points, campP):
    msh.initialize()
    msh.option.setNumber("General.AbortOnError", 1) 
        
    testcp = [[i[0],i[1],i[2]] for i in world_points]
    #testcamp = tuple([item for sublist in campP1 for item in sublist])
    
    x = round(campP[0][0])
    y = round(campP[1][0])
    z = round(campP[2][0])
    
    # Outline of object
    example_points = [[-3, 3, 0], [3, 3, 0], [3, -3, 0], [-3, -3, 0]]
    example_points = testcp
    # Camera position
    camera_point = msh.model.occ.add_point(x, y, z, LC)
    
    # Establish vertices
    point_list = []
    
    for i, point_coords in enumerate(example_points):
        # point_name = f'point_{i}'
        point_list.append(msh.model.occ.add_point(point_coords[0], 
                                                  point_coords[1], 
                                                  point_coords[2], LC))
    
    # Create a list to fill with all perimeter lines
    perimeter_list = []
    
    # Create outlines of the object
    for i in range(len(example_points)):
        line_handle = msh.model.occ.add_line(
            point_list[i], point_list[(i+1)%len(example_points)])
        perimeter_list.append(line_handle)
        if i == len(example_points) - 1:
            break
    
    # Create lines from each object perimeter point to the camera origin point
    plin_list = []
    
    for i in range(len(example_points)):
        plin_handle = msh.model.occ.add_line(point_list[i], camera_point)
        plin_list.append(plin_handle)
        if i == len(example_points) - 1:
            break
    
    # Create curve loops adjacent to the camera origin
    loop_list = []
    
    for i in range(len(example_points)):
        loop_handle = msh.model.occ.add_curve_loop([-plin_list[i], 
                    perimeter_list[i], plin_list[(i+1)%len(example_points)]])
        loop_list.append(loop_handle)
        if i == len(example_points) - 1:
            break
    
    # Create perimeter loop
    perimeter_handles = [line_handle for line_handle in perimeter_list]
    perimeter_loop = msh.model.occ.add_curve_loop(perimeter_handles)
    
    surface_loop_list = []
    mesh_perimeter = msh.model.occ.add_plane_surface([perimeter_loop])
    surface_loop_list.append(mesh_perimeter)
    
    
    
    for i in range(len(example_points)):
        mesh_test = msh.model.occ.add_plane_surface([loop_list[i]])
        surface_loop_list.append(mesh_test)
        if i == len(example_points) - 1:
            break
    #return mesh_test, mesh_perimeter
    

    #4840 4839 4834
    sl = msh.model.occ.addSurfaceLoop(surface_loop_list)
    # shells.append(sl)
    v = msh.model.occ.addVolume([sl]) 
    
    return mesh_test,v

# Coordinate visualizer - optional
def coordinate_visualizer(aruco_scale):
    
    # Draw coordinate axes on the image
    axis_length = aruco_scale *.2  # Adjust the length of the coordinate axes as per your preference
    axis_points, _ = cv.projectPoints(axis_length * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                      rotation_vector, translation_vect, CMTX, DIST)
    
    # Draw the coordinate axes
    for i in range(4):
        cv.drawFrameAxes(image, CMTX, DIST, rotation_vector, translation_vect,
                          axis_length, 2)
    
    # Display the image with coordinate axes
    cv.namedWindow("Image with Coordinate Axes", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
    
    #draw in a small window / for debugging
    cv.aruco.drawDetectedMarkers(image, corners, ids)
    cv.imshow("Image with Coordinate Axes", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#optional
def matplotlib_visualizer(cam_world,unit_vector_list):
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*cam_world[:, 0],color='hotpink',
               label='Camera Position OG',marker = '2')
    
    aruco_points_list=[]
    for value in aruco_points.values():
        for point in value:
            aruco_points_list.append(point)
            #aruco_points_list.append([500,500,0])
            ax.scatter(*point, color="black", marker="x" ) 
    ax.scatter(*center_point, color="red", marker="o")  
     
    # Plot the ray as a line from the center_point to cam_world
    ax.plot([center_point[0], cam_world[0]],
        [center_point[1], cam_world[1]],
        [center_point[2], cam_world[2]],
        color='green', linestyle='--')
    line_length = np.linalg.norm(center_point - cam_world)
    print("Length of the line:", line_length)
    ax.elev = 45
    ax.azim = 45
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_aspect('equal', 'box')
    plt.show()
    
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define a common color for all lines
    line_color = 'b'  # You can change 'b' to any valid Matplotlib color

    # Plot the line between cam_world and unit vectors in unit_vector_list
    for unit_vector in unit_vector_list:
        # Create points for the line
        x_points = [cam_world[0, 0], cam_world[0, 0] +
                    unit_vector[0, 0]*line_length]
        y_points = [cam_world[1, 0], cam_world[1, 0] + 
                    unit_vector[1, 0]*line_length]
        z_points = [cam_world[2, 0], cam_world[2, 0] + 
                    unit_vector[2, 0]*line_length]

        # # Use the common color for all lines
        # ax.plot(x_points, y_points, z_points, color=line_color)
        
    for value in aruco_points.values():
        for point in value:
            aruco_points_list.append(point)
            #aruco_points_list.append([500,500,0])
            ax.scatter(*point, color="black", marker="x" ) 
    # Customize the plot as 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Lines between cam_world and unit vectors')
    ax.set_aspect('equal', 'box')
    ax.elev = 45
    ax.azim = -30
    # Show the plot
    plt.show()
        
    return ray_vector,x_points 

'''
Main
'''

# Return a list of image file paths that match the pattern
image_paths = glob.glob(os.path.join(folder_path, imagename_pattern))

# Begin processing loop
for index, image_path in enumerate(image_paths):
    
    image = cv.imread(image_path)
    image = resize_image(image, SCALING)
    print()
    print(f"Scanning image #:{index}")
    print(f"Image adress:{image_path}")
    print()
    # Extract marker data
    corners, ids  = detect_markers(image) #darkness, aruco_corner_darkness
    

    if ids is None:
        continue
    # Try to locate an acceptable contour
    try:
        contour_points= (find_contours(image_path,debug_mode)[::20]) 
    except:
        print('(',image_path,'):find_contours error')
        print()
        continue
    
    # Append all marker data and package for solvepnp
    reshaped_array_img, reshaped_array_obj = multi_pose_estimator(ids)
    # Solvepnp
    H,camera_position_pnp,rotation_matrx,translation_vect,rotation_vector = pnp_solver()
    # transformn to real world coordinate system
    unit_vector_list,cam_world  = dimensional_transforms_contours(contour_points)

    # CaLCulate the ray between the center_point and cam_world
    center_point = [[0], [0], [0]]
    ray_vector = cam_world - center_point
    
    # Establish far side projection plane (needed for gmsh)
    far_point = -(cam_world)
    far_plane_constant = ray_vector[0]*far_point[0]+ray_vector[1]*far_point[1]
    +ray_vector[2]*far_point[2]

    origin_point = [cam_world[0],cam_world[1],cam_world[2]]
    # Initialize a list to store intersection points
    intersection_point_list = []
    
    # Iterate through each unit vector
    for unit_vector in unit_vector_list:
        # Calculate the intersection point with the plane
        # The eqn the line is x = x0 + t * u, y = y0 + t * v, z = z0 + t * w
        # Solve for t using the plane equation and the equation of the line
        t = (far_plane_constant - ray_vector[0] * origin_point[0] - 
             ray_vector[1] * origin_point[1] - ray_vector[2] * origin_point[2])\
            / (ray_vector[0] * unit_vector[0] + ray_vector[1] * unit_vector[1]
            + ray_vector[2] * unit_vector[2])
    
        # Calculate the intersection point
        intersection_point = (origin_point[0] + t * unit_vector[0],
                              origin_point[1] + t * unit_vector[1], 
                              origin_point[2] + t * unit_vector[2])
    
        intersection_point_list.append(intersection_point)
       
 
    # if debug_mode == 3:
    #     tuple_list = []
    #     # Iterate through the arrays in the tuple
    #     for array in corners:
    #         # Iterate through the rows in the array
    #         for row in array[0]:
    #             # Extract the two values from each row and convert them into a tuple
    #             value_tuple = tuple(row)
    #             # Append the tuple to the tuple_list
    #             tuple_list.append(value_tuple)
    #     unit_vector_list_arucos,cam_world_arucos = dimensional_transforms_contours(tuple_list)
            
    #     ray_vector,x_points = matplotlib_visualizer(cam_world,unit_vector_list) 
    #     ray_vector_aruco,x_points_aruco = matplotlib_visualizer(cam_world_arucos,
    #                                                    unit_vector_list_arucos) 

    # coordinate_visualizer(aruco_scale)

  

    if index == 0: 
        print("first image!!!")
        _,v1 =polyhedron_obj_converter(intersection_point_list, cam_world)
        continue
    else:
        _,v =polyhedron_obj_converter(intersection_point_list, cam_world)
    
  
    v2 = msh.model.occ.intersect([(3,v1)],[(3,v)],removeObject= True,
                                        removeTool = True)
        
     

# # Create the relevant msh data structures from the msh model
msh.model.occ.synchronize()
# Set visibility options to hide points, lines, and 3D faces
msh.option.setNumber("General.Verbosity", 1)  # 1= Show all messages
msh.option.setNumber("Geometry.Points", 0)   
msh.option.setNumber("Geometry.Lines", 0)   
msh.option.setNumber("Mesh.SurfaceFaces", 0)      
msh.option.setNumber("Mesh.SurfaceEdges", 0)      
msh.option.setNumber("Mesh.VolumeFaces", 1)     
msh.option.setNumber("Mesh.VolumeEdges", 0)     


# #msh.model.mesh.Triangles(0)


# Generate mesh
msh.model.mesh.generate()

# Write mesh data
msh.write("GFG.msh")

# Run the graphical user interface event loop
msh.fltk.run()
#msh.hide(all) hide meshs
#msh.optimize_threshold

# Finalize the msh API
msh.finalize()

        