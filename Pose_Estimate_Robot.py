#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Vector3
import numpy as np
import math
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R



'''
Welcome to the Pose Estimation Program!
  
This program:
  - Performs pose estimation using an ArUco Marker.
  - Author: Mai Thanh Dat - Nguyen Phuc Bao Nguyen
 '''
  
# Initialize ROS node
rospy.init_node('pose_estimation_publisher', anonymous=True)

# ROS publisher for the /odom topic
pose_pub = rospy.Publisher('/odom', Vector3, queue_size=10)

def publish_pose(pose):
    position = Vector3()
    position.x = pose['x']
    position.y = pose['y']
    position.z = pose['yaw']
    pose_pub.publish(position)

# Load camera calibration data (intrinsic matrix and distortion coefficients)
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'

# Function to convert quaternion to Euler angles (roll, pitch, yaw)
def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z  # in radians

# Load camera calibration data
cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode('K').mat()
dist_coeffs = cv_file.getNode('D').mat()

# URL of the IP camera
ip_camera_url = 'http://192.168.1.8:8080/video'  # Replace with your camera IP address and port

# Initialize video capture with the IP camera URL
cap = cv2.VideoCapture(ip_camera_url)
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from IP camera")
    exit()

# Initialize the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Initialize parameters for ArUco detection
parameters = aruco.DetectorParameters()

marker_size = 0.1  # Define the size of the marker in meters

# Set the loop rate (e.g., 10 Hz)
rate = rospy.Rate(10)

# Read frames from the IP camera and process them
while not rospy.is_shutdown():
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    if not ret:
        print("Error: Could not read frame from IP camera")
        break

    # Detect ArUco markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))
    
    # If markers are detected, estimate their pose
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # Extract translation (x, y, z)
            transform_translation_x = tvecs[i][0][0]
            transform_translation_y = -tvecs[i][0][1]
            transform_translation_z = tvecs[i][0][2]

            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()

            # Convert quaternion to Euler angles (in radians)
            roll_x, pitch_y, yaw_z = euler_from_quaternion(
                quat[0], quat[1], quat[2], quat[3]
            )

            # Convert yaw angle to degrees
            yaw_z_deg = math.degrees(-yaw_z)  # Negate yaw_z to align with typical ROS conventions

            # Store values in the pose dictionary
            pose = {}
            pose['x'] = transform_translation_x
            pose['y'] = transform_translation_y
            pose['yaw'] = yaw_z_deg
            position = (float(pose['x']), float(pose['y']), float(pose['yaw']))

            # Publish the position and yaw angle
            publish_pose(pose)

            # Print the position and yaw angle
            print(f"Position (x, y): ({transform_translation_x}, {transform_translation_y})")
            print(f"Yaw angle: {yaw_z_deg} degrees")

            # Draw text on the frame
            cv2.putText(frame, f"Position (x, y): ({transform_translation_x:.2f}, {transform_translation_y:.2f})", (10, 30 + i * 60))
            cv2.putText(frame, f"Yaw angle: {yaw_z_deg:.2f} degrees", (10, 50 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    else:
        print("No ArUco markers found.")

    # Display the annotated frame
    cv2.imshow('IP Camera Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Sleep to maintain the loop rate
    rate.sleep()

# Release resources
cap.release()
cv2.destroyAllWindows()
