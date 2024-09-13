import numpy as np
import math
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import time

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
        
    return roll_x, pitch_y, yaw_z # in radians


# Load camera calibration data (intrinsic matrix and distortion coefficients)
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
# Load camera calibration data
cv_file = cv2.FileStorage(camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode('K').mat()
dist_coeffs = cv_file.getNode('D').mat()


# URL of the IP camera
ip_camera_url = 'http://192.168.100.6:8080/video'  # Replace with your camera IP address and port

# Initialize video capture with the IP camera URL
cap = cv2.VideoCapture(ip_camera_url)
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera")
    exit()

# Get the size of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Initialize parameters for ArUco detection
parameters = aruco.DetectorParameters()
# Define the size of the marker in meters
marker_size = 0.2  

# Create a named window for imshow
cv2.namedWindow('Camera Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Frame', width, height)

# Read frames from the camera and process them
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame from camera")
        break

    # Detect ArUco markers in the image
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # If markers are detected, estimate their pose
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        # Draw axes for each marker
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))
            # Extract translation (x, y, z)
            # Store the translation (i.e. position) information
            transform_translation_x = tvecs[i][0][0]
            transform_translation_y = -tvecs[i][0][1]
            transform_translation_z = tvecs[i][0][2]
            # Store the rotation information
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()   
            # Quaternion format     
            transform_rotation_x = quat[0] 
            transform_rotation_y = quat[1] 
            transform_rotation_z = quat[2] 
            transform_rotation_w = quat[3]
            # Euler angle format in radians
            roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                        transform_rotation_y, 
                                                        transform_rotation_z, 
                                                        transform_rotation_w)
            
            roll_x = math.degrees(roll_x)
            pitch_y = math.degrees(pitch_y)
            yaw_z = -math.degrees(yaw_z)
            
            cv2.putText(frame, f"Position (x, y): ({transform_translation_x:.2f}, {transform_translation_y:.2f})", (10, 30 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Yaw angle: {yaw_z:.2f} degrees", (10, 50 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    else:
        print("No ArUco markers found.")

    # Display the image with drawn axes
    # Draw coordinate axes (red for X, green for Y)
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 2)  # Red vertical line
    cv2.line(frame, (0, height // 2), (width, height // 2), (0, 0, 255), 2)  # Green horizontal line
    cv2.imshow('Camera Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
