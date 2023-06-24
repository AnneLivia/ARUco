## ArUco using OpenCV

This project utilizes OpenCV to create an application that can detect ArUco markers and estimate their pose in an image.

## Code Flow
1. ArucoClass.py
    - This file contains a class with various methods:
      - `Detect corners`: This method detects the corners and ids of ArUco markers in an image.
      - `Draw markers`: This method draws the detected ArUco markers on the image, including the option to insert the marker ID text and draw a circle in the middle of the marker.
      - `Draw axes`: This method visualizes the orientation of the detected ArUco markers by drawing 3D axes.
      - `Get distance`: This method calculates the distance between the camera and the detected ArUco markers.
2. File app.py
    1. When running this program, you have the option to specify three arguments:
        - `arudict`: You can choose a specific ArUco dictionary to use. The default dictionary is `DICT_ARUCO_ORIGINAL`. For a list of available dictionaries, please refer to [OpenCV ArUco dictionary documentation](https://docs.opencv.org/4.x/d1/d21/aruco__dictionary_8hpp.html).
        - `calibpath`: If you want to visualize the orientation of the markers and calculate the distance, you need to provide the path to the camera calibration data.
        - `msz`: If you provide the camera calibration data path, you also need to specify the marker size in millimeters.

## Result image

The result image demonstrates the output of the application, showing the detected ArUco markers drawn on the image and the visualization of their orientation using 3D axes.