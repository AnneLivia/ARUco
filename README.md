## ArUco using OpenCV

This project utilizes OpenCV to create an application that can detect ArUco markers and estimate their pose in an image.

## Code Flow
1. ArucoClass.py
    - This file contains a class with various methods:
      - `Detect corners`: This method detects the corners and ids of ArUco markers in an image.
      - `Draw markers`: This method draws the detected ArUco markers on the image, including the option to insert the marker ID text and draw a circle in the middle of the marker.
      - `Draw axes`: This method visualizes the orientation of the detected ArUco markers by drawing 3D axes.
      - `Get distance`: This method calculates the distance between the camera and the detected ArUco markers.
      - `Image augmentation`: This method overlays an image on top of the detected marker.
2. File app.py
    1. When running this program, you have the option to specify three arguments:
        - `arudict`: You can choose a specific ArUco dictionary to use. The default dictionary is `DICT_ARUCO_ORIGINAL`. For a list of available dictionaries, please refer to [OpenCV ArUco dictionary documentation](https://docs.opencv.org/4.x/d1/d21/aruco__dictionary_8hpp.html).
        - `calibpath`: If you want to visualize the orientation of the markers and calculate the distance, you need to provide the path to the camera calibration data.
        - `msz`: If you provide the camera calibration data path, you also need to specify the marker size in millimeters.
        - `imgar`: Provide a path to the image you wish to augment
        - `viar`: Provide a path to the video you wish to augment

## Demo

The following GIFs demonstrate the output of the application:

1. **Visualization of marker orientation using 3D axes:**
   
    <img src="https://github.com/AnneLivia/ARUco/assets/31932673/1024f7ad-5257-43d5-8340-8642c1903b58" width="50%"/>
3. **Image Augmentation**
   
    <img src="https://github.com/AnneLivia/ARUco/assets/31932673/0c01cae3-f6c4-472a-bd23-e1533f488eef" width="50%"/>
5. **Video Augmentation**
   
    <img src="https://github.com/AnneLivia/ARUco/assets/31932673/f7fb1d6b-edd6-4875-993e-fe83332f6c5e" width="50%"/>
