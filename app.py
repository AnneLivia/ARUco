from ArucoClass import Aruco
import argparse
import cv2
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('-arudict', default='DICT_ARUCO_ORIGINAL', help='Inform the Aruco dictionary to be used')
parse.add_argument('-calibpath', help='Inform the path for the calibration data')
parse.add_argument('-msz', help='Inform the size of the marker in millimeters')
parse.add_argument('-imgar', help='Inform the path to the image to be augmented')

args = parse.parse_args()

if args.calibpath and not args.msz:
    parse.error('Inform the size of the marker in millimeters')

try:
    # creating the aruco detector
    aruco = Aruco(args.arudict)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break;

        # detecting markers (4 corners, the id and rejected candidates (useful for debug), of the marker is returned) 
        corners, ids, _ = aruco.detectCornersAndIds(frame)
        
        if corners:
            # running through all of the corners and ids
            # I can basically use cv2.aruco.drawDetectedMarkers(image, corners, ids) to draw it.
            for corner, id, i in zip(corners, ids, range(len(ids))):
                aruco.drawDetectedCornersOnMarkers(frame, corner, id)
                # if the path for the camera calibration data was passed, then we can show the axis (x, y and z)
                if args.calibpath:
                    cameraCalibrationData = np.load(args.calibpath)
                     # I only need those two camera calibration data
                    camMatrix, distCoeff, _, _ = [cameraCalibrationData[file] for file in cameraCalibrationData]
    
                    markerSize = int(args.msz)

                    cameraCalibrationDataDict = {"matrix" : camMatrix, "distCoeff": distCoeff}

                    rVec, tVec = aruco.getAxisVectors(corners, markerSize, cameraCalibrationDataDict)

                    aruco.drawAxisOnMarkers(frame, {"rVec": rVec[i], "tVec": tVec[i]}, cameraCalibrationDataDict, True, corner)
                if args.imgar:
                    augImage = cv2.imread(args.imgar)
                    aruco.createImageAugmentation(augImage, frame, corner)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) >= 0:
            break;
    
    cap.release()
    cv2.destroyAllWindows()

except ValueError:
    print('You must inform a valid Aruco dictionary. ' + 
          'Please refer to https://docs.opencv.org/4.x/d1/d21/aruco__dictionary_8hpp.html.')