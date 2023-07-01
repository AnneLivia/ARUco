import cv2
import numpy as np
# ARuco is a library that contains a lot of pre-defined markers that can be used to estimate pose,
# calibrate camera, construct augmented reality app, and so on. Those fiducial markers, are binary images
# that contains a specific code that identifies it. An ArUco marker is a synthetic square marker composed by 
# a wide black border and an inner binary matrix which determines its identifier (id).
# Aruco allows to create custom markers if needed. 
# fiducial is a term that indicates object or referent points utilized as a reference point to help on measurements,
# positioning and pose estimation of other elements.
# aruco has a dictionary with a lot of pre-defined markers.
class Aruco:
    #  A set of predefined dictionaries of markers
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    def __init__(self, arucoMarkerDict):
        if (arucoMarkerDict not in self.ARUCO_DICT):
            raise ValueError
        
        # if the informed aruco marked exists that it can be used
        self.arucoMarkerCode = self.ARUCO_DICT[arucoMarkerDict]
        
        # need to get the aruco detector params
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoDict = cv2.aruco.getPredefinedDictionary(self.arucoMarkerCode)

        # creating the detector
        self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

    def detectCornersAndIds(self, image):
        corners, ids, rejected  = self.detector.detectMarkers(image)
        return corners, ids, rejected
    
    def getBoundingBox(self, corner):
        # it's possible to reshape: corner.reshape(4, 2), and avoid three indexes
        # corners are in this order: top left, top right, bottom right, and bottom left
        topLeft = (int(corner[0][0][0]), int(corner[0][0][1]))
        topRight = (int(corner[0][1][0]), int(corner[0][1][1]))
        bottomRight = (int(corner[0][2][0]), int(corner[0][2][1]))
        bottomLeft = (int(corner[0][3][0]), int(corner[0][3][1]))
        return topLeft, topRight, bottomRight, bottomLeft
    
    def drawFourLines(self, boundingBox, image):
        # drawing top-left -> top-right, then top-right -> bottom-right, 
        # then bottom-right to bottom-left, bottom-left to top-left
        for i in range(1, 5):
            # getting previous point
            pt1 = boundingBox[i - 1]
            # if it's the 4th iteration, then it's necessary to draw a line bottom-left (3) to top-left (0)
            pt2 = boundingBox[0 if i == 4 else i]
            
            cv2.line(image, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

    def drawCircleInTheCenterOfTheMarker(self, topLeft, bottomRight, image):
        # horizontal (x) center and vertical (y) center 
        xCenter = int((topLeft[0] + bottomRight[0]) / 2)
        yCenter = int((topLeft[1] + bottomRight[1]) / 2)
        # -1 filled circle
        cv2.circle(image, (xCenter, yCenter), 8, (0, 255, 0), -1, cv2.LINE_AA) 

    def insertMarkerID(self, point, id, image):
        cv2.putText(image, str(id), point, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # to draw the square and the insert the id around the marker
    def drawDetectedCornersOnMarkers(self, image, corner, id, drawCircle=True, insertIdText=True):
        tl, tr, br, bl = self.getBoundingBox(corner)
        # drawing lines to build the square around the marker
        self.drawFourLines([tl, tr, br, bl], image)
        if drawCircle:
            # drawing circle aroung center
            self.drawCircleInTheCenterOfTheMarker(tl, br, image)
        if insertIdText:
            # insert the id in the frame
            self.insertMarkerID(tl, id, image)

    def getAxisVectors(self, corners, markerSize, cameraCalibrationData):
        rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 
            markerSize, 
            cameraCalibrationData["matrix"], 
            cameraCalibrationData["distCoeff"],
        )
        return rVec, tVec
    
    # to draw the x, y and z axis (draw the pose of the marker)
    def drawAxisOnMarkers(self, image, axisData, cameraCalibrationData, showDistance, corner):
        cv2.drawFrameAxes(
            image, 
            cameraCalibrationData["matrix"], 
            cameraCalibrationData["distCoeff"], 
            axisData["rVec"], 
            axisData["tVec"], 
            length=5, 
            thickness=5)
        if showDistance:
            x, y, z = self.getAxisDistance(axisData)
            cv2.putText(image, 
                        f'x: {str(round(x, 2))}, y: {str(round(y, 2))}, z: {str(round(z, 2))}', 
                        self.getBoundingBox(corner)[1], 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        0.6, 
                        (0, 0, 255), 
                        1, 
                        cv2.LINE_AA)
            

    def getAxisDistance(self, axisData):
        # example format: [[ 34.15974605 -16.51720873 332.43917878]]
        x, y, z = axisData["tVec"][0]
        return x, y, z
    
    # this method is going to display an image over the aruco marker that is being caputured
    # through the webcam, if a specific id is encountered
    def createImageAugmentation(self, augmentedImage, frame, corners):

        imageHeight, imageWidth = augmentedImage.shape[:2]
        frameHeight, frameWidth = frame.shape[:2]

        # top-left, top-right, bottom-right and bottom-left, for the image to be augmented
        imagePoints = np.array([[0, 0], [imageWidth, 0], [imageWidth, imageHeight], [0, imageHeight]])
        # top-left, top-right, bottom-right and bottom-left, of the marker
        markerPoints = corners

        # find the homography (Perspective Transformation). A Homography is a geometric transformation 
        # ( a 3×3 matrix ) that maps points from one image to the corresponding points in another image.  
        # In augmented reality, it is used to align virtual objects in a camera-captured scene, 
        # making them appear part of the real world. This is achieved by finding keypoint matches 
        # between the virtual object image and the camera image. Based on these matches, the homography 
        # is calculated, allowing the virtual object's position and perspective to match the real scene. 
        # This makes the virtual object appear on the same plane or move with the camera
        H, _ = cv2.findHomography(srcPoints=imagePoints, dstPoints=markerPoints)

        # next, it's necessary to warp the perspective
        warpImage = cv2.warpPerspective(augmentedImage, H, (frameWidth, frameHeight))
        
        # need to create a mask of the frame (the augmented image is going to be 1's, the rest is 0's)
        mask = np.zeros((frameHeight, frameWidth), dtype=np.uint8)

        # creating a mask, where the region of the marker is 1. Need to squeeze the points, 
        # because fillConvexPoly requires the dimensions to be (4, 2), it was (1, 4, 2). 
        # squeeze removes the additional channel (axes of length one)
        cv2.fillConvexPoly(mask, np.squeeze(markerPoints).astype(np.int32), 255)

        # inserting the image on the frame
        cv2.bitwise_and(warpImage, warpImage, frame, mask=mask)

    def createVideoAugmentation(self, augVideo, frame, corner):
        self.createImageAugmentation(augVideo, frame, corner)

