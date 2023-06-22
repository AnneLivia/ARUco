import cv2
import argparse;

# ARuco is a library that contains a lot of pre-defined markers that can be used to estimate pose,
# calibrate camera, construct augmented reality app, and so on. Those fiducial markers, are binary images
# that contains a specific code that identifies it. An ArUco marker is a synthetic square marker composed by 
# a wide black border and an inner binary matrix which determines its identifier (id).
# Aruco allows to create custom markers if needed. 
# fiducial is a term that indicates object or referent points utilized as a reference point to help on measurements,
# positioning and pose estimation of other elements.
# aruco has a dictionary with a lot of pre-defined markers.

parser = argparse.ArgumentParser()
parser.add_argument('-artag', default='DICT_ARUCO_ORIGINAL', help='Pass a valid ARUCO tag in order to create a detector')
args = parser.parse_args()

# loading webcam
cap = cv2.VideoCapture(0)

# select ArUCO dictionary to be used
# dictionary list: https://docs.opencv.org/4.x/de/d67/group__objdetect__aruco.html#gga4e13135a118f497c6172311d601ce00da6eb1a3e9c94c7123d8b1904a57193f16
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

arDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args.artag]);
arParams = cv2.aruco.DetectorParameters();
arDetector = cv2.aruco.ArucoDetector(arDict, arParams)

def getPoints(corner):
    topLeft = (int(corner[0][0]), int(corner[0][1]));
    topRight = (int(corner[1][0]), int(corner[1][1]));
    bottomRight = (int(corner[2][0]), int(corner[2][1]));
    bottomLeft = (int(corner[3][0]), int(corner[3][1]));

    return topLeft, topRight, bottomLeft, bottomRight

def drawDetectedLinesAroundMarkers(frame, corners, ids):
    # running through all corners and ids
    for corner, id in zip(corners, ids):
        # corners are in this order: top left, top right, bottom right, and bottom left
        corner = corner.reshape(4, 2)
        # drawing lines to build the square around the marker
        topLeft, topRight, bottomLeft, bottomRight = getPoints(corner)

        # the connection must be: topLeft->topRight, topRight->bottomRight, bottomRight->bottomLeft, bottomLeft->topLeft
        cv2.line(frame, topLeft, topRight, (0, 0, 255), 2, cv2.LINE_AA);
        cv2.line(frame, topRight, bottomRight, (0, 0, 255), 2, cv2.LINE_AA);
        cv2.line(frame, bottomRight, bottomLeft, (0, 0, 255), 2, cv2.LINE_AA);
        cv2.line(frame, bottomLeft, topLeft, (0, 0, 255), 2, cv2.LINE_AA);

        # drawing circle aroung center
        xCenter = int((topLeft[0] + topRight[0]) / 2); # horizontal (x) center (topleft to topRight)
        yCenter = int((topLeft[1] + bottomLeft[1]) / 2); # vertical (y) center (topLeft to bottomLeft)
        cv2.circle(frame, (xCenter, yCenter), 6, (0, 255, 0), -1, cv2.LINE_AA)

        # insert the id in the frame
        cv2.putText(frame, str(id), (
            topLeft[0], topLeft[1]
        ), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read();

    # detecting markers (4 corners, the id and rejected candidates (useful for debug), of the marker is returned) 
    corners, ids, _ = arDetector.detectMarkers(frame);
   
    if (corners):
        drawDetectedLinesAroundMarkers(frame, corners, ids)

    if not ret:
        break;

    cv2.imshow('VideoStream', frame)

    if cv2.waitKey(1) >= 0:
        break;

cv2.destroyAllWindows()
