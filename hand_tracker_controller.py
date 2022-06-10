## Hand Tracker Controller for robotic manipulator control
# resolve the function issue: https://github.com/google/mediapipe/issues/2818
# code mainly comes from the tutorial: https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
# mediapipe documentation: https://google.github.io/mediapipe/solutions/hands#python-solution-api
# color thresholding from: https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/

import cv2
from matplotlib.colors import hsv_to_rgb
import mediapipe as mp
import numpy as np
import time

# detect colors
# place marker on robot manipulator based on average of x and y values
# maybe something to ignore outliers
class manipulatorDetector():
    def __init__(self):
        self.img = 0

    def findBlueJoint(self, img):
        blue = np.uint8([[[255,250,240 ]]])
        hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
        print( hsv_blue )

        # Reading the image
        img = cv2.imread('hand_tracker_controller/gloved_hand2.jpg')

        # convert to hsv colorspace
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower bound and upper bound for Green color
        # these values were determined using the color picker at the beginning of the code
        # with some tolerance
        lower_bound = np.array([95,  5, 210])   
        upper_bound = np.array([105, 255, 255])

        # find the colors within the boundaries
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        #define kernel size  
        kernel = np.ones((7,7),np.uint8)

        # Remove unnecessary noise from mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Segment only the detected region
        segmented_img = cv2.bitwise_and(img, img, mask=mask)

                # Find contours from the mask
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

        # Draw contour on original image
        output = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

        return segmented_img

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelCom = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelCom, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

class actuationSensor():
    def __init__(self):
        self.isAvgObtained = 0
        self.i = 0
        self.avgMaxActuationDist = 0

    ## get the maximum actuation distance
    # checks if first iteration and gets the starting time of the recording
    # eventually take an argument for what finger to detect
    def getMaxActuationDistance(self, lmlist):
        if (self.i == 0):
            self.startTime = time.time()
            self.maxActuationDistArray = np.array([])

        # checks if 5 seconds have passed
        if ((time.time() - self.startTime) < 5):
            self.maxActuationDistArray = np.append(self.maxActuationDistArray, abs(lmlist[8][2] - lmlist[5][2]))
            print(str(self.maxActuationDistArray[self.i]))

        else:
            self.avgMaxActuationDist = np.average(self.maxActuationDistArray)
            print(str(self.avgMaxActuationDist))
            self.isAvgObtained = 1
        self.i += 1

def main():
    manipulator = manipulatorDetector()
    detector = handDetector()
    hand = actuationSensor()
    render_im_path = "hand_tracker_controller/gloved_hand2.jpg"

    img = cv2.imread(render_im_path)
    #print(img)
    cv2.imshow("Image", img)

    segmented_img = manipulator.findBlueJoint(img)

    cv2.imshow("segmented", segmented_img)
    cv2.waitKey(0)
    
    # img = detector.findHands(img)
    # lmlist = detector.findPosition(img)
    # # prints the coordinates of the Landmark with the ID passed to lmlist
    # if len(lmlist) != 0:
    #     print(lmlist[0])

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()