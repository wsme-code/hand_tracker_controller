## Hand Tracker Controller for robotic manipulator control
# resolve the function issue: https://github.com/google/mediapipe/issues/2818
# code mainly comes from the tutorial: https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
# mediapipe documentation: https://google.github.io/mediapipe/solutions/hands#python-solution-api
# color thresholding from: https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/

import cv2
import mediapipe as mp
import numpy as np
import time

# detect colors
# place marker on robot manipulator based on average of x and y values
# maybe something to ignore outliers
class manipulatorDetector():
    def __init__(self):
        self.img = 0

    def findJoints(self, img):
        # lists color boundaries for blue
        # each color is configured in openCV's BGR color space
        color_boundaries = [
            ([255, 251, 246], [236, 168, 0]),
        ]

        # detect colors in image
        for (lower, upper) in color_boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(self.img, lower, upper)
            output = cv2.bitwise_and(self.img, self.img, mask = mask)
            # show the images
            cv2.imshow("images", np.hstack([self.img, output]))
            cv2.waitKey(0)

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
    render_im_path = "gloved_hand2.jpg"

    img = cv2.imread(render_im_path)
    #print(img)
    cv2.imshow("Image", img)

    manipulator.findJoints(img)

    # img = detector.findHands(img)
    # lmlist = detector.findPosition(img)
    # # prints the coordinates of the Landmark with the ID passed to lmlist
    # if len(lmlist) != 0:
    #     print(lmlist[0])

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()