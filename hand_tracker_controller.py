## Hand Tracker Controller for robotic manipulator control
# resolve the function issue: https://github.com/google/mediapipe/issues/2818
# code mainly comes from the tutorial: https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
# mediapipe documentation: https://google.github.io/mediapipe/solutions/hands#python-solution-api

import cv2
import mediapipe as mp
import numpy as np
import time
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
        
    def findHands(self,img, draw = True):
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
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    hand = actuationSensor()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        # prints the coordinates of the Landmark with the ID passed to lmlist
        if len(lmlist) != 0:
            print(lmlist[0])

        ## start of distance sensing code
        # sense right hand index finger contraction
        if len(lmlist) != 0:
            hand.getMaxActuationDistance(lmlist)
            if (hand.isAvgObtained == 1):
                contraction = abs(lmlist[8][2] - lmlist[5][2]) / hand.avgMaxActuationDist
                print("Contraction distance is: " + str(contraction))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()