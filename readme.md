# Hand Tracker Controller for Robotic Manipulator Control
Computer vision for robotic manipulartor control

- code mainly comes from the tutorial: https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
- mediapipe documentation: https://google.github.io/mediapipe/solutions/hands#python-solution-api

Notes
===============================
5/29/2022
Issue with protocol buffer after a google update on mediapipe. Resolved with `pip install --upgrade protobuf==3.20.0`

Before 5/29/2022
https://github.com/google/mediapipe/issues/2818 <br>
An issue would occur with the code from the analyticsvidhya tutorial where an incorrect argument was being passed to the `mpHands.Hands` function. To resolve, the `modelCom` or model complexity parameter was added
