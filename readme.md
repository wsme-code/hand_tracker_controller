# Hand Tracker Controller for robotic manipulator control
Computer vision for robotic manipulartor control

- code mainly comes from the tutorial: https://www.analyticsvidhya.com/blog/2021/07/building-a-hand-tracking-system-using-opencv/
- mediapipe documentation: https://google.github.io/mediapipe/getting_started/hello_world_cpp.html

Notes
===============================
https://github.com/google/mediapipe/issues/2818
An issue would occur with the code from the analyticsvidhya code where an incorrect argument was being passed to the `mpHands.Hands` function. To resolve, the `modelCom` or model complexity parameter was added
