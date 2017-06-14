# DolDetect: A Real-time Dolphins Detection
This is a c++ engine for real time dolphin detection, providing both .so and .dll shared library.

# Dependencies
* opencv 2.4.13

# Build Requirements
This project can be built on linux and windows paltform, including shared library and some example codes.
## Linux
* system: ubuntu 14.04
* cmake: 3.5.1
## Windows
* system: window 10
* ide: vs2013

# Algorithm Introduction
This project uses some non-deep methods, which provides some real time performace.
Here is the framework of our algorithm:
1. Use SURF to detect and extract keypoints of image
2. Filter keypoints with a pretrained visual vocabulary
3. Cluster keypoints with dbscan
4. Generate candidate bounding-boxes
5. Extract HOG features for candidate bounding-boxes
6. Classify candidate bounding-boxes with a linear svm

# Demo Illustration
![Alt text](/demo.jpg)
