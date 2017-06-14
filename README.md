# DolDetect: A Real-time Dolphins Detection Engine
This is a c++ engine for real time dolphin detection, providing both .so and .dll shared library.

### Dependencies
* opencv 2.4.13

### Build Requirements
This project can be built on linux and windows paltform, including shared library and some example codes. 
* Linux
  * system: ubuntu 14.04
  * cmake: 3.5.1
* Windows
  * system: window 10
  * ide: vs2013

### Algorithm Introduction
This project uses some non-deep methods, which provides some real time performace.
Here is the framework of our algorithm:
1. SURF: detect and extract keypoints of image
2. Visual vocabulary: filter keypoints
3. DBscan: cluster keypoints to get positions of potential objects
4. Generate candidate bounding-boxes
5. HOG: extract features for candidate bounding-boxes
6. SVM: classify candidate bounding-boxes

### Demo Illustration
![Alt text](/demo.jpg)
