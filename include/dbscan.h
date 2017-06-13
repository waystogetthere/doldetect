#pragma once
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"

#define DIST_L2( a, b ) \
	sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y))

#define DIST_L2_MAT( a, b ) \
	norm(a, b, CV_L2)

struct db_point {
	enum { NOISE = 1, BORDER = 2, CORE = 3 };
	float x, y;     // coordinate
	cv::Mat feature;// (used in building vocabulary)
	int id;         // original index
	int	cluster_id; // cluster index
	int type;       // point type
	int pts;        // num of reachable points
	db_point() : x(0), y(0), id(0), cluster_id(0), type(NOISE), pts(1) {}
	db_point(float x_, float y_, int id_) : x(x_), y(y_), id(id_), cluster_id(id_), type(NOISE), pts(1) {}
	db_point(cv::Mat feature_, int id_) : feature(feature_), id(id_), cluster_id(id_), type(NOISE), pts(1) {}
};

void dbscan(std::vector<db_point>& dataset, std::vector<int>& clusters,
	float eps, int minpts, int datatype = 1);