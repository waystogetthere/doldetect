#include "doldetect_dll.h"
#include "dbscan.h"
#include "opencv2/nonfree/nonfree.hpp"

#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;
using namespace cv;

//////////////// Global ////////////////
static SurfDescriptorExtractor extractor;
static HOGDescriptor hog(Size(96, 32), Size(16, 16), Size(16, 16), Size(8, 8), 9);


//////////////// Visual Vocabulary ////////////////
class Templates
{
public:
	static Templates& Instance() {  // singleton
		static Templates instance;
		return instance;
	}
	void genTemplates();
	void loadDictionary();
	int predict(vector<float>& hog_feat);
	Mat descriptors_;  // vocabulary
	vector<int> centers_;
	CvSVM svm_;

private:
	Templates() {};
	~Templates() {};
	Templates(Templates const&);      // don't Implement
	void operator=(Templates const&); // don't Implement
};

void Templates::genTemplates()
{
	FileStorage fs("templates.xml", FileStorage::WRITE);
	vector<KeyPoint> keypoints;
	Mat descriptors, img;
	string key = "template_00";
	char filepath[20];

	int cnt = 0;
	for (int i = 1; i <= 24; ++i) {
		// step 1: read image
#ifdef __linux__
		sprintf(filepath, "pos/%d.jpg", i);
#else
		sprintf_s(filepath, "pos/%d.jpg", i);
#endif
		img = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
		if (!img.data) {
			printf("fail to read: %s\n", filepath);
			continue;
		}

		// step 2: detect and extract
		SurfFeatureDetector detector(800);
		detector.detect(img, keypoints);
		extractor.compute(img, keypoints, descriptors);

		// step 3: display keypoints
		for (size_t j = 0; j < keypoints.size(); ++j)
			circle(img, keypoints[j].pt, 4, Scalar(0, 0, 0), -1);
		imshow("template", img);
		char ch = waitKey();
		if (ch != 27) {
			// step 4: save descriptor
			key[9] = '0' + cnt / 10;
			key[10] = '0' + cnt % 10;
			fs << key << descriptors;
			++cnt;
		}
	}
	fs << "total" << cnt;
	fs.release();
}

void Templates::loadDictionary()
{
	int cnt;
	// step 0: check whether templates are loaded
	if (descriptors_.data)
		return;

	// step 1: read dictionary file
	FileStorage fs("dictionary.xml", FileStorage::READ);
	string key = "item_000";
	vector<Mat> descriptors;
	fs["desc_tot"] >> cnt;
	for (int i = 0; i < cnt; ++i) {
		key[5] = '0' + i / 100;
		key[6] = '0' + (i % 100) / 10;
		key[7] = '0' + i % 10;
		Mat tmp;
		fs[key] >> tmp;
		if (tmp.data)
			descriptors.push_back(tmp);
	}
	if (descriptors.empty()) {
		printf("invalid dictionary xml file\n");
		return;
	}

	// step 2: copy into descriptors_
	Mat tmp(descriptors.size(), descriptors[0].cols, CV_32F);
	for (size_t i = 0; i < descriptors.size(); ++i) {
		Mat row = tmp.rowRange(i, i + 1);
		descriptors[i].copyTo(row);
	}
	descriptors_ = tmp;

	// step 3: load center indices
	key = "center_00";
	fs["center_tot"] >> cnt;
	int center_id;
	for (int i = 0; i < cnt; ++i) {
		key[7] = '0' + i / 10;
		key[8] = '0' + i % 10;
		fs[key] >> center_id;
		centers_.push_back(center_id);
	}
	fs.release();

	// step 4: load svm model
	svm_.load("svm.xml");
}

int Templates::predict(vector<float>& hog_feat)
{
	Mat feat(1, hog_feat.size(), CV_32F);
	float *ptr = feat.ptr<float>(0);
	for (size_t i = 0; i < hog_feat.size(); ++i)
		ptr[i] = hog_feat[i];
	return int(svm_.predict(feat));
}

//////////////// Detect ////////////////
void doldetect(Mat& img, vector<Rect>& bboxes,
	int minHessian, float minDist, float eps, int minPts, int maxPts, bool use_svm)
{
	// =========== static ============
	static vector<Rect> pre_bboxes;
	static vector<int> pre_cnt;
	// ===============================
	bboxes.resize(0);

	if (!img.data)
		return;

	if (img.cols == 0 && img.rows == 0)
		return;

	// step 1: load templates
	Templates::Instance().loadDictionary();

	vector<KeyPoint> keypoints;
	Mat descriptor;
	Mat gray = img.clone();
	if (gray.channels() == 3)
		cvtColor(gray, gray, CV_RGB2GRAY);

	// step 2: detect & extract
	SurfFeatureDetector detector(minHessian);
	detector.detect(gray, keypoints);
	extractor.compute(gray, keypoints, descriptor);

	// step 3: filtering by dictionary
	Mat& descriptors = Templates::Instance().descriptors_;
	vector<int>& centers = Templates::Instance().centers_;
	double minn, dist;
	int minid, bboxes_cnt = 0;
	vector<int> candidate_points, indices;
	vector<int> pt_cnt(pre_bboxes.size(), 0);
	vector<double> boxes_center(2 * pre_bboxes.size(), 0);

	for (int i = 0; i < descriptor.rows; ++i) {
		Mat row_a = descriptor.rowRange(i, i + 1);
		minn = norm(row_a, descriptors.rowRange(0, 1));
		minid = 0;
		for (int j = 1; j < descriptors.rows; ++j) {
			dist = norm(row_a, descriptors.rowRange(j, j + 1), CV_L2);
			if (dist < minn) {
				minn = dist;
				minid = j;
			}
		}

		bool ok = false;
		for (size_t j = 0; j < centers.size(); ++j)
		if (minid == centers[j]) {
			ok = true;
			break;
		}

		if (ok && minn < minDist) {
			bool selected = false;
			if (!pre_bboxes.empty()) {
				// check whether current point is contained by an existing boxes
				for (size_t j = 0; j < pre_bboxes.size(); ++j) {
					if (!pre_cnt[j])
						continue;
					if (keypoints[i].pt.x >= pre_bboxes[j].x &&
						keypoints[i].pt.x < (pre_bboxes[j].x + pre_bboxes[j].width) &&
						keypoints[i].pt.y >= pre_bboxes[j].y &&
						keypoints[j].pt.y < (pre_bboxes[j].y + pre_bboxes[j].height)) {
						boxes_center[j * 2] += keypoints[i].pt.x;
						boxes_center[j * 2 + 1] += keypoints[i].pt.y;
						++pt_cnt[j];
						if (pt_cnt[j] == 1)
							++ bboxes_cnt;
						selected = true;
						break;
					}
				}
			}
			if (!selected) {
				candidate_points.push_back(i);
				indices.push_back(minid);
			}
		}
	}

	if (candidate_points.empty() && !bboxes_cnt)
		return;

	vector<int> clusters;
	vector<db_point> dataset;
	for (size_t i = 0; i < candidate_points.size(); ++i) {
		int &id = candidate_points[i];
		dataset.push_back(db_point(keypoints[id].pt.x, keypoints[id].pt.y, i));
	}

	// step 4: dbscan
	dbscan(dataset, clusters, eps, minPts);
	if (!clusters[0] && !bboxes_cnt)
		return;

	// step 5: filtering
	int minH, maxH, minW, maxW, width, height, candidate_cnt = 0;
	double half_width, half_height;
	vector<int> candidate_boxes;
	if (clusters[0]) {
		int cnt = 0;
		size_t cur = 1;
		while (cur < clusters.size()) {
			int num = clusters[cur++];
			int type_0 = indices[clusters[cur]];
			bool ok = false;
			for (size_t i = cur + 1; i < cur + num; ++i) {
				if (indices[clusters[i]] != type_0) {
					// candidate must contains at least 2 typesof point
					ok = num < maxPts;
					break;
				}
			}
			if (ok) {
				int id = candidate_points[clusters[cur]];
				minH = maxH = int(keypoints[id].pt.y);
				minW = maxW = int(keypoints[id].pt.x);
				for (size_t j = cur + 1; j < cur + num; ++j) {
					id = candidate_points[clusters[j]];
					if (keypoints[id].pt.y < minH) minH = int(keypoints[id].pt.y);
					if (keypoints[id].pt.y > maxH) maxH = int(keypoints[id].pt.y);
					if (keypoints[id].pt.x < minW) minW = int(keypoints[id].pt.x);
					if (keypoints[id].pt.x > maxW) maxW = int(keypoints[id].pt.x);
				}
				width = maxW - minW;
				height = maxH - minH;
				if (max(width, height) > 4 * min(width, height)) {
					// current cluster is too 'slim'
					cur += num;
					++cnt;
					continue;
				}
				if (3 * height > width) {  // stretch width (move minW & maxW)
					if (9 * height > 8 * width) {
						half_width = 4.0 * width;
						half_height = 4.0 * width / 3.0;
					} else {
						half_width = height * 4.5;
						half_height = height * 1.5;
					}
				} else { // stretch height (move minH & maxH)
					if (width > 8 * height) {
						half_width = 12.0 * height;
						half_height = 4.0 * height;
					} else {
						half_width = width * 1.5;
						half_height = width * 0.5;
					}
				}
				minW = max(0, int(minW - half_width + width * 0.5));
				maxW = min(img.cols, int(maxW + half_width - width * 0.5));
				minH = max(0, int(minH - half_height + height * 0.5));
				maxH = min(img.rows, int(maxH + half_height - height * 0.5));
				candidate_boxes.push_back(minW);
				candidate_boxes.push_back(minH);
				candidate_boxes.push_back(maxW - minW);
				candidate_boxes.push_back(maxH - minH);
				++ candidate_cnt;
			}
			++ cnt;
			cur += num;
		}
	}

	// step 6: moving preivous bboxes
	if (bboxes_cnt) {
		int center_x, center_y, bboxes_w, bboxes_h, new_x, new_y;
		for (size_t i = 0; i < pt_cnt.size(); ++i)
		if (pt_cnt[i]) {
			center_x = int(boxes_center[i * 2] / pt_cnt[i]);
			center_y = int(boxes_center[i * 2 + 1] / pt_cnt[i]);
			bboxes_w = pre_bboxes[i].width;
			bboxes_h = pre_bboxes[i].height;
			new_x = max(0, center_x - bboxes_w / 2);
			new_y = max(0, center_y - bboxes_h / 2);
			bboxes_w = min(img.cols - new_x, bboxes_w);
			bboxes_h = min(img.rows - new_y, bboxes_h);
			candidate_boxes.push_back(int(0.9 * pre_bboxes[i].x + 0.1 * new_x));
			candidate_boxes.push_back(int(0.9 * pre_bboxes[i].y + 0.1 * new_y));
			candidate_boxes.push_back(bboxes_w);
			candidate_boxes.push_back(bboxes_h);
			++candidate_cnt;
			pre_cnt[i] = 0;
		}
	}
	if (!candidate_cnt)
		return;

	// step 7: classifiy (hog + svm)
	vector<float> hog_feat;
	for (int i = 0; i < candidate_cnt; ++i) {
		Rect rect(candidate_boxes[i * 4], candidate_boxes[i * 4 + 1], candidate_boxes[i * 4 + 2], candidate_boxes[i * 4 + 3]);
		Mat roi = gray(rect);
		resize(roi, roi, Size(96, 32), (0, 0), (0, 0), cv::INTER_LINEAR);
		hog.compute(roi, hog_feat, Size(96, 32), Size(0, 0));
		if (!use_svm || Templates::Instance().predict(hog_feat) == 1) {
			bboxes.push_back(rect);
		}
	}

	// step 8: update pre_bboxes
	for (size_t i = 0; i < pre_cnt.size(); ++i)
		pre_cnt[i] = max(0, pre_cnt[i] - 1);
	for (size_t i = 0; i < bboxes.size(); ++i) {
		size_t j = 0;
		for (; j < pre_cnt.size(); ++j)
		if (pre_cnt[j] == 0) {
			pre_bboxes[j] = bboxes[i];
			pre_cnt[j] = 5;
			break;
		}
		if (j == pre_cnt.size()) {
			pre_bboxes.push_back(bboxes[i]);
			pre_cnt.push_back(5);
		}
	}
}

void genTemplates()
{
	Templates::Instance().genTemplates();
}
