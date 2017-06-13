#include "doldetect.h"
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;
int main()
{
	VideoCapture cap;
	cap.open("8.avi");
	if (!cap.isOpened()) {
		cout << "fail to open video" << endl;
		return 0;
	}
	Mat frame, roi;
	vector<Rect> bboxes;
	int frame_id = 0, cnt = 0;
	while (cap.read(frame)) {
		if (!frame.data) {
			break;
		}
		resize(frame, frame, Size(960, 540), (0, 0), (0, 0), cv::INTER_LINEAR);
		doldetect(frame, bboxes, 400, 0.5, 40, 3, 6, true);
		cout << "[frame " << frame_id << "] " << bboxes.size() << endl;
		for (size_t i = 0; i < bboxes.size(); ++i) {
			rectangle(frame, Point(bboxes[i].x, bboxes[i].y),
				Point(bboxes[i].x + bboxes[i].width, bboxes[i].y + bboxes[i].height), Scalar(0, 255, 0));
			roi = frame(bboxes[i]);
			stringstream ss;
			ss << "img_" << cnt << ".jpg";
			imwrite(ss.str(), roi);
			++cnt;
		}
		imshow("video", frame);
		waitKey(10);
		++frame_id;
	}
	return 0;
}