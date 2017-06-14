#include "doldetect_dll.h"
#include <cstdio>

#ifdef __linux__
#define PRINTF printf
#define SCANF scanf
#define SPRINTF sprintf
#else
#define PRINTF printf_s
#define SCANF scanf_s
#define SPRINTF sprintf_s
#endif

using namespace std;
using namespace cv;
int main()
{
	char path[100], sample_path[100];

	PRINTF("please input image/video path: ");
	SCANF("%s", path);
	VideoCapture capture;
	capture.open(path);
	if (!capture.isOpened()) {
		PRINTF("fail to read: %s\n", path);
		return 1;
	}
	Mat frame, roi, img;
	vector<Rect> bboxes;
	int frame_id = 0, cnt = 0;
	while (capture.read(frame)) {
		if (!frame.data) {
			break;
		}
		resize(frame, frame, Size(960, 540), (0, 0), (0, 0), cv::INTER_LINEAR);
		doldetect(frame, bboxes, 400, 0.5, 40, 3, 6, true);
		img = frame.clone();
		PRINTF("[frame %d] %d\n", frame_id, int(bboxes.size()));
		for (size_t i = 0; i < bboxes.size(); ++i) {
			rectangle(img, Point(bboxes[i].x, bboxes[i].y),
				Point(bboxes[i].x + bboxes[i].width, bboxes[i].y + bboxes[i].height), Scalar(0, 255, 0));
			roi = frame(bboxes[i]);
			resize(roi, roi, Size(96, 32), (0, 0), (0, 0), cv::INTER_LINEAR);
#ifdef __linux__
			SPRINTF(sample_path, "data/img_%d.jpg", cnt);
#else
			SPRINTF(sample_path, "data\\img_%d.jpg", cnt);
#endif
			imwrite(sample_path, roi);
			++cnt;
		}
		imshow("video", img);
		waitKey(5);
		++frame_id;
	}
	return 0;
}