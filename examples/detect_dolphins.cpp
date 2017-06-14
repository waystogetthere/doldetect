#include "doldetect_dll.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

#ifdef __linux__
#define PRINTF printf
#define SCANF scanf
#define SSCANF sscanf
#else
#define PRINTF printf_s
#define SCANF scanf_s
#define SSCANF sscanf_s
#endif

using namespace cv;
using namespace std;

int main()
{
	vector<Rect> bboxes;
	char path[100];

	PRINTF("please input image/video path: ");
	SCANF("%s", path);
	int path_len = strlen(path);
	if (path_len < 4) {
		PRINTF("invalid path\n");
		return 1;
	}
	if (strcmp(path + path_len - 3, "jpg") == 0 ||
		strcmp(path + path_len - 3, "png") == 0 ||
		strcmp(path + path_len - 4, "jpeg") == 0) {
		PRINTF("get image input: %s\n", path);
		Mat img = imread(path);
		if (!img.data) {
			PRINTF("fail to read: %s\n", path);
			return 1;
		}
		doldetect(img, bboxes, 400, 0.5, 40, 3);
		for (size_t i = 0; i < bboxes.size(); ++i) {
			rectangle(img, Point(bboxes[i].x, bboxes[i].y),
				Point(bboxes[i].x + bboxes[i].width, bboxes[i].y + bboxes[i].height), Scalar(0, 255, 0));
		}
		imshow("image", img);
		waitKey();
	} else if (strcmp(path + path_len - 3, "avi") == 0 ||
		strcmp(path + path_len - 3, "mp4") == 0){
		PRINTF("get video input: %s\n", path);
		VideoCapture capture;
		capture.open(path);
		if (!capture.isOpened()) {
			PRINTF("fail to read: %s\n", path);
			return 1;
		}
		Mat frame;
		int frame_id = 0;
		while (capture.read(frame)) {
			resize(frame, frame, Size(960, 540), (0, 0), (0, 0), cv::INTER_LINEAR);
			if (!frame.data) {
				PRINTF("done\n");
				waitKey();
				return 0;
			}
			doldetect(frame, bboxes, 400, 0.5, 40, 3);
      PRINTF("[frame %d] %d\n", frame_id, int(bboxes.size()));
			for (size_t i = 0; i < bboxes.size(); ++i) {
				rectangle(frame, Point(bboxes[i].x, bboxes[i].y),
					Point(bboxes[i].x + bboxes[i].width, bboxes[i].y + bboxes[i].height), Scalar(0, 255, 0));
			}
			imshow("video", frame);
			waitKey(5);
			++frame_id;
		}
	} else {
		PRINTF("invalid path\n");
		return 1;
	}
	return 0;
}
