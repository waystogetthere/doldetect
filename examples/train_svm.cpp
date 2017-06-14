#include <vector>
#include <cstdio>
#include "opencv2/opencv.hpp"

#ifdef __linux__
#define PRINTF printf
#define SCANF scanf
#define SPRINTF sprintf
#define SSCANF sscanf
#else
#define PRINTF printf_s
#define SCANF scanf_s
#define SPRINTF sprintf_s
#define SSCANF sscanf_s
#endif

using namespace cv;
using namespace std;

HOGDescriptor hog(Size(96, 32), Size(16, 16), Size(16, 16), Size(8, 8), 9);

int main()
{
	int tot_neg, tot_pos, cnt_neg = 0, cnt_pos = 0;
	char pos_path[100], neg_path[100], img_path[100];
	vector<vector<float> > hog_features;
	vector<float> labels;

	PRINTF("please input positive samples path: ");
	SCANF("%s", pos_path);
	PRINTF("total of positive samples: ");
	SCANF("%d", &tot_pos);
	PRINTF("Please input negative samples path: ");
	SCANF("%s", neg_path);
	PRINTF("total of negative samples: ");
	SCANF("%d", &tot_neg);

	// step 1: process positive samples
	Mat img;
	vector<float> hog_feat;
	PRINTF("process samples: %s ...\n", pos_path);
	for (int i = 1; i <= tot_pos; ++i) {
		// -- load image --
#ifdef __linux__
		SPRINTF(img_path, "%s/img (%d).jpg", pos_path, i);
#else
		SPRINTF(img_path, "%s\\img (%d).jpg", pos_path, i);
#endif
		img = imread(img_path, 0); // hog only accept gray-scale image
		if (!img.data) {
			PRINTF("fail to read: %s\n", img_path);
			continue;
		}
		// -- resize to 96x32 --
		resize(img, img, Size(96, 32), (0, 0), (0, 0), INTER_LINEAR);
		// -- extract hog feature --
		hog.compute(img, hog_feat, Size(96, 32), Size(0, 0));
		hog_features.push_back(hog_feat);
		labels.push_back(1.);
		++cnt_pos;
	}
	PRINTF("%d done\n", cnt_pos);

	// step 2: process negative samples
	PRINTF("process samples: %s ... \n", neg_path);
	for (int i = 1; i <= tot_neg; ++i) {
		// -- load image --
#ifdef __linux__
		SPRINTF(img_path, "%s/img (%d).jpg", neg_path, i);
#else
		SPRINTF(img_path, "%s\\img (%d).jpg", neg_path, i);
#endif
		img = imread(img_path, 0); // hog only accept gray-scale image
		if (!img.data) {
			PRINTF("fail to read: %s\n", img_path);
			continue;
		}
		// -- resize to 96x32 --
		resize(img, img, Size(96, 32), (0, 0), (0, 0), INTER_LINEAR);
		// -- extract hog feature --
		hog.compute(img, hog_feat, Size(96, 32), Size(0, 0));
		hog_features.push_back(hog_feat);
		labels.push_back(-1.);
		++cnt_neg;
	}
	PRINTF("%d done\n", cnt_neg);

	// step 3: prepare training data
	int tot = hog_features.size();
	if (!tot) {
		PRINTF("fail to load training data\n");
		return 1;
	}
	int  feat_dim = hog_features[0].size();
	PRINTF("prepare training data ...\n");
	Mat trainingData(tot, feat_dim, CV_32F), labelsMat(tot, 1, CV_32F);
	float *ptr1, *ptr2;
	for (int i = 0; i < tot; ++i) {
		ptr1 = labelsMat.ptr<float>(i);
		ptr2 = trainingData.ptr<float>(i);
		ptr1[0] = labels[i];
		for (int j = 0; j < feat_dim; ++j)
			ptr2[j] = hog_features[i][j];
	}
	PRINTF("%d x %d done\n", trainingData.rows, trainingData.cols);
	
	// step 4: train svm
	PRINTF("beging svm training ...\n");
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM svm;
	svm.train(trainingData, labelsMat, Mat(), Mat(), params);
	svm.save("svm.xml");
	PRINTF("save to:svm.xml done.\n");
	return 0;
}
