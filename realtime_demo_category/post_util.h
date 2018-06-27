#include "stdafx.h"
#include <opencv2/core/core.hpp>    // coreモジュールのヘッダーをインクルード
#include <opencv2/highgui/highgui.hpp> // highguiモジュールのヘッダーをインクルード
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <time.h>
#include <omp.h>


using namespace std;
using namespace cv;

static bool kakunin = false;

typedef struct postits {
	int recognized_location_rectangle;
	vector<Mat> postit_image_analyzing;
	vector<vector<Point2f>>postit_points;
} Postits;

typedef struct postit_info {
	bool has_key;
	bool show;
	Point2f center;
	vector<Point2f> points;
	vector<Point2f> points_saved;
	vector<Point2f> xmeans_points;
	vector<time_t> move;
	vector<time_t> rotate;
	int naruhodo;
	int last_time;
	time_t first_time;
	int cluster_num;

} PostitInfo;

void getPostits(Postits *, cv::Mat, int);
vector<int> readDots(Mat, int);
Point2f max2f(Point2f*, int, int);
Point max2i(Point*, int, int);
Point2f min2f(Point2f*, int, int);
float norm(Point2f);
Point2f mean2f(Point2f *, int, int);
Point mean2i(Point*, int, int);
Point2f fmean2i(Point*, int, int);
float meanmat(cv::Mat);