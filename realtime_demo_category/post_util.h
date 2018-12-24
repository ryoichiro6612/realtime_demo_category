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
#define KERNEL 19
#define C_TEI 8
#define OUTER_SIZE 450
//#define KERNEL 9
//#define C_TEI 5
//#define OUTER_SIZE 200

//typedef struct postits {
//	int recognized_location_rectangle;
//	vector<Mat> postit_image_analyzing;
//	vector<vector<Point2f>>postit_points;
//	vector<vector<vector<Point2f>>> location_points;
//} Postits;
typedef struct postitpoint {
	int id;
	vector<Point2f>points;
}PostitPoint;
typedef struct postitresult {
	int recognized_location_rectangle;
	vector<PostitPoint> postitpoints;
	vector<vector<vector<Point2f>>> location_points;
}PostitResult;

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

double sum_d(vector<Point2f> point1, vector<Point2f> point2);

void getPostitsGpu(PostitResult * postits, cuda::GpuMat gpuframe, int outer_size);

void getPostits(PostitResult * postits, cv::Mat frame, int outer_size);
bool NearLC(vector<Point2f>, vector<Point2f>);
bool NearID(vector<Point2f> points1, vector<Point2f> points2);
vector<int> readDots(Mat, int);
Point2f max2f(Point2f*, int, int);
Point max2i(Point*, int, int);
Point2f min2f(Point2f*, int, int);
float norm(Point2f);
Point2f mean2f(Point2f *, int, int);
Point mean2i(Point*, int, int);
Point2f fmean2i(Point*, int, int);
float meanmat(cv::Mat);