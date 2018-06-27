// realtime_demo_category.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <windows.h>
#include <time.h>
#include <thread>
#include <mutex>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "schifra_galois_field.hpp"
#include "schifra_galois_field_polynomial.hpp"
#include "schifra_sequential_root_generator_polynomial_creator.hpp"
#include "schifra_reed_solomon_encoder.hpp"
#include "schifra_reed_solomon_decoder.hpp"
#include "schifra_reed_solomon_block.hpp"
#include "schifra_error_processes.hpp"


using namespace std;
using namespace cv;

#include "analyze_config.h"
#include "read_config.h"
#include "postit_config.h"	

#include "post_util.h"

double dist_thre = DIST_THRE;
double angle_thre = ANGLE_THRE;
int time_thre = TIME_THRE;
Scalar Color[8] = { Scalar(0,255,0), Scalar(0,0,255),  Scalar(0,255,255),
Scalar(0,128,255),Scalar(255,0,255),Scalar(255,0,0),Scalar(255,255,0),Scalar(255, 255,255) };


int reedsolomon();
int reedsolomonDecode(vector<int> bit_array);
int GetRandom(int min, int max);
std::vector<std::string> ssplit(const std::string &str, char sep);
vector<vector<int>> dbscan_fusen(vector<vector<Point>>data, double eps);
vector<int>core_fusen(int, double, vector<vector<Point>>, int*);
vector<int>neibor_fusen(int, double, vector<vector<Point>>, int*);
double fusen_dict(vector<Point>, vector<Point>);
double tentosen_dict(Point, Point, Point);
int gaiseki(Point p1, Point p2) {
	return p1.x*p2.y - p1.y*p2.x;
}
int naiseki(Point p1, Point p2) {
	return p1.x*p2.x + p2.y*p1.y;
}

template <class X> X modo(X* start, int size) {

	int max = 0, cnt = 0, index;

	for (int j = 0; j<size; j++, cnt = 0)
	{
		for (int i = j; i<size; i++)
		{
			if (start[j] == start[i])
				cnt++;
		}
		if (max < cnt) {
			max = cnt;
			index = j;
		}
	}
	return start[index];
}

void mouse_callback(int event, int x, int y, int flags, void*p) {
	vector<Point2f> * pdesk_area = (vector<Point2f>*)p;
	if (event == CV_EVENT_LBUTTONDOWN) {
		(*pdesk_area).push_back(Point2f((float)x, (float)y));
	}
}

vector<vector<int>>calc_cross(vector<vector<int>>A, vector<vector<int>>C) {
	vector<vector<int>> X;
	int i;
	for (i = 0; i < C.size(); i++) {
		X.push_back(vector<int>(A.size()));
		int j;
		for (j = 0; j < A.size(); j++) {
			int count = 0;
			int k, l;
			for (k = 0; k < C[i].size(); k++) {
				for (l = 0; l < A[j].size(); l++) {
					if (C[i][k] == A[j][l]) {
						count++;
					}
				}
			}
			X[i][j] = count;
		}
	}
	return X;
}


double calc_entropy(vector<vector<int>>X, int N) {
	double entropy = 0;
	int i;
	for (i = 0; i < X.size(); i++) {
		int Ci = 0;
		int j;
		double ent = 0;
		for (j = 0; j < X[i].size(); j++) {
			Ci += X[i][j];
		}
		if (Ci == 0) {
			break;
		}
		for (j = 0; j < X[i].size(); j++) {
			double p = 1.0 * X[i][j] / Ci;
			if (p == 0) {
				ent += 0;
				continue;
			}
			ent += -p * log(p);
		}
		entropy += 1.0 * Ci / N * ent;
	}
	return entropy;
}

double calc_purity(vector<vector<int>>X, int N) {
	double purity = 0;
	int i;
	for (i = 0; i < X.size(); i++) {
		int j;
		int max = 0;
		for (j = 0; j < X[i].size(); j++) {
			if (max < X[i][j]) {
				max = X[i][j];
			}
		}
		purity += max;
	}
	purity /= 1.0 * N;
	return purity;
}
double calc_F(vector<vector<int>> X, int N) {
	vector<int>A_num(X[0].size(), 0);
	vector<int>C_num(X.size(), 0);
	int i, j;
	for (i = 0; i < X.size(); i++) {
		for (j = 0; j < X[i].size(); j++) {
			A_num[j] += X[i][j];
			C_num[i] += X[i][j];
		}
	}

	double Fti = 0;
	for (j = 0; j < A_num.size(); j++) {
		double maxF = 0;
		double maxR = 0;
		double maxP = 0;
		for (i = 0; i < C_num.size(); i++) {
			double Pij = 1.0* X[i][j] / C_num[i];
			double Rij = 1.0 * X[i][j] / A_num[j];
			double F = 2.0*Pij*Rij / (Pij + Rij);
			if (maxF < F) {
				maxF = F;
				maxR = Rij;
				maxP = Pij;
			}
		}
		Fti += A_num[j] / (1.0*N) * maxF;
	}
	return Fti;
}

void cameraCapture(Mat &frame) {
	//*usb camera use
	//video reader and writer
	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	while (1) {
		cap >> frame;
	}
}

int main(int argc, char * argv[])
{
	//*/
	if (sizeof(argv) / sizeof(argv[0]) != 3) {
		cout << "usage: " << argv[0] << " outer_size fps\n";
	}
	//int outer_size = atoi(argv[1]);
	//int fps = atoi(argv[2]);
	int outer_size = 350;
	int fps = 1;
	bool skipmode = true;



	// real time parameters
	int newest_key_saved = -1;
	int projector_resolution[2] = { 1280, 765 };


	cv::Mat frame,cameraFrame;
	if (kakunin) {
		namedWindow("frame", CV_WINDOW_AUTOSIZE);
	}
	namedWindow("projection", CV_WINDOW_AUTOSIZE);

	//*usb camera use
	//video reader and writer
	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	cap >> frame;

	/*cv::namedWindow("test");
	while (1) {
		DWORD timer1 = GetTickCount();
		cap >> frame;
		DWORD timer2 = GetTickCount();
		cout << "capture:" << timer2- timer1 << "ms" << endl;
		imshow("test", frame);
		waitKey(2000);
	}*/

	cout << frame.rows << endl;
	cout << frame.cols << endl;

	int frame_size_0 = frame.rows;
	int frame_size_1 = frame.cols;
	//*/
	//projection area out
	int brightness = 100;
	Mat projection_img(projector_resolution[1], projector_resolution[0], CV_8UC3, Scalar(brightness, brightness, brightness));
	cv::namedWindow("projection");
	cv::imshow("projection", projection_img);
	cv::waitKey(1);

	//*first select desk area
	vector<Point2f> desk_area;
	cv::namedWindow("SELECT DESK", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("SELECT DESK", mouse_callback, &desk_area);
	desk_area.push_back(Point2f(0.0, 0.0));
	desk_area.push_back(Point2f(float(frame.cols), float(frame.rows)));
	desk_area.push_back(Point2f(float(frame.cols), 0));
	desk_area.push_back(Point2f(0, float(frame.rows)));
	/*while (1) {
		cap >> frame;
		Mat frame_for_select(int(frame.rows * 0.2), int(frame.cols * 0.2), frame.type());
		cv::resize(frame, frame_for_select, frame_for_select.size());
		cv::imshow("SELECT DESK", frame_for_select);
		char c = cv::waitKey(30);
		if (c >= 0) {
			break;
		}
	}
	int i;
	for (i = 0; i < desk_area.size(); i++) {
		desk_area[i] *= 5;
	}*/



	int mi;
	vector<PostitInfo> postit_saved(256);
	for (mi = 0; mi < postit_saved.size(); mi++) {
		postit_saved[mi].has_key = false;
		postit_saved[mi].show = false;
		postit_saved[mi].naruhodo = 0;
	}

	int timer_table[10][10];


	//kokokara loop
	int timer_count = 0;

	//Arg1 arg{&rawImage, &cam, false};
	//thread th1 = thread(imageCapture, &arg);

	int newest_key = -1;
	int newest_time = -1;

	thread t1(cameraCapture, ref(cameraFrame));
	std::mutex mtx_camera;
	waitKey(1000);

	while (1) {
		cout << "progress  " << std::to_string(int(timer_count / fps) / 60) << ":" << std::to_string((timer_count / fps) % 60) << "\n";
		DWORD now_timer, prev_timer;
		if (timer_count == 10) {
			//break;
		}
		//*usb camera capture frame
		DWORD timer_start = GetTickCount();
		mtx_camera.lock();
		cameraFrame.copyTo(frame);
		mtx_camera.unlock();
		DWORD timer_temp = GetTickCount();
		prev_timer = timer_temp;
		
		cout << "camera capture:" << timer_temp - timer_start << "ms" << endl;

		//extract only desk_area
		Mat frame_deskarea(frame, Rect((int)desk_area[0].x, (int)desk_area[0].y,
			(int)((desk_area[1] - desk_area[0]).x), (int)(desk_area[1] - desk_area[0]).y));

		now_timer = GetTickCount();
		
		cout << "desk_area extract:" << now_timer - prev_timer << "ms" << endl;
		prev_timer = now_timer;
		cout << "capture:" << now_timer - timer_start << "ms" <<endl;
		
		Postits postits;
		getPostits(&postits, frame_deskarea, outer_size);
		now_timer = GetTickCount();
		cout << "recognition:" << now_timer - prev_timer << "ms" << endl;
		prev_timer = now_timer;
		vector<vector<Point2f>
> postit_points = postits.postit_points;
		//add buffer
		int i, j;
		for (i = 0; i < postit_points.size(); i++) {
			for (j = 0; j < postit_points[i].size(); j++) {
				postit_points[i][j] += desk_area[0];
			}
		}


		//read postit's id and save
		vector<int> postit_ids;
		vector<vector<int>> bit_array_list;

		for (i = 0; i < postit_points.size(); i++) {
			Mat postit_image_analyzing = postits.postit_image_analyzing[i];
			vector<int> bit_array = readDots(postit_image_analyzing, outer_size);
			bit_array_list.push_back(bit_array);
			int result_num = -1;
			result_num = reedsolomonDecode(bit_array);
			postit_ids.push_back(result_num);
			//save postit image
			if (result_num > 0) {
				if (kakunin) {
					cout << result_num << endl;
				}
				// already exist
				if (postit_saved[result_num].has_key) {

					// judge move and rotate
					// calc dist
					double dist = double(norm(
						mean2f(&postit_saved[result_num].points_saved[0], 4, 0) - mean2f(&postit_points[i][0], 4, 0)));
					// calc angle(degree)
					Point2f angle_vec_before = postit_saved[result_num].points_saved[0] - postit_saved[result_num].points_saved[1];
					Point2f angle_vec_after = postit_points[i][0] - postit_points[i][1];
					double vec_cos = (angle_vec_before.x * angle_vec_after.x + angle_vec_before.y*angle_vec_after.y) /
						(norm(angle_vec_before)*norm(angle_vec_after));
					double angle = acos(vec_cos) * 180 / M_PI;
					//add information
					if (angle > angle_thre) {
						time_t timer;
						time(&timer);
						postit_saved[result_num].rotate.push_back(timer);
					}
					/*
					if (dist > dist_thre) {
					int count_move = postit_saved[result_num].move.size();
					if (count_move == 0) {
					postit_saved[result_num].move.push_back(time / fps);
					}
					else if(time/fps - postit_saved[result_num].move[count_move - 1] > 5){
					postit_saved[result_num]
					}
					}
					//*/
					//renew
					postit_saved[result_num].show = true;
					postit_saved[result_num].points = postit_points[i];
					postit_saved[result_num].points_saved = postit_points[i];
					postit_saved[result_num].last_time = timer_count / fps;
				}
				else {
					postit_saved[result_num].has_key = true;
					postit_saved[result_num].show = true;
					postit_saved[result_num].points = postit_points[i];
					postit_saved[result_num].points_saved = postit_points[i];
					//postit_saved[result_num].naruhodo = 0;
					postit_saved[result_num].last_time = timer_count / fps;
					time(&postit_saved[result_num].first_time);
				}
			}
		}
		now_timer = GetTickCount();
		cout << "id recognition:" << now_timer - prev_timer << "ms" << endl;
		prev_timer = now_timer;
		for (i = 0; i < postit_saved.size(); i++) {
			PostitInfo val = postit_saved[i];
			if (postit_saved[i].show) {
				if ((timer_count / fps - val.last_time) > time_thre) {
					postit_saved[i].points[0] = Point2f(-5, 0);
					postit_saved[i].points[1] = Point2f(0, 0);
					postit_saved[i].points[2] = Point2f(0, -5);
					postit_saved[i].points[3] = Point2f(-5, -5);
					postit_saved[i].show = false;
				}
			}
		}

		//find newest postit
		int key;
		for (key = 0; key < postit_saved.size(); key++) {
			if (postit_saved[key].show) {
				if (postit_saved[key].rotate.size() != 0) {
					if (postit_saved[key].rotate.back() > newest_time) {
						newest_key = key;
						newest_time = postit_saved[key].rotate.back();
					}
				}

			}
		}
		for (key = 0; key < postit_saved.size(); key++) {
			if (postit_saved[key].show) {

				if (postit_saved[key].first_time > newest_time) {
					newest_key = key;
					newest_time = postit_saved[key].first_time;
				}
			}
		}
		if (newest_key != newest_key_saved) {
			newest_key_saved = newest_key;
			FILE * naruhodo_csv;
			fopen_s(&naruhodo_csv, "./datas/csv/naruhodo.csv", "w");
			//fopen_s(&naruhodo_csv, "./datas/csv/naruhodo.csv", "a");
			fclose(naruhodo_csv);
		}
		//add naruhodo
		FILE * naruhodo_csv;

		fopen_s(&naruhodo_csv, "./datas/csv/naruhodo.csv", "r");
		int naruhodo_count = 0;
		char s[10];

		while (fgets(s, 9, naruhodo_csv) != NULL) {
			naruhodo_count++;
		}
		fclose(naruhodo_csv);
		fopen_s(&naruhodo_csv, "./datas/csv/naruhodo.csv", "w");
		//fopen_s(&naruhodo_csv, "./datas/csv/naruhodo.csv", "a");
		fclose(naruhodo_csv);
		if (newest_key != -1) {
			postit_saved[newest_key].naruhodo += naruhodo_count;
		}
		//cout << "newest key:" << newest_key << endl;
		//draw information
		int brightness = 50;
		float down_scale_x = 1.0 * float(projector_resolution[0]) / (desk_area[1].x - desk_area[0].x);
		float down_scale_y = 1.0 * float(projector_resolution[1]) / (desk_area[1].y - desk_area[0].y);
		int x_buffer = 0;
		int y_buffer = 0;

		//* dbscan husen cluster number
		vector <vector<Point>> husen_points;
		vector<int> vec_id;
		for (key = 0; key < postit_saved.size(); key++) {
			if (postit_saved[key].show) {
				vec_id.push_back(key);
				vector<Point> Points;
				int i;
				for (i = 0; i < postit_saved[key].points.size(); i++) {
					Points.push_back(Point((int)postit_saved[key].points[i].x, (int)postit_saved[key].points[i].y));
				}
				husen_points.push_back(Points);
			}
		}
		vector<vector<int>>C = dbscan_fusen(husen_points, 80);

		for (i = 0; i < C.size(); i++) {
			for (j = 0; j < C[i].size(); j++) {
				if (vec_id.size() != 0) {
					int key = vec_id[C[i][j]];
					postit_saved[key].cluster_num = i;
				}

			}
		}
		//*/

		projection_img = Mat(projector_resolution[1], projector_resolution[0], CV_8UC3, Scalar(brightness, brightness, brightness));

		/*
		vector<int> key_vec;
		vector<Point2f> center_vec;
		for (key = 0; key < postit_saved.size(); key++) {
		if (postit_saved[key].show == true) {
		Point2f center = mean2f(&postit_saved[key].points[0], 4, 0);
		key_vec.push_back(key);
		postit_saved[key].center = center;
		center_vec.push_back(center);

		postit_saved[key].xmeans_points = vector<Point2f>(16);
		postit_saved[key].xmeans_points[0] = postit_saved[key].points[0];
		postit_saved[key].xmeans_points[1] = postit_saved[key].points[1];
		postit_saved[key].xmeans_points[2] = postit_saved[key].points[2];
		postit_saved[key].xmeans_points[3] = postit_saved[key].points[3];
		postit_saved[key].xmeans_points[4] = (postit_saved[key].points[0] + postit_saved[key].points[1])*0.5;
		postit_saved[key].xmeans_points[5] = (postit_saved[key].points[1] + postit_saved[key].points[2])*0.5;
		postit_saved[key].xmeans_points[6] = (postit_saved[key].points[2] + postit_saved[key].points[3])*0.5;
		postit_saved[key].xmeans_points[7] = (postit_saved[key].points[0] + postit_saved[key].points[3])*0.5;
		postit_saved[key].xmeans_points[8] = center + (postit_saved[key].points[0] - center)*0.5;
		postit_saved[key].xmeans_points[9] = center + (postit_saved[key].points[1] - center)* 0.5;
		postit_saved[key].xmeans_points[10] = center + (postit_saved[key].points[2] - center)* 0.5;
		postit_saved[key].xmeans_points[11] = center + (postit_saved[key].points[3] - center)* 0.5;
		postit_saved[key].xmeans_points[12] = (postit_saved[key].xmeans_points[8] + postit_saved[key].xmeans_points[9])*0.5;
		postit_saved[key].xmeans_points[13] = (postit_saved[key].xmeans_points[9] + postit_saved[key].xmeans_points[10])*0.5;
		postit_saved[key].xmeans_points[14] = (postit_saved[key].xmeans_points[10] + postit_saved[key].xmeans_points[11])*0.5;
		postit_saved[key].xmeans_points[15] = (postit_saved[key].xmeans_points[8] + postit_saved[key].xmeans_points[11])*0.5;
		}
		}

		int npts[4] = {16,4,8,16};
		int npp;
		int npoints = npts[npp];
		string center_csv("");
		for (i = 0; i < key_vec.size(); i++) {
		int key = key_vec[i];
		//*
		for (j = 0; j < npoints; j++) {
		char data_string[100];
		//sprintf_s(data_string,100, "%f,%f\n", center_vec[i].x, center_vec[i].y);
		sprintf_s(data_string, 100, "%f,%f\n", postit_saved[key].xmeans_points[j].x, postit_saved[key].xmeans_points[j].y);
		center_csv += data_string;
		}
		//*/
		/*
		char data_string[100];
		sprintf_s(data_string,100, "%f,%f\n", center_vec[i].x, center_vec[i].y);
		//sprintf_s(data_string, 100, "%f,%f\n", postit_saved[key].xmeans_points[j].x, postit_saved[key].xmeans_points[j].y);
		center_csv += data_string;
		//*/
		//}

		//cout << center_csv;
		/*
		string labels_string = xmeans(&expression, &center_csv[0]);
		string charmi("\n");
		vector<string>label_vec_string = ssplit(labels_string, charmi[0]);
		cout << &labels_string[0];
		cout << key_vec.size() << endl;
		cout << label_vec_string.size() << endl;
		for (i = 0; i < label_vec_string.size() / npoints; i++) {
		int key = key_vec[i];
		int vec_label[100];
		for (j = 0; j < npoints; j++) {
		vec_label[j] = atoi(&label_vec_string[npoints*i + j][0]);
		}
		int label;
		label = modo(vec_label, npoints);
		postit_saved[key].cluster_num = label;
		}
		*/
		/*
		if (key_vec.size() != 0) {
		int clusterCount = 2;
		Mat sample(key_vec.size(), 1, CV_64FC2);
		Mat labels;
		Mat centers;
		for (i = 0; i < key_vec.size(); i++) {
		int key = key_vec[i];
		sample.at<Vec2d>(i, 0)[0] = (double)postit_saved[key].center.x;
		sample.at<Vec2d>(i, 0)[1] = (double)postit_saved[key].center.y;
		}
		cv::Mat points = cv::Mat::zeros(key_vec.size(), 1, CV_32FC2);
		for (int i = 0; i<key_vec.size(); ++i) {
		int key = key_vec[i];
		points.at<cv::Vec2f>(i, 0)[0] = postit_saved[key].center.x;
		points.at<cv::Vec2f>(i, 0)[1] = postit_saved[key].center.y;
		}

		kmeans(points, MIN(key_vec.size(), clusterCount), labels,
		TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
		5, KMEANS_PP_CENTERS, centers);
		std::cout << labels << endl;
		std::cout << CV_8SC1;
		int mimi;

		for (i = 0; i < key_vec.size(); i++) {
		int key = key_vec[i];
		postit_saved[key].cluster_num = labels.at<int>(i);
		}
		//for (i = 0; i < labels.rows; i++) {
		//	for (j = 0; j < labels.cols; j++) {

		//}
		//}
		}
		//*/

		/*
		int ncluster = 5;
		vector<Point2f> cluster_center(ncluster);
		vector<vector<int>> cluster_mat(ncluster);
		int clust;
		for (clust = 0; clust < cluster_center.size(); clust++) {
		cluster_center[clust] = Point2f(0, 0);
		}
		//step1 first state
		for (key = 0; key < postit_saved.size(); key++) {
		if (postit_saved[key].show == true) {
		postit_saved[key].center = mean2f(&postit_saved[key].points[0], 4, 0);
		int cluster = GetRandom(1, ncluster-1);
		postit_saved[key].cluster_num = cluster;
		cluster_mat[cluster].push_back(key);
		}
		}
		bool changed = true;
		while (changed) {
		changed = false;
		//step2 calculate center
		for (clust = 0; clust < ncluster; clust++) {
		int j;
		for (j = 0; j<cluster_mat[clust].size(); j++) {
		int key = cluster_mat[clust][j];
		cluster_center[clust] += postit_saved[key].center;
		}
		if (cluster_mat[clust].size() > 0) {
		cluster_center[clust] *= 1.0 / int(cluster_mat[clust].size());
		}
		else {
		cluster_center[clust] = Point2f(-1000, -1000);
		}
		}
		//step3 change cluster nearest center
		for (key = 0; key < postit_saved.size(); key++) {
		if (postit_saved[key].show == true) {
		int nearest_cluster = 0;
		float min_dist = norm(cluster_center[0] - postit_saved[key].center);
		for (clust = 0; clust < ncluster; clust++) {
		float dist = norm(cluster_center[clust] - postit_saved[key].center);
		if (dist < min_dist) {
		min_dist = dist;
		nearest_cluster = clust;
		}
		}
		if (postit_saved[key].cluster_num != nearest_cluster) {
		changed = true;
		postit_saved[key].cluster_num = nearest_cluster;
		}

		}
		}
		}
		//*/

		//projection tuduki
		projection_img = Mat(projector_resolution[1], projector_resolution[0], CV_8UC3, Scalar(brightness, brightness, brightness));
		for (key = 0; key < postit_saved.size(); key++) {

			if (postit_saved[key].show == true) {
				//cout << key << endl;
				vector<Point> caliblated_points(4);
				int i;
				for (i = 0; i < 4; i++) {
					vector<Point2f> before_points{ Point2f((float)desk_area[0].x,(float)desk_area[0].y), Point2f((float)desk_area[2].x,(float)desk_area[2].y),
						Point2f((float)desk_area[1].x,(float)desk_area[1].y), Point2f((float)desk_area[3].x,(float)desk_area[3].y) };
					vector<Point2f> after_points{ Point2f(0,0), Point2f(projector_resolution[0], 0),
						Point2f(projector_resolution[0],projector_resolution[1]), Point2f(0, projector_resolution[1]) };
					Mat M = cv::getPerspectiveTransform(before_points, after_points);
					Point2f each_point = postit_saved[key].points[i];
					cv::Mat eachpoint_3d = (cv::Mat_<double>(3, 1) << (double)each_point.x, (double)each_point.y, 1);
					cv::Mat caliblatedPoint_3d(3, 1, CV_64FC1);
					caliblatedPoint_3d = M*eachpoint_3d;
					caliblated_points[i] = Point(projector_resolution[0] - (int)caliblatedPoint_3d.at<double>(0, 0), projector_resolution[1] - (int)caliblatedPoint_3d.at<double>(1, 0));
					/*
					caliblated_points[i] = Point((int)((each_point.x - desk_area[0].x)*down_scale_x) - x_buffer,
					(int)((each_point.y - desk_area[0].y)*down_scale_y) - y_buffer);
					//*/
				}

				float buff_ratio = 0.0;
				caliblated_points[0] += ((caliblated_points[1] - caliblated_points[2]) +
					(caliblated_points[3] - caliblated_points[2])) * buff_ratio;
				caliblated_points[1] += ((caliblated_points[0] - caliblated_points[3]) + (
					caliblated_points[2] - caliblated_points[3])) * buff_ratio;
				caliblated_points[2] += ((caliblated_points[1] - caliblated_points[0]) + (
					caliblated_points[3] - caliblated_points[0])) * buff_ratio;
				caliblated_points[3] += ((caliblated_points[0] - caliblated_points[1]) + (
					caliblated_points[2] - caliblated_points[1])) * buff_ratio;


				/*
				Point hosei(-120, -100);
				for (i = 0; i < 4; i++) {
				caliblated_points[i] += hosei;
				}
				//*/

				/*
				if (key == newest_key) {
				cv::drawContours(projection_img,
				vector<vector<Point>>{caliblated_points}, 0,
				Scalar(0, 0, 255), 2);
				}
				else {
				cv::drawContours(projection_img,
				vector<vector<Point>>{caliblated_points}, 0,
				Scalar(226, 188, 163), 2);
				}

				//*/
				/*
				cv::drawContours(projection_img,
				vector<vector<Point>>{caliblated_points}, 0,
				Color[postit_saved[key].cluster_num], 2);
				//*/
				Point2f center_point = mean2i(&caliblated_points[0], 4, 0);

				//*
				cv::putText(projection_img, to_string(key),
					Point(int(center_point.x - 60), int(center_point.y - 20) + 0),
					CV_FONT_HERSHEY_PLAIN, 2.5, Color[postit_saved[key].cluster_num], 5);

				//*/

				/*
				if (key < 64) {
				cv::drawContours(projection_img,
				vector<vector<Point>>{caliblated_points}, 0,
				Scalar(0, 0, 255), 2);
				}
				else if (key <128) {
				cv::drawContours(projection_img,
				vector<vector<Point>>{caliblated_points}, 0,
				Scalar(0, 255, 0), 2);
				}
				else if (key < 192) {
				cv::drawContours(projection_img,
				vector<vector<Point>>{caliblated_points}, 0,
				Scalar(255, 0, 0), 2);
				}
				else if (key < 256) {
				cv::drawContours(projection_img,
				vector<vector<Point>>{caliblated_points}, 0,
				Scalar(0, 255, 255), 2);
				}
				//*/
				/*
				cv::putText(projection_img, to_string(key),
				Point(int(center_point.x + 80), int(center_point.y) + 100),
				CV_FONT_HERSHEY_PLAIN, 4.0, Scalar(100, 70, 202), 5);
				//*/

				time_t d = postit_saved[key].first_time;
				struct tm t_st;
				localtime_s(&t_st, &d);

				/*
				cv::putText(projection_img, to_string(t_st.tm_hour) + ":" + to_string(t_st.tm_min) + ":" + to_string(t_st.tm_sec),
				Point(center_point.x - 100, max2i(&caliblated_points[0], 4, 0).y + 20),
				CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(197, 126, 24), 2);
				//*/
			}
			/*
			#cv2.drawContours(projection_img,
			#                 [np.array(caliblated_points)], 0,
			#                 (226, 188, 163), -1)
			#    center_point = np.mean(caliblated_points, axis=0)



			#cv2.putText(projection_img, str(postit_saved[key]["naruhodo"]), (
			#    int(center_point[0] +  80),
			#    int(center_point[1]) + 100), cv2.FONT_HERSHEY_PLAIN, 4.0,
			#            (100, 70, 202), 5)



			d = postit_saved[key]["first_time"]
			#    cv2.putText(projection_img, str(d.hour) + ":" + str(d.minute) + ":" + str(d.second), (
			#        int(center_point[0] - 100),
			#        int(np.max(caliblated_points, axis = 0)[1]) + 20), cv2.FONT_HERSHEY_PLAIN, 1.5,
			#                (197, 126, 24), 2)
			//*/

		}



		/*
		Mat resized_projection_img((int)(projection_img.rows / 5.0), (int)(projection_img.cols / 5.0), projection_img.type());
		cv::resize(projection_img, resized_projection_img, resized_projection_img.size());
		*/
		imshow("projection", projection_img);
		int c;
		c = cv::waitKey(1);
		if (c == 27) {
			break;
		}

		now_timer = GetTickCount();
		cout << "add_information:" << now_timer - prev_timer << "ms" << endl;
		prev_timer = now_timer;
		cout << "all :" << now_timer - timer_start << "ms" << endl;
		timer_count++;
	}
	/*
	// Stop capturing images
	error = cam.StopCapture();
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}

	// Disconnect the camera
	error = cam.Disconnect();
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}


	cout << "Done!" << endl;
	*/
	//cout << "Done! Press Enter to exit..." << endl;
	//cin.ignore();
	return 0;

	/*
	//set camera
	PrintBuildInfo();

	Error error;

	// Since this application saves images in the current folder
	// we must ensure that we have permission to write to this folder.
	// If we do not have permission, fail right away.
	FILE* tempFile;
	fopen_s(&tempFile, "test.txt", "w+");
	if (tempFile == NULL)
	{
	cout << "Failed to create file in current folder.  Please check permissions." << endl;
	return -1;
	}
	fclose(tempFile);
	remove("test.txt");

	BusManager busMgr;
	unsigned int numCameras;
	error = busMgr.GetNumOfCameras(&numCameras);
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}

	cout << "Number of cameras detected: " << numCameras << endl;


	PGRGuid guid;
	error = busMgr.GetCameraFromIndex(0, &guid);
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}


	Camera cam;

	// Connect to a camera
	error = cam.Connect(&guid);
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}

	// Get the camera information
	CameraInfo camInfo;
	error = cam.GetCameraInfo(&camInfo);
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}
	PrintCameraInfo(&camInfo);

	// Get the camera configuration
	FC2Config config;
	error = cam.GetConfiguration(&config);
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}

	// Set the number of driver buffers used to 10.
	config.numBuffers = 10;

	// Set the camera configuration
	error = cam.SetConfiguration(&config);
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}

	// Start capturing images
	error = cam.StartCapture();
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}


	//kokokara loop
	while (1) {
	Image rawImage;

	// Retrieve an image
	error = cam.RetrieveBuffer(&rawImage);
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}

	cout << "Grabbed image " << endl;
	cout << rawImage.GetRows() << ":" << rawImage.GetCols() << endl;

	// Create a converted image
	Image convertedImage;
	Image grayImage;



	// Convert the raw image
	error = rawImage.Convert(PIXEL_FORMAT_BGR, &convertedImage);
	error = rawImage.Convert(PIXEL_FORMAT_MONO8, &grayImage);
	cv::Mat frame;
	unsigned int rowBytes = (unsigned int)((double)convertedImage.GetReceivedDataSize() / (double)convertedImage.GetRows());
	frame = cv::Mat(convertedImage.GetRows(), convertedImage.GetCols(), CV_8UC3, convertedImage.GetData(), rowBytes);
	Mat show_img = cv::Mat(cv::Size(frame.cols / 5, frame.rows / 5), CV_8UC3);
	cv::resize(frame, show_img, show_img.size());
	cv::namedWindow("out", WINDOW_AUTOSIZE);
	imshow("out", frame);
	char key = waitKey(1);
	if (key == 27) {
	break;
	}
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}


	// Save the image. If a file format is not passed in, then the file
	// extension is parsed to attempt to determine the file format.
	/*error = convertedImage.Save("mi.jpg");
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}
	}

	//kokokara

	// Stop capturing images
	error = cam.StopCapture();
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}

	// Disconnect the camera
	error = cam.Disconnect();
	if (error != PGRERROR_OK)
	{
	PrintError(error);
	return -1;
	}


	cout << "Done!" << endl;
	//cout << "Done! Press Enter to exit..." << endl;
	//cin.ignore();
	return 0;
	//*/

}

int reedsolomonDecode(vector<int> bit_array) {
	/* Finite Field Parameters */
	const std::size_t field_descriptor = 8;
	const std::size_t generator_polynomial_index = 0;
	const std::size_t generator_polynomial_root_count = 14;

	/* Reed Solomon Code Parameters */
	const std::size_t code_length = 15;
	const std::size_t fec_length = 14;
	const std::size_t data_length = code_length - fec_length;

	/* Instantiate Finite Field and Generator Polynomials */
	schifra::galois::field field(field_descriptor,
		schifra::galois::primitive_polynomial_size05,
		schifra::galois::primitive_polynomial05);

	schifra::galois::field_polynomial generator_polynomial(field);

	if (
		!schifra::make_sequential_root_generator_polynomial(field,
			generator_polynomial_index,
			generator_polynomial_root_count,
			generator_polynomial)
		)
	{
		std::cout << "Error - Failed to create sequential root generator!" << std::endl;
		return -1;
	}

	/* Instantiate Encoder and Decoder (Codec) */
	typedef schifra::reed_solomon::shortened_encoder<code_length, fec_length, data_length> encoder_t;
	typedef schifra::reed_solomon::shortened_decoder<code_length, fec_length, data_length> decoder_t;

	encoder_t encoder(field, generator_polynomial);
	decoder_t decoder(field, generator_polynomial_index);



	/* Instantiate RS Block For Codec */
	schifra::reed_solomon::block<code_length, fec_length> bit_array_block;
	int i;
	for (i = 0; i < 15; i++) {
		bit_array_block.data[i] = bit_array[i];
	}
	//bit_array_block.data[0] = 100;
	//bit_array_block.data[1] = 100;



	/* Add errors at every 8th location starting at position zero */
	//schifra::corrupt_message_all_errors00(block, 0, 8);

	/*
	if (kakunin) {
	cout << "Original BitArray:[";
	for (i = 0; i < 15; i++) {
	cout << int(bit_array_block.data[i]) << ", ";
	}
	cout << "]\n";
	}
	//*/



	if (!decoder.decode(bit_array_block))
	{
		std::cout << "Error - Critical decoding failure!" << std::endl;
		return -1;
	}
	string decoded_string((int)data_length, 0);
	bit_array_block.data_to_string(decoded_string);

	//*
	if (kakunin) {
		cout << "Decoded  BitArray:[";
		for (i = 0; i < 15; i++) {
			cout << int(bit_array_block.data[i]) << ", ";
		}
		cout << "]\n";
	}

	//*/
	if (decoded_string[0] < 0) {
		return (int)decoded_string[0] + 256;
	}
	else {
		return (int)decoded_string[0];
	}
}
int reedsolomon() {
	/* Finite Field Parameters */
	const std::size_t field_descriptor = 8;
	const std::size_t generator_polynomial_index = 120;
	const std::size_t generator_polynomial_root_count = 14;

	/* Reed Solomon Code Parameters */
	const std::size_t code_length = 15;
	const std::size_t fec_length = 14;
	const std::size_t data_length = code_length - fec_length;

	/* Instantiate Finite Field and Generator Polynomials */
	schifra::galois::field field(field_descriptor,
		schifra::galois::primitive_polynomial_size06,
		schifra::galois::primitive_polynomial06);

	schifra::galois::field_polynomial generator_polynomial(field);

	if (
		!schifra::make_sequential_root_generator_polynomial(field,
			generator_polynomial_index,
			generator_polynomial_root_count,
			generator_polynomial)
		)
	{
		std::cout << "Error - Failed to create sequential root generator!" << std::endl;
		return -256;
	}

	/* Instantiate Encoder and Decoder (Codec) */
	typedef schifra::reed_solomon::shortened_encoder<code_length, fec_length, data_length> encoder_t;
	typedef schifra::reed_solomon::shortened_decoder<code_length, fec_length, data_length> decoder_t;

	encoder_t encoder(field, generator_polynomial);
	decoder_t decoder(field, generator_polynomial_index);

	std::string message = "Where did I come from, and what am I supposed to be doing...";
	message = "a";

	/* Pad message with nulls up until the code-word length */
	message.resize(code_length, 0x00);

	std::cout << "Original Message:   [" << message << "]" << std::endl;

	/* Instantiate RS Block For Codec */
	schifra::reed_solomon::block<code_length, fec_length> block;

	/* Transform message into Reed-Solomon encoded codeword */
	if (!encoder.encode(message, block))
	{
		std::cout << "Error - Critical encoding failure!" << std::endl;
		return -1;
	}

	/* Add errors at every 8th location starting at position zero */
	schifra::corrupt_message_all_errors00(block, 0, 8);
	block.data[2] = 100;
	std::cout << "Corrupted Codeword: [" << block << "]" << std::endl;

	if (!decoder.decode(block))
	{
		std::cout << "Error - Critical decoding failure!" << std::endl;
		return -1;
	}
	else if (!schifra::is_block_equivelent(block, message))
	{
		std::cout << "Error - Error correction failed!" << std::endl;
		return -1;
	}

	block.data_to_string(message);

	std::cout << "Corrected Message:  [" << message << "]" << std::endl;

	std::cout << "Encoder Parameters [" << encoder_t::trait::code_length << ","
		<< encoder_t::trait::data_length << ","
		<< encoder_t::trait::fec_length << "]" << std::endl;

	std::cout << "Decoder Parameters [" << decoder_t::trait::code_length << ","
		<< decoder_t::trait::data_length << ","
		<< decoder_t::trait::fec_length << "]" << std::endl;

	return 0;
}


int GetRandom(int min, int max)
{
	return min + (int)(rand()*(max - min + 1.0) / (1.0 + RAND_MAX));
}

std::vector<std::string> ssplit(const std::string &str, char sep)
{
	std::vector<std::string> v;        // �������ʂ��i�[����x�N�^�[
	auto first = str.begin();              // �e�L�X�g�̍ŏ����w���C�e���[�^
	while (first != str.end()) {         // �e�L�X�g���c���Ă���ԃ��[�v
		auto last = first;                      // ���������񖖔��ւ̃C�e���[�^
		while (last != str.end() && *last != sep)       // ���� or �Z�p���[�^�����܂Ői�߂�
			++last;
		v.push_back(std::string(first, last));       // �����������o��
		if (last != str.end())
			++last;
		first = last;          // ���̏����̂��߂ɃC�e���[�^��ݒ�
	}
	return v;
}

vector<vector<int>> dbscan_fusen(vector<vector<Point>>data, double eps) {
	int * classed = (int *)malloc(sizeof(int) * data.size());
	int i;
	for (i = 0; i < data.size(); i++) {
		classed[i] = false;
	}
	int count_classed = 0;
	vector<vector<int>> C;

	int cls = 0;

	while (1) {
		C.push_back(vector<int>());
		for (i = 0; i < data.size(); i++) {
			if (classed[i] == false) {
				break;
			}
		}
		//C[cls].push_back(i);
		//classed[i] = true;
		//count_classed++;
		vector<int> core_id = core_fusen(i, eps, data, classed);
		int j;
		for (j = 0; j < core_id.size(); j++) {
			C[cls].push_back(core_id[j]);
			//classed[core_id[j]] = true;
			count_classed++;
		}
		cls++;
		if (count_classed >= data.size()) {
			break;
		}
	}
	return C;
}

vector<int>core_fusen(int id, double eps, vector<vector<Point>>data, int*classed) {

	vector<int>core_id;
	vector<int>neibor_id = neibor_fusen(id, eps, data, classed);
	core_id.push_back(id);
	classed[id] = true;
	if (neibor_id.size() == 0) {
		return core_id;
	}
	else {
		int i;
		for (i = 0; i < neibor_id.size(); i++) {
			classed[neibor_id[i]] = true;
		}
		for (i = 0; i < neibor_id.size(); i++) {
			vector<int> temp = core_fusen(neibor_id[i], eps, data, classed);
			int j;
			for (j = 0; j < temp.size(); j++) {
				core_id.push_back(temp[j]);
			}
		}
	}
	return core_id;
}
vector<int>neibor_fusen(int id, double eps, vector<vector<Point>>data, int*classed) {
	int i = 0;
	vector<int>neibor_id;
	for (i = 0; i < data.size(); i++) {
		if (i != id) {
			if (classed[i] == false) {
				if (fusen_dict(data[id], data[i]) <= eps) {
					neibor_id.push_back(i);
				}
			}
		}
	}
	return neibor_id;
}

double fusen_dict(vector<Point>points1, vector<Point>points2) {
	int i, j;
	double min_dict = -1;
	double dict;
	for (i = 0; i < points1.size(); i++) {
		for (j = 0; j < points2.size(); j++) {
			int gseki1 = gaiseki(points1[i] - points2[j], points2[(j + 1) % 4] - points2[j]);
			int gseki2 = gaiseki(points1[(i + 1) % 4] - points2[j], points2[(j + 1) % 4] - points2[j]);
			int gseki3 = gaiseki(points2[j] - points1[i], points1[(i + 1) % 4] - points1[i]);
			int gseki4 = gaiseki(points2[(j + 1) % 4] - points1[i], points1[(i + 1) % 4] - points1[i]);
			if (gseki1 / 1.0 * gseki2 / 1.0 < 0) {
				if (gseki3 / 1.0 * gseki4 / 1.0 < 0) {
					return 0;
				}
			}
			dict = tentosen_dict(points1[i], points2[j], points2[(j + 1) % 4]);
			if (min_dict == -1 || dict < min_dict) {
				min_dict = dict;
			}
			dict = tentosen_dict(points2[i], points1[j], points1[(j + 1) % 4]);
			if (min_dict == -1 || dict < min_dict) {
				min_dict = dict;
			}
		}
	}
	//printf("%lf\n", min_dict);
	return min_dict;
}

double tentosen_dict(Point q1, Point p1, Point p2) {
	int nseki = naiseki(q1 - p1, p2 - p1);
	int nseki2 = naiseki(p2 - p1, p2 - p1);

	if (nseki < 0) {
		return (double)norm(p1 - q1);
	}
	else if (nseki <nseki2) {
		double d2 = nseki / sqrt((double)nseki2);
		return sqrt(naiseki(q1 - p1, q1 - p1) - pow(d2, 2));
	}
	else {
		return (double)norm(p2 - q1);
	}

}
