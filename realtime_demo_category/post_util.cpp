#include "stdafx.h"

#include <opencv2/core/core.hpp>    // coreモジュールのヘッダーをインクルード
#include <opencv2/highgui/highgui.hpp> // highguiモジュールのヘッダーをインクルード
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include <mutex>
#include <fstream>


#include "analyze_config.h"
#include "read_config.h"
#include "postit_config.h"

#include "post_util.h"
#include "realtime_demo_category.h"
#pragma comment (lib, "WinMM.Lib")




using namespace std;

void miru(Mat m) {
	int i,j;
	int rows = m.rows;
	int cols = m.cols;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {

			if (int(m.at<unsigned char>(i, j))) {
				cout << "1";
			}
			else {
				cout << "0";
			}

			//cout << m.at<double>(i, j) << " ";
		}
		cout << "\n";
	}
	cout << "\n";
}

void Timer(LARGE_INTEGER &prev_timer, string label) {
	LARGE_INTEGER now_timer;
	QueryPerformanceCounter(&now_timer);
	extern LARGE_INTEGER freq;
	zikken_output << label << "," << double(now_timer.QuadPart - prev_timer.QuadPart)*1000/freq.QuadPart << endl;
	//cout << label << "," << double(now_timer.QuadPart - prev_timer.QuadPart) * 1000 / freq.QuadPart << endl;
	QueryPerformanceCounter(&prev_timer);
}

double sum_d(vector<Point2f> point1, vector<Point2f>point2) {
	int size = point1.size();
	int i;
	double min_d;
	for (i = 0; i < size; i++) {
		double d = 0;
		for (j = 0; j < size; j++) {
			norm(point1[(i+j)%size] - point2[j])
		}
	}
}


void getPostits(Postits * postits, cv::Mat frame, int outer_size) {
	LARGE_INTEGER recognition_start;
	LARGE_INTEGER now_timer, prev_timer;
	QueryPerformanceCounter(&recognition_start);
	prev_timer = recognition_start;
	Mat show_img;
	if (kakunin) {
		cv::namedWindow("show", cv::WINDOW_AUTOSIZE);
		imwrite("frame_dame.jpg", frame);
		//*
		Mat frame_for_select(int(frame.rows * 0.5), int(frame.cols * 0.5), frame.type());
		cv::resize(frame, frame_for_select, frame_for_select.size());
		namedWindow("frame", CV_WINDOW_AUTOSIZE);
		cv::flip(frame_for_select, frame_for_select, -1);
		cv::imshow("frame", frame_for_select);
		waitKey(1);
		//*/
	}

	

	//frame.copyTo(frame_original);
	Mat frame_original;
	if (kakunin == true) {
		frame_original = Mat(frame.rows, frame.cols, frame.type());
		frame.copyTo(frame_original);
	}
	else {
		frame_original = frame;
	}
	Timer(prev_timer, " recognition コピー : ");
	/*
	namedWindow("original", CV_WINDOW_AUTOSIZE);
	imshow("original", frame_original);
	waitKey(1);
	cout << CV_8UC3;
	cout << frame.type();
	//cv::Mat grayImage(frame.rows, frame.cols, CV_8UC1);
	//*/

	Mat grayImage;
	cv::cvtColor(frame, grayImage, CV_RGB2GRAY);
	Timer(prev_timer, "recognition グレー化");
	cv::Mat binImage(frame.rows, frame.cols, CV_8UC1);
	cv::adaptiveThreshold(grayImage, binImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 9);
	//imwrite("mybinImageha.jpg", binImage);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Timer(prev_timer, "recognition 2値化:");


	cv::findContours(binImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	
	Timer(prev_timer, "recognition 輪郭抽出");


	vector<vector<vector<Point2f>>> location_xy(8);
	std::mutex mtx_location;
	int i;
	int ss = 0;

	vector<RotatedRect> rects;

	for (i = 0; i < contours.size(); i++) {
		vector<Point> contour = contours[i];
		cv::RotatedRect rect = minAreaRect(contour);
		float area = rect.size.area();
		vector<Point2f> rect_point(4);
		vector<Point> rect_int_point(4);
		rect.points(&rect_point[0]);
		int mi;
		float outer_lower = OUTER_LOWER;
		float outer_upper = OUTER_UPPER;
		for (mi = 0; mi < 4; mi++) {
			rect_int_point[mi] = Point((int)rect_point[mi].x, (int)rect_point[mi].y);
		}
		//cout << area << "\n";
		if (outer_size * outer_lower < area && area < outer_size * outer_upper) {
			int idx = hierarchy[i][2];
			if (idx != -1){
				vector<Point2f> rect_point;
				rect.points(&rect_point[0]);
				extern vector<vector<vector<Point2f>>> before_location_points;
				int location_id, nlocation;
				for (location_id = 0; location_id < before_location_points.size(); location_id++) {
					for (nlocation = 0; nlocation < before_location_points[location_id].size(); nlocation++) {
						vector<Point2f> before_rect = before_location_points[location_id][nlocation];
						double dict = sum_d(before_rect, rect_point);
						if (dict < 100) {
							location_xy[location_id].push_back(rect_point);
							before_location_points[location_id].erase(before_location_points.begin + nlocation);
						}
					}
				}
#pragma omp critical
				{
					rects.push_back(rect);
				}
			}
		}
	}
	int mj;
	extern bool iti_heiretu;
	if (iti_heiretu) {
		//やること　長方形抽出
		int rows = frame_original.rows;
		int cols = frame_original.cols;
		#pragma omp parallel for
		for (mj = 0; mj < rects.size(); mj++) {
			//LARGE_INTEGER hei_timer, hei1_timer;
			//QueryPerformanceCounter(&hei_timer);
			

			cv::RotatedRect rect = rects[mj];
			float area = rect.size.area();
			vector<Point2f> rect_point(4);
			vector<Point> rect_int_point(4);
			rect.points(&rect_point[0]);
			int mi;
			for (mi = 0; mi < 4; mi++) {
				rect_int_point[mi] = Point((int)rect_point[mi].x, (int)rect_point[mi].y);
			}
			float wh_ratio = rect.size.width / rect.size.height;
			if (wh_ratio < 1) {
				wh_ratio = 1 / wh_ratio;
			}
			if (wh_ratio > 2.5) {
				continue;
			}
			//extract larger area
			Point2f box[4];
			rect.points(box);
			Point2f max_point;
			Point2f min_point;
			max_point = max2f(box, 4, 0);
			min_point = min2f(box, 4, 0);

			float expand_ratio = EXPAND_RATIO;
			int point_buffer = int(POINT_BUFFER * expand_ratio); //used when searching in larger area and extract larger area of location point area

			float point_buffer_for_larger = point_buffer / expand_ratio * pow(outer_size / 220, 0.5);
			int min_y = max(0, int(min_point.y - point_buffer_for_larger));
			int max_y = min(rows, int(max_point.y + point_buffer_for_larger));
			int min_x = max(0, int(min_point.x - point_buffer_for_larger));
			int max_x = min(cols, int(max_point.x + point_buffer_for_larger));
			cv::Rect larger_area_rect(min_x, min_y, max_x - min_x, max_y - min_y);
			cv::Mat larger_area(frame_original, larger_area_rect);
			cv::Mat larger_area_grayImage(larger_area.size(), CV_8UC1);
			cv::cvtColor(larger_area, larger_area_grayImage, CV_RGB2GRAY);
			cv::Mat bin_larger_area(larger_area.size(), CV_8UC1);
			cv::threshold(larger_area_grayImage, bin_larger_area, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

			bool flag_changed = false;
			//find contours inlarger area
			std::vector<std::vector<Point>> contours_in_larger;
			std::vector<Vec4i>hierarchy_in_larger;
			cv::findContours(bin_larger_area, contours_in_larger, hierarchy_in_larger, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
			int j;

			float outer_lower = OUTER_LOWER;
			float outer_upper = OUTER_UPPER;

			for (j = 0; j < contours_in_larger.size(); j++) {
				cv::RotatedRect rect_in_larger;
				vector<Point> contour_in_larger;
				contour_in_larger = contours_in_larger[j];
				rect_in_larger = minAreaRect(contour_in_larger);
				float area_in_larger = rect_in_larger.size.area();
				Point2f box_in_larger[4];
				rect_in_larger.points(box_in_larger);
				if (outer_size * outer_lower < area_in_larger && area_in_larger < outer_size * outer_upper) {
					int idx = hierarchy_in_larger[j][2];
					if (idx != -1) {
						int k;
						for (k = 0; k < 4; k++) {
							box[k] = Point2f(min_point.x - point_buffer_for_larger + box_in_larger[k].x,
								min_point.y - point_buffer_for_larger + box_in_larger[k].y);
						}
						flag_changed = true;
					}
				}
			}
			//find candidate of left-top
			vector<int> candidate;
			for (j = 0; j < 4; j++) {
				if (norm(box[j] - box[(j + 1) % 4]) > norm(box[(j + 1) % 4] - box[(j + 2) % 4])) {
					candidate.push_back(j);
				}
			}

			const int location_dot_num = LOCATION_DOT_NUM;
			int x_buffer = int(X_BUFFER * expand_ratio);//from rectangle's edge
			int y_buffer = int(Y_BUFFER * expand_ratio); //from rectangle's edge
			int space_x = int(SPACE_X * expand_ratio);
			int rect_len = int(RECT_LEN * expand_ratio);
			int location_width = x_buffer * 2 + space_x * (location_dot_num - 1) + rect_len;
			int location_height = int(LOCATION_HEIGHT * expand_ratio);
			int line_width = int(LINE_WIDTH * expand_ratio);
			int dot_read_thre = DOT_READ_THRE;
			int dot_read_area = int(DOT_READ_AREA * expand_ratio);

			//read each location's id
			for (j = 0; j < candidate.size(); j++) {
				int id = candidate[j];
				vector<Point2f> before_points;
				int k;
				for (k = 0; k < 4; k++) {
					before_points.push_back(box[(k + id) % 4]);
				}
				RotatedRect before_points_rotatedrect;
				before_points_rotatedrect = cv::minAreaRect(before_points);
				Rect before_points_rect;
				before_points_rect = before_points_rotatedrect.boundingRect();
				if (before_points_rect.x < 0) {
					before_points_rect.x = 0;
				}
				if (before_points_rect.y < 0) {
					before_points_rect.y = 0;
				}
				if (before_points_rect.width + before_points_rect.x > cols) {
					before_points_rect.width = cols - before_points_rect.x;
				}
				if (before_points_rect.height + before_points_rect.y > rows) {
					before_points_rect.height = rows - before_points_rect.y;
				}
				for (k = 0; k < 4; k++) {
					before_points[k].x -= larger_area_rect.x;
					before_points[k].y -= larger_area_rect.y;
				}

				/*Mat before_points_area(larger_area, Rect(before_points_rect.tl() - larger_area_rect.tl(),
					Size(before_points_rect.width, before_points_rect.height)));*/
				//Mat before_points_area(frame_original, before_points_rect);
				vector<Point2f> after_points{
					Point2f(0 + point_buffer, 0 + point_buffer),
					Point2f(location_width + point_buffer, 0 + point_buffer),
					Point2f(location_width + point_buffer, location_height + point_buffer),
					Point2f(0 + point_buffer, location_height + point_buffer)
				};
				/*
				cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
				cv::imshow("out", before_points_area);
				cv::waitKey(1);
				//*/
				cv::Mat M = cv::getPerspectiveTransform(before_points, after_points);
				cv::Mat dst(location_height + point_buffer * 2, location_width + point_buffer * 2, CV_8UC1);
				cv::warpPerspective(larger_area, dst, M, dst.size());

				cv::Mat dst_grayImage(dst.size(), CV_8UC1);
				cv::Mat dst_binImage(dst.size(), CV_8UC1);
				cv::cvtColor(dst, dst_grayImage, CV_RGB2GRAY);
				cv::threshold(dst_grayImage, dst_binImage, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
				/*
				cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
				cv::imshow("out", dst_binImage);
				cv::waitKey(1);
				//*/

				Point2f box_a_sorted[4];
				if (flag_changed == false) {

					Mat dst_copy(dst_binImage.size(), CV_8UC1);
					dst_binImage.copyTo(dst_copy);

					std::vector<std::vector<Point>> contours_a;
					std::vector<Vec4i>hierarchy_a;
					cv::findContours(dst_binImage, contours_a, hierarchy_a, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

					//find contours again
					for (k = 0; k < contours_a.size(); k++) {
						RotatedRect rect_a = cv::minAreaRect(contours_a[k]);
						Point2f box_a[4];
						float area_a;
						rect_a.points(box_a);
						area_a = rect_a.size.area();
						int idx_a = hierarchy_a[k][2];
						float location_area_size = location_width * location_height;
						if (idx_a != -1
							&& area_a > location_area_size * outer_lower
							&& area_a < location_area_size * outer_upper) {
							vector<vector<Point>> contour(1);
							contour[0] = contours_a[k];
							//cv::drawContours(frame, contour, 0, (0, 255, 255), 2);
							//calc left-top
							int l;
							int box_left_top_idx;
							for (l = 0; l < 4; l++) {
								if (box_a[l].x < rect_a.center.x && box_a[l].y < rect_a.center.y) {
									box_left_top_idx = l;
									break;
								}
							}
							for (l = 0; l < 4; l++) {
								box_a_sorted[l] = box_a[(l + box_left_top_idx) % 4];
							}
						}
						else {
							int l;
							for (l = 0; l < 4; l++) {
								box_a_sorted[l] = Point(0, 0);
							}
						}
					}
				}
				else {
					for (k = 0; k < 4; k++) {
						box_a_sorted[k] = after_points[k];
					}
				}
				if (box_a_sorted[1].x == 0) {
					continue;
				}

				//read each point
				float area_width = max2f(box_a_sorted, 4, 0).x - min2f(box_a_sorted, 4, 0).x;
				float area_height = max2f(box_a_sorted, 4, 0).y - min2f(box_a_sorted, 4, 0).y;
				float center_x = mean2f(box_a_sorted, 4, 0).x;
				float location_ratio = float(area_width + area_height) / (location_width + location_height + line_width * 2);
				float space_x_mod = space_x * location_ratio;
				float to_center_dst = (location_width / 2.0 - (x_buffer + rect_len / 2.0 + space_x * 2))*location_ratio;

				bool dot_point[location_dot_num];

				for (k = 0; k < location_dot_num; k++) {
					dot_point[k] = 0;
					int x = max(0, int(center_x - to_center_dst + space_x_mod * (k - 2) - dot_read_area));
					int y = max(0, int(box_a_sorted[0].y + area_height / 2 - dot_read_area));
					int width = 2 * dot_read_area;
					int height = 2 * dot_read_area;
					/*
					if (x + width < dst_binImage.cols) {
					x = min(x, dst_binImage.cols);
					width = dst_binImage.cols - x;
					}
					if (y + height < dst_binImage.rows) {
					y = min(y, dst_binImage.rows);
					height = dst_binImage.rows - y;
					}
					*/
					cv::Rect dst_rect(x,
						y,
						width,
						height);
					cv::Mat dst_area(dst_binImage, dst_rect);
					//miru(dst_area);
					double ha = cv::mean(dst_area).val[0];
					if (cv::mean(dst_area).val[0] < double(dot_read_thre)) {
						dot_point[k] = 1;
					}

				}
				if (dot_point[0] == 1 && dot_point[location_dot_num - 1] == 0) {
					vector<Point2f>  box_a_after(4);

					//calc dot id
					int dot_id = 0;
					int m;
					for (m = 1; m < location_dot_num - 1; m++) {
						if (dot_point[m] == 1) {
							dot_id += int(pow(2, m - 1));
						}
					}
					//homography_inv
					vector<Point2f> after_points(4);
					for (m = 0; m < 4; m++) {
						after_points[m] = box[(m + id) % 4];
					}
					vector<Point2f>before_points{
						Point2f(0 + point_buffer, 0 + point_buffer),
						Point2f(location_width + point_buffer, 0 + point_buffer),
						Point2f(location_width + point_buffer, location_height + point_buffer),
						Point2f(0 + point_buffer, location_height + point_buffer)
					};
					cv::Mat M_inv(3, 3, CV_64FC1);
					M_inv = getPerspectiveTransform(before_points, after_points);

					for (m = 0; m < 4; m++) {
						cv::Mat box_3d = (cv::Mat_<double>(3, 1) << box_a_sorted[m].x, box_a_sorted[m].y, 1);
						//Mat box_3d(3, 1, CV_64FC1, { before_points[m].x, before_points[m].y, 1 });
						cv::Mat box_a_after_each(3, 1, CV_64FC1);
						box_a_after_each = M_inv * box_3d;
						//cv::warpPerspective(box_3d[m], box_a_after_each, M_inv, box_a_after_each.size());
						float x = float(box_a_after_each.at<double>(0, 0));
						float y = float(box_a_after_each.at<double>(1, 0));
						box_a_after[m].x = x;
						box_a_after[m].y = y;
						if (kakunin)
							cv::circle(frame, Point(int(box_a_after_each.at<double>(0, 0)), int(box_a_after_each.at<double>(1, 0))), 5, Scalar(100, 100, 100), 5);
					}
					mtx_location.lock();
					location_xy[dot_id].push_back(box_a_after);
					mtx_location.unlock();
					//write each id in this point
					if (kakunin) {
						cv::putText(frame, to_string(dot_id),
							Point(int(mean2f(&box_a_after[0], 4, 0).x), int(mean2f(&box_a_after[0], 4, 0).y)),
							CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 255, 0), 5);
					}
				}
			}
			//QueryPerformanceCounter(&hei1_timer);
			//LARGE_INTEGER freq;
			//QueryPerformanceFrequency(&freq);
//#pragma omp critical
			//{
			//	cout << "hei 1loop" << "," << double(hei1_timer.QuadPart - hei_timer.QuadPart) * 1000 / freq.QuadPart << endl;
			//}
		}

		//#pragma omp parallel for
		//for (mj = 0; mj < marker_contours.size(); mj++) {
			/*vector<Point> contour = marker_contours[mj];
			cv::RotatedRect rect = minAreaRect(contour);
			float area = rect.size.area();
			vector<Point2f> rect_point(4);
			vector<Point> rect_int_point(4);
			rect.points(&rect_point[0]);
			int mi;
			for (mi = 0; mi < 4; mi++) {
				rect_int_point[mi] = Point((int)rect_point[mi].x, (int)rect_point[mi].y);
			}*/
			//if(100 < area && area < 3000){

			/*
			//cout << area << "\n";
			cv::drawContours(frame, contours, i, Scalar(255, 0, 255), 1.0);
			//cv::drawContours(frame, vector<vector<Point>>{rect_int_point}, 0, Scalar(0, 0, 255), 5.0);
			cv::putText(frame, to_string(int(area)), contours[i][0], CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 128, 0), 3);
			//*
			show_img = cv::Mat(cv::Size(frame.cols / 5, frame.rows / 5), CV_8UC3);
			cv::resize(frame, show_img, show_img.size());
			cv::imshow("show", show_img);
			cv::waitKey(1);
			//*/
			/*float wh_ratio = rect.size.width / rect.size.height;
			if (wh_ratio < 1) {
				wh_ratio = 1 / wh_ratio;
			}
			if (wh_ratio > 2.5) {
				continue;
			}

			if (kakunin) {
				cv::drawContours(frame, contours, i, Scalar(255, 0, 255), 1.0);
				cv::drawContours(frame, vector<vector<Point>>{rect_int_point}, 0, Scalar(0, 255, 255), 5.0);
				cv::putText(frame, to_string(int(area)), contours[i][0], CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 128, 255), 2.0);
			}*/
			//extract larger area
			/*Point2f box[4];
			rect.points(box);
			Point2f max_point;
			Point2f min_point;
			max_point = max2f(box, 4, 0);
			min_point = min2f(box, 4, 0);
			float point_buffer_for_larger = point_buffer / expand_ratio * pow(outer_size / 220, 0.5);
			int min_y = max(0, int(min_point.y - point_buffer_for_larger));
			int max_y = min(frame_original.rows, int(max_point.y + point_buffer_for_larger));
			int min_x = max(0, int(min_point.x - point_buffer_for_larger));
			int max_x = min(frame_original.cols, int(max_point.x + point_buffer_for_larger));
			cv::Rect larger_area_rect(min_x, min_y, max_x - min_x, max_y - min_y);
			cv::Mat larger_area(frame_original, larger_area_rect);
			cv::Mat larger_area_grayImage(larger_area.size(), CV_8UC1);
			cv::cvtColor(larger_area, larger_area_grayImage, CV_RGB2GRAY);
			cv::Mat bin_larger_area(larger_area.size(), CV_8UC1);
			cv::threshold(larger_area_grayImage, bin_larger_area, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);*/

			/*
			namedWindow("larger_area", CV_WINDOW_AUTOSIZE);
			imshow("larger_area", frame_original);
			waitKey(1);
			//*/


			//find contours inlarger area
			/*bool flag_changed = false;

			std::vector<std::vector<Point>> contours_in_larger;
			std::vector<Vec4i>hierarchy_in_larger;
			cv::findContours(bin_larger_area, contours_in_larger, hierarchy_in_larger, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
			int j;
			for (j = 0; j != contours_in_larger.size(); j++) {
				cv::RotatedRect rect_in_larger;
				vector<Point> contour_in_larger;
				contour_in_larger = contours_in_larger[j];
				rect_in_larger = minAreaRect(contour_in_larger);
				float area_in_larger = rect_in_larger.size.area();
				Point2f box_in_larger[4];
				rect_in_larger.points(box_in_larger);
				if (outer_size * outer_lower < area_in_larger && area_in_larger < outer_size * outer_upper) {
					int idx = hierarchy_in_larger[j][2];
					if (idx != -1) {
						int k;
						for (k = 0; k < 4; k++) {
							box[k] = Point2f(min_point.x - point_buffer_for_larger + box_in_larger[k].x,
								min_point.y - point_buffer_for_larger + box_in_larger[k].y);
						}
						flag_changed = true;
					}
				}
			}*/


			//find candidate of left-top
		/*	vector<int> candidate;
			for (j = 0; j < 4; j++) {
				if (norm(box[j] - box[(j + 1) % 4]) > norm(box[(j + 1) % 4] - box[(j + 2) % 4])) {
					candidate.push_back(j);
				}
			}*/


			//read each location's id
			//for (j = 0; j < candidate.size(); j++) {
			//	/*
			//	cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
			//	cv::imshow("out", frame_original);
			//	cv::waitKey(1);
			//	*/

			//	int id = candidate[j];
			//	vector<Point2f> before_points;
			//	int k;
			//	for (k = 0; k < 4; k++) {
			//		before_points.push_back(box[(k + id) % 4]);
			//	}
			//	RotatedRect before_points_rotatedrect;
			//	before_points_rotatedrect = minAreaRect(before_points);
			//	Rect before_points_rect;
			//	before_points_rect = before_points_rotatedrect.boundingRect();
			//	if (before_points_rect.x < 0) {
			//		before_points_rect.x = 0;
			//	}
			//	if (before_points_rect.y < 0) {
			//		before_points_rect.y = 0;
			//	}
			//	if (before_points_rect.width + before_points_rect.x > frame_original.cols) {
			//		before_points_rect.width = frame_original.cols - before_points_rect.x;
			//	}
			//	if (before_points_rect.height + before_points_rect.y > frame_original.rows) {
			//		before_points_rect.height = frame_original.rows - before_points_rect.y;
			//	}

			//	Mat before_points_area(frame_original, before_points_rect);
			//	vector<Point2f> after_points{
			//		Point2f(0 + point_buffer, 0 + point_buffer),
			//		Point2f(location_width + point_buffer, 0 + point_buffer),
			//		Point2f(location_width + point_buffer, location_height + point_buffer),
			//		Point2f(0 + point_buffer, location_height + point_buffer)
			//	};
			//	/*
			//	cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
			//	cv::imshow("out", before_points_area);
			//	cv::waitKey(1);
			//	//*/
			//	cv::Mat M = cv::getPerspectiveTransform(before_points, after_points);
			//	cv::Mat dst(location_height + point_buffer * 2, location_width + point_buffer * 2, frame_original.type());
			//	cv::warpPerspective(frame_original, dst, M, dst.size());

			//	cv::Mat dst_grayImage(dst.size(), CV_8UC1);
			//	cv::Mat dst_binImage(dst.size(), CV_8UC1);
			//	cv::cvtColor(dst, dst_grayImage, CV_RGB2GRAY);
			//	cv::threshold(dst_grayImage, dst_binImage, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//	/*
			//	cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
			//	cv::imshow("out", dst_binImage);
			//	cv::waitKey(1);
			//	//*/

			//	Point2f box_a_sorted[4];
			//	if (flag_changed == false) {

			//		Mat dst_copy(dst_binImage.size(), CV_8UC1);
			//		dst_binImage.copyTo(dst_copy);

			//		std::vector<std::vector<Point>> contours_a;
			//		std::vector<Vec4i>hierarchy_a;
			//		cv::findContours(dst_binImage, contours_a, hierarchy_a, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

			//		//find contours again
			//		for (k = 0; k < contours_a.size(); k++) {
			//			RotatedRect rect_a = cv::minAreaRect(contours_a[k]);
			//			Point2f box_a[4];
			//			float area_a;
			//			rect_a.points(box_a);
			//			area_a = rect_a.size.area();
			//			int idx_a = hierarchy_a[k][2];
			//			float location_area_size = location_width * location_height;
			//			if (idx_a != -1
			//				&& area_a > location_area_size * outer_lower
			//				&& area_a < location_area_size * outer_upper) {
			//				vector<vector<Point>> contour(1);
			//				contour[0] = contours_a[k];
			//				//cv::drawContours(frame, contour, 0, (0, 255, 255), 2);
			//				//calc left-top
			//				int l;
			//				int box_left_top_idx;
			//				for (l = 0; l < 4; l++) {
			//					if (box_a[l].x < rect_a.center.x && box_a[l].y < rect_a.center.y) {
			//						box_left_top_idx = l;
			//						break;
			//					}
			//				}
			//				for (l = 0; l < 4; l++) {
			//					box_a_sorted[l] = box_a[(l + box_left_top_idx) % 4];
			//				}
			//			}
			//			else {
			//				int l;
			//				for (l = 0; l < 4; l++) {
			//					box_a_sorted[l] = Point(0, 0);
			//				}
			//			}
			//		}
			//	}
			//	else {
			//		for (k = 0; k < 4; k++) {
			//			box_a_sorted[k] = after_points[k];
			//		}
			//	}
			//	if (box_a_sorted[1].x == 0) {
			//		continue;
			//	}

			//	//read each point
			//	float area_width = max2f(box_a_sorted, 4, 0).x - min2f(box_a_sorted, 4, 0).x;
			//	float area_height = max2f(box_a_sorted, 4, 0).y - min2f(box_a_sorted, 4, 0).y;
			//	float center_x = mean2f(box_a_sorted, 4, 0).x;
			//	float location_ratio = float(area_width + area_height) / (location_width + location_height + line_width * 2);
			//	float space_x_mod = space_x * location_ratio;
			//	float to_center_dst = (location_width / 2.0 - (x_buffer + rect_len / 2.0 + space_x * 2))*location_ratio;

			//	bool dot_point[location_dot_num];

			//	for (k = 0; k < location_dot_num; k++) {
			//		dot_point[k] = 0;
			//		int x = max(0, int(center_x - to_center_dst + space_x_mod * (k - 2) - dot_read_area));
			//		int y = max(0, int(box_a_sorted[0].y + area_height / 2 - dot_read_area));
			//		int width = 2 * dot_read_area;
			//		int height = 2 * dot_read_area;
			//		/*
			//		if (x + width < dst_binImage.cols) {
			//		x = min(x, dst_binImage.cols);
			//		width = dst_binImage.cols - x;
			//		}
			//		if (y + height < dst_binImage.rows) {
			//		y = min(y, dst_binImage.rows);
			//		height = dst_binImage.rows - y;
			//		}
			//		*/
			//		cv::Rect dst_rect(x,
			//			y,
			//			width,
			//			height);
			//		cv::Mat dst_area(dst_binImage, dst_rect);
			//		//miru(dst_area);
			//		double ha = cv::mean(dst_area).val[0];
			//		if (cv::mean(dst_area).val[0] < double(dot_read_thre)) {
			//			dot_point[k] = 1;
			//		}

			//	}
			//	if (dot_point[0] == 1 && dot_point[location_dot_num - 1] == 0) {
			//		vector<Point2f>  box_a_after(4);

			//		//calc dot id
			//		int dot_id = 0;
			//		int m;
			//		for (m = 1; m < location_dot_num - 1; m++) {
			//			if (dot_point[m] == 1) {
			//				dot_id += int(pow(2, m - 1));
			//			}
			//		}
			//		//homography_inv
			//		vector<Point2f> after_points(4);
			//		for (m = 0; m < 4; m++) {
			//			after_points[m] = box[(m + id) % 4];
			//		}
			//		vector<Point2f>before_points{
			//			Point2f(0 + point_buffer, 0 + point_buffer),
			//			Point2f(location_width + point_buffer, 0 + point_buffer),
			//			Point2f(location_width + point_buffer, location_height + point_buffer),
			//			Point2f(0 + point_buffer, location_height + point_buffer)
			//		};
			//		cv::Mat M_inv(3, 3, CV_64FC1);
			//		M_inv = getPerspectiveTransform(before_points, after_points);

			//		for (m = 0; m < 4; m++) {
			//			cv::Mat box_3d = (cv::Mat_<double>(3, 1) << box_a_sorted[m].x, box_a_sorted[m].y, 1);
			//			//Mat box_3d(3, 1, CV_64FC1, { before_points[m].x, before_points[m].y, 1 });
			//			cv::Mat box_a_after_each(3, 1, CV_64FC1);
			//			box_a_after_each = M_inv * box_3d;
			//			//cv::warpPerspective(box_3d[m], box_a_after_each, M_inv, box_a_after_each.size());
			//			float x = float(box_a_after_each.at<double>(0, 0));
			//			float y = float(box_a_after_each.at<double>(1, 0));
			//			box_a_after[m].x = x;
			//			box_a_after[m].y = y;
			//			if (kakunin)
			//				cv::circle(frame, Point(int(box_a_after_each.at<double>(0, 0)), int(box_a_after_each.at<double>(1, 0))), 5, Scalar(100, 100, 100), 5);
			//		}
			//		mtx_location.lock();
			//		location_xy[dot_id].push_back(box_a_after);
			//		mtx_location.unlock();
			//		//printf("aaa");
			//		//write each id in this point
			//		if (kakunin) {
			//			cv::putText(frame, to_string(dot_id),
			//				Point(int(mean2f(&box_a_after[0], 4, 0).x), int(mean2f(&box_a_after[0], 4, 0).y)),
			//				CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 255, 0), 5);
			//		}
			//	}
			//}
		//}
	}
	else {
	for (mj = 0; mj < marker_contours.size(); mj++) {
		//LARGE_INTEGER tyoku_timer;
		//QueryPerformanceCounter(&tyoku_timer);
		int rows = frame_original.rows;
		int cols = frame_original.cols;
		float expand_ratio = EXPAND_RATIO;
		const int location_dot_num = LOCATION_DOT_NUM;
		int x_buffer = int(X_BUFFER * expand_ratio);//from rectangle's edge
		int y_buffer = int(Y_BUFFER * expand_ratio); //from rectangle's edge
		int space_x = int(SPACE_X * expand_ratio);
		int rect_len = int(RECT_LEN * expand_ratio);
		int location_width = x_buffer * 2 + space_x * (location_dot_num - 1) + rect_len;
		int location_height = int(LOCATION_HEIGHT * expand_ratio);
		int line_width = int(LINE_WIDTH * expand_ratio);

		//other parameters
		int point_buffer = int(POINT_BUFFER * expand_ratio); //used when searching in larger area and extract larger area of location point area
		float outer_lower = OUTER_LOWER;
		float outer_upper = OUTER_UPPER;
		int dot_read_thre = DOT_READ_THRE;
		int dot_read_area = int(DOT_READ_AREA * expand_ratio);
		vector<Point> contour = marker_contours[mj];
		cv::RotatedRect rect = minAreaRect(contour);
		float area = rect.size.area();
		vector<Point2f> rect_point(4);
		vector<Point> rect_int_point(4);
		rect.points(&rect_point[0]);
		int mi;
		for (mi = 0; mi < 4; mi++) {
			rect_int_point[mi] = Point((int)rect_point[mi].x, (int)rect_point[mi].y);
		}
		float wh_ratio = rect.size.width / rect.size.height;
		if (wh_ratio < 1) {
			wh_ratio = 1 / wh_ratio;
		}
		if (wh_ratio > 2.5) {
			continue;
		}
		//extract larger area
		Point2f box[4];
		rect.points(box);
		Point2f max_point;
		Point2f min_point;
		max_point = max2f(box, 4, 0);
		min_point = min2f(box, 4, 0);
		float point_buffer_for_larger = point_buffer / expand_ratio * pow(outer_size / 220, 0.5);
		int min_y = max(0, int(min_point.y - point_buffer_for_larger));
		int max_y = min(rows, int(max_point.y + point_buffer_for_larger));
		int min_x = max(0, int(min_point.x - point_buffer_for_larger));
		int max_x = min(cols, int(max_point.x + point_buffer_for_larger));
		cv::Rect larger_area_rect(min_x, min_y, max_x - min_x, max_y - min_y);
		cv::Mat larger_area(frame_original, larger_area_rect);
		cv::Mat larger_area_grayImage(larger_area.size(), CV_8UC1);
		cv::cvtColor(larger_area, larger_area_grayImage, CV_RGB2GRAY);
		cv::Mat bin_larger_area(larger_area.size(), CV_8UC1);
		cv::threshold(larger_area_grayImage, bin_larger_area, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

		bool flag_changed = false;
		//find contours inlarger area
		std::vector<std::vector<Point>> contours_in_larger;
		std::vector<Vec4i>hierarchy_in_larger;
		cv::findContours(bin_larger_area, contours_in_larger, hierarchy_in_larger, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		int j;
		for (j = 0; j < contours_in_larger.size(); j++) {
			cv::RotatedRect rect_in_larger;
			vector<Point> contour_in_larger;
			contour_in_larger = contours_in_larger[j];
			rect_in_larger = minAreaRect(contour_in_larger);
			float area_in_larger = rect_in_larger.size.area();
			Point2f box_in_larger[4];
			rect_in_larger.points(box_in_larger);
			if (outer_size * outer_lower < area_in_larger && area_in_larger < outer_size * outer_upper) {
				int idx = hierarchy_in_larger[j][2];
				if (idx != -1) {
					int k;
					for (k = 0; k < 4; k++) {
						box[k] = Point2f(min_point.x - point_buffer_for_larger + box_in_larger[k].x,
							min_point.y - point_buffer_for_larger + box_in_larger[k].y);
					}
					flag_changed = true;
				}
			}
		}
		//find candidate of left-top
		vector<int> candidate;
		for (j = 0; j < 4; j++) {
			if (norm(box[j] - box[(j + 1) % 4]) > norm(box[(j + 1) % 4] - box[(j + 2) % 4])) {
				candidate.push_back(j);
			}
		}

		//read each location's id
		for (j = 0; j < candidate.size(); j++) {
			int id = candidate[j];
			vector<Point2f> before_points;
			int k;
			for (k = 0; k < 4; k++) {
				before_points.push_back(box[(k + id) % 4]);
			}
			RotatedRect before_points_rotatedrect;
			before_points_rotatedrect = cv::minAreaRect(before_points);
			Rect before_points_rect;
			before_points_rect = before_points_rotatedrect.boundingRect();
			if (before_points_rect.x < 0) {
				before_points_rect.x = 0;
			}
			if (before_points_rect.y < 0) {
				before_points_rect.y = 0;
			}
			if (before_points_rect.width + before_points_rect.x > cols) {
				before_points_rect.width = cols - before_points_rect.x;
			}
			if (before_points_rect.height + before_points_rect.y > rows) {
				before_points_rect.height = rows - before_points_rect.y;
			}
			for (k = 0; k < 4; k++) {
				before_points[k].x -= larger_area_rect.x;
				before_points[k].y -= larger_area_rect.y;
			}

			/*Mat before_points_area(larger_area, Rect(before_points_rect.tl() - larger_area_rect.tl(),
				Size(before_points_rect.width, before_points_rect.height)));*/
				//Mat before_points_area(frame_original, before_points_rect);
			vector<Point2f> after_points{
				Point2f(0 + point_buffer, 0 + point_buffer),
				Point2f(location_width + point_buffer, 0 + point_buffer),
				Point2f(location_width + point_buffer, location_height + point_buffer),
				Point2f(0 + point_buffer, location_height + point_buffer)
			};
			/*
			cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
			cv::imshow("out", before_points_area);
			cv::waitKey(1);
			//*/
			cv::Mat M = cv::getPerspectiveTransform(before_points, after_points);
			cv::Mat dst(location_height + point_buffer * 2, location_width + point_buffer * 2, CV_8UC1);
			cv::warpPerspective(larger_area, dst, M, dst.size());

			cv::Mat dst_grayImage(dst.size(), CV_8UC1);
			cv::Mat dst_binImage(dst.size(), CV_8UC1);
			cv::cvtColor(dst, dst_grayImage, CV_RGB2GRAY);
			cv::threshold(dst_grayImage, dst_binImage, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
			/*
			cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
			cv::imshow("out", dst_binImage);
			cv::waitKey(1);
			//*/

			Point2f box_a_sorted[4];
			if (flag_changed == false) {

				Mat dst_copy(dst_binImage.size(), CV_8UC1);
				dst_binImage.copyTo(dst_copy);

				std::vector<std::vector<Point>> contours_a;
				std::vector<Vec4i>hierarchy_a;
				cv::findContours(dst_binImage, contours_a, hierarchy_a, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

				//find contours again
				for (k = 0; k < contours_a.size(); k++) {
					RotatedRect rect_a = cv::minAreaRect(contours_a[k]);
					Point2f box_a[4];
					float area_a;
					rect_a.points(box_a);
					area_a = rect_a.size.area();
					int idx_a = hierarchy_a[k][2];
					float location_area_size = location_width * location_height;
					if (idx_a != -1
						&& area_a > location_area_size * outer_lower
						&& area_a < location_area_size * outer_upper) {
						vector<vector<Point>> contour(1);
						contour[0] = contours_a[k];
						//cv::drawContours(frame, contour, 0, (0, 255, 255), 2);
						//calc left-top
						int l;
						int box_left_top_idx;
						for (l = 0; l < 4; l++) {
							if (box_a[l].x < rect_a.center.x && box_a[l].y < rect_a.center.y) {
								box_left_top_idx = l;
								break;
							}
						}
						for (l = 0; l < 4; l++) {
							box_a_sorted[l] = box_a[(l + box_left_top_idx) % 4];
						}
					}
					else {
						int l;
						for (l = 0; l < 4; l++) {
							box_a_sorted[l] = Point(0, 0);
						}
					}
				}
			}
			else {
				for (k = 0; k < 4; k++) {
					box_a_sorted[k] = after_points[k];
				}
			}
			if (box_a_sorted[1].x == 0) {
				continue;
			}

			//read each point
			float area_width = max2f(box_a_sorted, 4, 0).x - min2f(box_a_sorted, 4, 0).x;
			float area_height = max2f(box_a_sorted, 4, 0).y - min2f(box_a_sorted, 4, 0).y;
			float center_x = mean2f(box_a_sorted, 4, 0).x;
			float location_ratio = float(area_width + area_height) / (location_width + location_height + line_width * 2);
			float space_x_mod = space_x * location_ratio;
			float to_center_dst = (location_width / 2.0 - (x_buffer + rect_len / 2.0 + space_x * 2))*location_ratio;

			bool dot_point[location_dot_num];

			for (k = 0; k < location_dot_num; k++) {
				dot_point[k] = 0;
				int x = max(0, int(center_x - to_center_dst + space_x_mod * (k - 2) - dot_read_area));
				int y = max(0, int(box_a_sorted[0].y + area_height / 2 - dot_read_area));
				int width = 2 * dot_read_area;
				int height = 2 * dot_read_area;
				/*
				if (x + width < dst_binImage.cols) {
				x = min(x, dst_binImage.cols);
				width = dst_binImage.cols - x;
				}
				if (y + height < dst_binImage.rows) {
				y = min(y, dst_binImage.rows);
				height = dst_binImage.rows - y;
				}
				*/
				cv::Rect dst_rect(x,
					y,
					width,
					height);
				cv::Mat dst_area(dst_binImage, dst_rect);
				//miru(dst_area);
				double ha = cv::mean(dst_area).val[0];
				if (cv::mean(dst_area).val[0] < double(dot_read_thre)) {
					dot_point[k] = 1;
				}

			}
			if (dot_point[0] == 1 && dot_point[location_dot_num - 1] == 0) {
				vector<Point2f>  box_a_after(4);

				//calc dot id
				int dot_id = 0;
				int m;
				for (m = 1; m < location_dot_num - 1; m++) {
					if (dot_point[m] == 1) {
						dot_id += int(pow(2, m - 1));
					}
				}
				//homography_inv
				vector<Point2f> after_points(4);
				for (m = 0; m < 4; m++) {
					after_points[m] = box[(m + id) % 4];
				}
				vector<Point2f>before_points{
					Point2f(0 + point_buffer, 0 + point_buffer),
					Point2f(location_width + point_buffer, 0 + point_buffer),
					Point2f(location_width + point_buffer, location_height + point_buffer),
					Point2f(0 + point_buffer, location_height + point_buffer)
				};
				cv::Mat M_inv(3, 3, CV_64FC1);
				M_inv = getPerspectiveTransform(before_points, after_points);

				for (m = 0; m < 4; m++) {
					cv::Mat box_3d = (cv::Mat_<double>(3, 1) << box_a_sorted[m].x, box_a_sorted[m].y, 1);
					//Mat box_3d(3, 1, CV_64FC1, { before_points[m].x, before_points[m].y, 1 });
					cv::Mat box_a_after_each(3, 1, CV_64FC1);
					box_a_after_each = M_inv * box_3d;
					//cv::warpPerspective(box_3d[m], box_a_after_each, M_inv, box_a_after_each.size());
					float x = float(box_a_after_each.at<double>(0, 0));
					float y = float(box_a_after_each.at<double>(1, 0));
					box_a_after[m].x = x;
					box_a_after[m].y = y;
					if (kakunin)
						cv::circle(frame, Point(int(box_a_after_each.at<double>(0, 0)), int(box_a_after_each.at<double>(1, 0))), 5, Scalar(100, 100, 100), 5);
				}
				mtx_location.lock();
				location_xy[dot_id].push_back(box_a_after);
				mtx_location.unlock();
				//write each id in this point
				if (kakunin) {
					cv::putText(frame, to_string(dot_id),
						Point(int(mean2f(&box_a_after[0], 4, 0).x), int(mean2f(&box_a_after[0], 4, 0).y)),
						CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 255, 0), 5);
				}
			}
		}
		//Timer(tyoku_timer, "tyoku 1loop");

	}
	//for (mj = 0; mj < marker_contours.size(); mj++) {
	//	float expand_ratio = EXPAND_RATIO;
	//	const int location_dot_num = LOCATION_DOT_NUM;
	//	int x_buffer = int(X_BUFFER * expand_ratio);//from rectangle's edge
	//	int y_buffer = int(Y_BUFFER * expand_ratio); //from rectangle's edge
	//	int space_x = int(SPACE_X * expand_ratio);
	//	int rect_len = int(RECT_LEN * expand_ratio);
	//	int location_width = x_buffer * 2 + space_x * (location_dot_num - 1) + rect_len;
	//	int location_height = int(LOCATION_HEIGHT * expand_ratio);
	//	int line_width = int(LINE_WIDTH * expand_ratio);

	//	//other parameters
	//	int point_buffer = int(POINT_BUFFER * expand_ratio); //used when searching in larger area and extract larger area of location point area
	//	float outer_lower = OUTER_LOWER;
	//	float outer_upper = OUTER_UPPER;
	//	int dot_read_thre = DOT_READ_THRE;
	//	int dot_read_area = int(DOT_READ_AREA * expand_ratio);
	//	//Timer(prev_timer, "最初");
	//	vector<Point> contour = marker_contours[mj];
	//	//Timer(prev_timer, "contour取ってくる");
	//	cv::RotatedRect rect = minAreaRect(contour);
	//	//Timer(prev_timer, "minAreaRect特定");
	//	float area = rect.size.area();
	//	vector<Point2f> rect_point(4);
	//	vector<Point> rect_int_point(4);
	//	rect.points(&rect_point[0]);
	//	int mi;
	//	for (mi = 0; mi < 4; mi++) {
	//		rect_int_point[mi] = Point((int)rect_point[mi].x, (int)rect_point[mi].y);
	//	}
	//	//Timer(prev_timer, "その他");
	//	//if(100 < area && area < 3000){

	//	/*
	//	//cout << area << "\n";
	//	cv::drawContours(frame, contours, i, Scalar(255, 0, 255), 1.0);
	//	//cv::drawContours(frame, vector<vector<Point>>{rect_int_point}, 0, Scalar(0, 0, 255), 5.0);
	//	cv::putText(frame, to_string(int(area)), contours[i][0], CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 128, 0), 3);
	//	//*
	//	show_img = cv::Mat(cv::Size(frame.cols / 5, frame.rows / 5), CV_8UC3);
	//	cv::resize(frame, show_img, show_img.size());
	//	cv::imshow("show", show_img);
	//	cv::waitKey(1);
	//	//*/
	//	float wh_ratio = rect.size.width / rect.size.height;
	//	if (wh_ratio < 1) {
	//		wh_ratio = 1 / wh_ratio;
	//	}
	//	if (wh_ratio > 2.5) {
	//		continue;
	//	}

	//	if (kakunin) {
	//		cv::drawContours(frame, contours, i, Scalar(255, 0, 255), 1.0);
	//		cv::drawContours(frame, vector<vector<Point>>{rect_int_point}, 0, Scalar(0, 255, 255), 5.0);
	//		cv::putText(frame, to_string(int(area)), contours[i][0], CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 128, 255), 2.0);
	//	}
	//	//extract larger area
	//	//Timer(prev_timer, "rect");
	//	Point2f box[4];
	//	rect.points(box);
	//	Point2f max_point;
	//	Point2f min_point;
	//	max_point = max2f(box, 4, 0);
	//	min_point = min2f(box, 4, 0);
	//	float point_buffer_for_larger = point_buffer / expand_ratio * pow(outer_size / 220, 0.5);
	//	int min_y = max(0, int(min_point.y - point_buffer_for_larger));
	//	int max_y = min(frame_original.rows, int(max_point.y + point_buffer_for_larger));
	//	int min_x = max(0, int(min_point.x - point_buffer_for_larger));
	//	int max_x = min(frame_original.cols, int(max_point.x + point_buffer_for_larger));
	//	cv::Rect larger_area_rect(min_x, min_y, max_x - min_x, max_y - min_y);
	//	cv::Mat larger_area(frame_original, larger_area_rect);
	//	cv::Mat larger_area_grayImage(larger_area.size(), CV_8UC1);
	//	cv::cvtColor(larger_area, larger_area_grayImage, CV_RGB2GRAY);
	//	cv::Mat bin_larger_area(larger_area.size(), CV_8UC1);
	//	cv::threshold(larger_area_grayImage, bin_larger_area, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//	/*
	//	namedWindow("larger_area", CV_WINDOW_AUTOSIZE);
	//	imshow("larger_area", frame_original);
	//	waitKey(1);
	//	//*/


	//	//Timer(prev_timer, "larger_area");
	//	//find contours inlarger area
	//	bool flag_changed = false;

	//	std::vector<std::vector<Point>> contours_in_larger;
	//	std::vector<Vec4i>hierarchy_in_larger;
	//	cv::findContours(bin_larger_area, contours_in_larger, hierarchy_in_larger, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	//	int j;
	//	for (j = 0; j != contours_in_larger.size(); j++) {
	//		cv::RotatedRect rect_in_larger;
	//		vector<Point> contour_in_larger;
	//		contour_in_larger = contours_in_larger[j];
	//		rect_in_larger = minAreaRect(contour_in_larger);
	//		float area_in_larger = rect_in_larger.size.area();
	//		Point2f box_in_larger[4];
	//		rect_in_larger.points(box_in_larger);
	//		if (outer_size * outer_lower < area_in_larger && area_in_larger < outer_size * outer_upper) {
	//			int idx = hierarchy_in_larger[j][2];
	//			if (idx != -1) {
	//				int k;
	//				for (k = 0; k < 4; k++) {
	//					box[k] = Point2f(min_point.x - point_buffer_for_larger + box_in_larger[k].x,
	//						min_point.y - point_buffer_for_larger + box_in_larger[k].y);
	//				}
	//				flag_changed = true;
	//			}
	//		}
	//	}


	//	//Timer(prev_timer, "find contours larger area");
	//	//find candidate of left-top
	//	vector<int> candidate;
	//	for (j = 0; j < 4; j++) {
	//		if (norm(box[j] - box[(j + 1) % 4]) > norm(box[(j + 1) % 4] - box[(j + 2) % 4])) {
	//			candidate.push_back(j);
	//		}
	//	}

	//	int find_candidate_of_lt;
	//	//Timer(prev_timer, "lefttop");
	//	//read each location's id
	//	for (j = 0; j < candidate.size(); j++) {
	//		/*
	//		cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
	//		cv::imshow("out", frame_original);
	//		cv::waitKey(1);
	//		*/

	//		int id = candidate[j];
	//		vector<Point2f> before_points;
	//		int k;
	//		for (k = 0; k < 4; k++) {
	//			before_points.push_back(box[(k + id) % 4]);
	//		}
	//		RotatedRect before_points_rotatedrect;
	//		before_points_rotatedrect = minAreaRect(before_points);
	//		Rect before_points_rect;
	//		before_points_rect = before_points_rotatedrect.boundingRect();
	//		if (before_points_rect.x < 0) {
	//			before_points_rect.x = 0;
	//		}
	//		if (before_points_rect.y < 0) {
	//			before_points_rect.y = 0;
	//		}
	//		if (before_points_rect.width + before_points_rect.x > frame_original.cols) {
	//			before_points_rect.width = frame_original.cols - before_points_rect.x;
	//		}
	//		if (before_points_rect.height + before_points_rect.y > frame_original.rows) {
	//			before_points_rect.height = frame_original.rows - before_points_rect.y;
	//		}

	//		//Mat before_points_area(frame_original, before_points_rect);
	//		Mat before_points_area(larger_area, Rect(before_points_rect.tl() - larger_area_rect.tl(),
	//			Size(before_points_rect.width, before_points_rect.height)));
	//		vector<Point2f> after_points{
	//			Point2f(0 + point_buffer, 0 + point_buffer),
	//			Point2f(location_width + point_buffer, 0 + point_buffer),
	//			Point2f(location_width + point_buffer, location_height + point_buffer),
	//			Point2f(0 + point_buffer, location_height + point_buffer)
	//		};
	//		/*
	//		cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
	//		cv::imshow("out", before_points_area);
	//		cv::waitKey(1);
	//		//*/
	//		cv::Mat M = cv::getPerspectiveTransform(before_points, after_points);
	//		cv::Mat dst(location_height + point_buffer * 2, location_width + point_buffer * 2, frame_original.type());
	//		cv::warpPerspective(frame_original, dst, M, dst.size());

	//		cv::Mat dst_grayImage(dst.size(), CV_8UC1);
	//		cv::Mat dst_binImage(dst.size(), CV_8UC1);
	//		cv::cvtColor(dst, dst_grayImage, CV_RGB2GRAY);
	//		cv::threshold(dst_grayImage, dst_binImage, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//		/*
	//		cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
	//		cv::imshow("out", dst_binImage);
	//		cv::waitKey(1);
	//		//*/

	//		Point2f box_a_sorted[4];
	//		if (flag_changed == false) {

	//			Mat dst_copy(dst_binImage.size(), CV_8UC1);
	//			dst_binImage.copyTo(dst_copy);

	//			std::vector<std::vector<Point>> contours_a;
	//			std::vector<Vec4i>hierarchy_a;
	//			cv::findContours(dst_binImage, contours_a, hierarchy_a, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	//			//find contours again
	//			for (k = 0; k < contours_a.size(); k++) {
	//				RotatedRect rect_a = cv::minAreaRect(contours_a[k]);
	//				Point2f box_a[4];
	//				float area_a;
	//				rect_a.points(box_a);
	//				area_a = rect_a.size.area();
	//				int idx_a = hierarchy_a[k][2];
	//				float location_area_size = location_width * location_height;
	//				if (idx_a != -1
	//					&& area_a > location_area_size * outer_lower
	//					&& area_a < location_area_size * outer_upper) {
	//					vector<vector<Point>> contour(1);
	//					contour[0] = contours_a[k];
	//					//cv::drawContours(frame, contour, 0, (0, 255, 255), 2);
	//					//calc left-top
	//					int l;
	//					int box_left_top_idx;
	//					for (l = 0; l < 4; l++) {
	//						if (box_a[l].x < rect_a.center.x && box_a[l].y < rect_a.center.y) {
	//							box_left_top_idx = l;
	//							break;
	//						}
	//					}
	//					for (l = 0; l < 4; l++) {
	//						box_a_sorted[l] = box_a[(l + box_left_top_idx) % 4];
	//					}
	//				}
	//				else {
	//					int l;
	//					for (l = 0; l < 4; l++) {
	//						box_a_sorted[l] = Point(0, 0);
	//					}
	//				}
	//			}
	//		}
	//		else {
	//			for (k = 0; k < 4; k++) {
	//				box_a_sorted[k] = after_points[k];
	//			}
	//		}
	//		if (box_a_sorted[1].x == 0) {
	//			continue;
	//		}

	//		//read each point
	//		float area_width = max2f(box_a_sorted, 4, 0).x - min2f(box_a_sorted, 4, 0).x;
	//		float area_height = max2f(box_a_sorted, 4, 0).y - min2f(box_a_sorted, 4, 0).y;
	//		float center_x = mean2f(box_a_sorted, 4, 0).x;
	//		float location_ratio = float(area_width + area_height) / (location_width + location_height + line_width * 2);
	//		float space_x_mod = space_x * location_ratio;
	//		float to_center_dst = (location_width / 2.0 - (x_buffer + rect_len / 2.0 + space_x * 2))*location_ratio;

	//		bool dot_point[location_dot_num];

	//		for (k = 0; k < location_dot_num; k++) {
	//			dot_point[k] = 0;
	//			int x = max(0, int(center_x - to_center_dst + space_x_mod * (k - 2) - dot_read_area));
	//			int y = max(0, int(box_a_sorted[0].y + area_height / 2 - dot_read_area));
	//			int width = 2 * dot_read_area;
	//			int height = 2 * dot_read_area;
	//			/*
	//			if (x + width < dst_binImage.cols) {
	//			x = min(x, dst_binImage.cols);
	//			width = dst_binImage.cols - x;
	//			}
	//			if (y + height < dst_binImage.rows) {
	//			y = min(y, dst_binImage.rows);
	//			height = dst_binImage.rows - y;
	//			}
	//			*/
	//			cv::Rect dst_rect(x,
	//				y,
	//				width,
	//				height);
	//			cv::Mat dst_area(dst_binImage, dst_rect);
	//			//miru(dst_area);
	//			double ha = cv::mean(dst_area).val[0];
	//			if (cv::mean(dst_area).val[0] < double(dot_read_thre)) {
	//				dot_point[k] = 1;
	//			}

	//		}
	//		if (dot_point[0] == 1 && dot_point[location_dot_num - 1] == 0) {
	//			vector<Point2f>  box_a_after(4);

	//			//calc dot id
	//			int dot_id = 0;
	//			int m;
	//			for (m = 1; m < location_dot_num - 1; m++) {
	//				if (dot_point[m] == 1) {
	//					dot_id += int(pow(2, m - 1));
	//				}
	//			}
	//			//homography_inv
	//			vector<Point2f> after_points(4);
	//			for (m = 0; m < 4; m++) {
	//				after_points[m] = box[(m + id) % 4];
	//			}
	//			vector<Point2f>before_points{
	//				Point2f(0 + point_buffer, 0 + point_buffer),
	//				Point2f(location_width + point_buffer, 0 + point_buffer),
	//				Point2f(location_width + point_buffer, location_height + point_buffer),
	//				Point2f(0 + point_buffer, location_height + point_buffer)
	//			};
	//			cv::Mat M_inv(3, 3, CV_64FC1);
	//			M_inv = getPerspectiveTransform(before_points, after_points);

	//			for (m = 0; m < 4; m++) {
	//				cv::Mat box_3d = (cv::Mat_<double>(3, 1) << box_a_sorted[m].x, box_a_sorted[m].y, 1);
	//				//Mat box_3d(3, 1, CV_64FC1, { before_points[m].x, before_points[m].y, 1 });
	//				cv::Mat box_a_after_each(3, 1, CV_64FC1);
	//				box_a_after_each = M_inv * box_3d;
	//				//cv::warpPerspective(box_3d[m], box_a_after_each, M_inv, box_a_after_each.size());
	//				float x = float(box_a_after_each.at<double>(0, 0));
	//				float y = float(box_a_after_each.at<double>(1, 0));
	//				box_a_after[m].x = x;
	//				box_a_after[m].y = y;
	//				if (kakunin)
	//					cv::circle(frame, Point(int(box_a_after_each.at<double>(0, 0)), int(box_a_after_each.at<double>(1, 0))), 5, Scalar(100, 100, 100), 5);
	//			}
	//			mtx_location.lock();
	//			location_xy[dot_id].push_back(box_a_after);
	//			mtx_location.unlock();
	//			//printf("aaa");
	//			//write each id in this point
	//			if (kakunin) {
	//				cv::putText(frame, to_string(dot_id),
	//					Point(int(mean2f(&box_a_after[0], 4, 0).x), int(mean2f(&box_a_after[0], 4, 0).y)),
	//					CV_FONT_HERSHEY_PLAIN, 5.0, Scalar(0, 255, 0), 5);
	//			}
	//		}
	//	}
	//	//Timer(prev_timer, "read location id");
	//}
	}

	postits->location_points = location_xy;

	float expand_ratio = EXPAND_RATIO;

	//postit parameters
	int postit_width = int(POSTIT_WIDTH * expand_ratio);
	int postit_height = int(POSTIT_HEIGHT * expand_ratio);

	//location parameters for outer
	const int location_dot_num = LOCATION_DOT_NUM;
	int horizon_x_buffer = int(HORIZON_X_BUFFER * expand_ratio); //from postit's edge
	int horizon_y_buffer = int(HORIZON_Y_BUFFER * expand_ratio);//from postit's edge

	//common parameters for inner
	int space_x = int(SPACE_X * expand_ratio);
	int rect_len = int(RECT_LEN * expand_ratio);
	int x_buffer = int(X_BUFFER * expand_ratio);//from rectangle's edge
	int y_buffer = int(Y_BUFFER * expand_ratio); //from rectangle's edge

	int bit_num = BIT_NUM;
	int bit_width = x_buffer * 2 + space_x * (bit_num - 1) + rect_len;
	int bit_height = int(BIT_HEIGHT * expand_ratio);

	//location parameters for outer

	int location_width = x_buffer * 2 + space_x * (location_dot_num - 1) + rect_len;
	int location_height = int(LOCATION_HEIGHT * expand_ratio);
	int horizon_space = postit_width / 2 - horizon_x_buffer - location_width / 2;


	//common parameters for outer
	int line_width = int(LINE_WIDTH * expand_ratio);
	int rect_rect_space_horizon = (horizon_space - location_width - bit_width * 2) / 3;
	int rect_rect_space_vertical = (postit_height / 2 - horizon_y_buffer - location_height - location_width / 2 - bit_width * 2) / 3;


	//other parameters
	int error_thresh = ERROR_THRESH;
	int dot_read_thre = DOT_READ_THRE;
	int dot_read_area = int(DOT_READ_AREA * expand_ratio);
	int point_buffer = int(POINT_BUFFER * expand_ratio); //used when searching in larger area and extract larger area of location point area
	int larger_buffer = int(LARGER_BUFFER * expand_ratio); //used for setting larger area in whole postit area for analyzing
	int search_buffer = int(SEARCH_BUFFER * expand_ratio); //area of searching information bit rectangle
	float outer_lower = OUTER_LOWER;
	float outer_upper = OUTER_UPPER;
	bool find_grand_child = FIND_GRAND_CHILD;

	/*
	show_img = cv::Mat(cv::Size(frame.cols / 5, frame.rows / 5), CV_8UC3);
	cv::resize(frame, show_img, show_img.size());
	cv::imshow("show", show_img);
	cv::waitKey(1);
	//*/
	cv::findContours(binImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	zikken_output << "探索位置マーカ数=" << marker_contours.size() << endl;
	Timer(prev_timer, "recognition 位置マーカ探索:");

	cout << omp_get_thread_num() << endl;
	//make from to vector
	Point2f vec_from_to[8][8];
	float correction_width = 5 * expand_ratio;
	float correction_height = 5 * expand_ratio;
	Point2f right_basic_vec(float(postit_width / 2 - horizon_x_buffer - location_width / 2) / (location_width + correction_width), 0.0);
	Point2f under_basic_vec(0.0, float(postit_height / 2 - horizon_y_buffer - location_height / 2) / (location_height + correction_height));
	vec_from_to[0][1] = right_basic_vec;
	vec_from_to[0][2] = right_basic_vec * 2;
	vec_from_to[0][3] = under_basic_vec * 2;
	vec_from_to[0][4] = right_basic_vec + under_basic_vec * 2;
	vec_from_to[0][5] = right_basic_vec * 2 + under_basic_vec * 2;
	vec_from_to[0][6] = under_basic_vec;
	vec_from_to[0][7] = right_basic_vec * 2 + under_basic_vec;
	vec_from_to[1][2] = right_basic_vec;
	vec_from_to[1][3] = -right_basic_vec + 2 * under_basic_vec;
	vec_from_to[1][4] = under_basic_vec * 2;
	vec_from_to[1][5] = right_basic_vec + under_basic_vec * 2;
	vec_from_to[1][6] = -right_basic_vec + under_basic_vec;
	vec_from_to[1][7] = right_basic_vec + under_basic_vec;
	vec_from_to[2][3] = -right_basic_vec * 2 + under_basic_vec * 2;
	vec_from_to[2][4] = -right_basic_vec + under_basic_vec * 2;
	vec_from_to[2][5] = under_basic_vec * 2;
	vec_from_to[2][6] = -right_basic_vec * 2 + under_basic_vec;
	vec_from_to[2][7] = under_basic_vec;
	vec_from_to[3][4] = right_basic_vec;
	vec_from_to[3][5] = right_basic_vec * 2;
	vec_from_to[3][6] = -under_basic_vec;
	vec_from_to[3][7] = right_basic_vec * 2 - under_basic_vec;
	vec_from_to[4][5] = right_basic_vec;
	vec_from_to[4][6] = -right_basic_vec - under_basic_vec;
	vec_from_to[4][7] = right_basic_vec - under_basic_vec;
	vec_from_to[5][6] = -right_basic_vec * 2 - under_basic_vec;
	vec_from_to[5][7] = -under_basic_vec;
	vec_from_to[6][7] = right_basic_vec * 2;

	//other from to vectors
	int j;
	for (i = 0; i < 8; i++) {
		for (j = 0; j < 8; j++) {
			//print vec_from_to[i][j]
			if (i > j) {
				vec_from_to[i][j] = -vec_from_to[j][i];
			}
		}
	}
	//search each point
	//for left_top
	vector<vector<Point2f>> location_points_all;
	int m, n;
	for (m = 0; m < location_xy.size(); m++) {
		//location_xy[0] includes boxes for id "0"
		for (n = 0; n < location_xy[m].size(); n++) {
			if (location_xy[m][n][0].x != -1) {
				Point2f	right_vector = location_xy[m][n][1] - location_xy[m][n][0];
				Point2f  below_vector = location_xy[m][n][3] - location_xy[m][n][0];
				//swap when m is 6 or 7(side place)
				if (m == 6 || m == 7) {
					Point2f below_vector_copy = below_vector;
					below_vector = Point2f(right_vector.y, -right_vector.x);
					right_vector = Point2f(below_vector_copy.y, below_vector_copy.x);
				}
				Point2f sum_vector = right_vector + below_vector;
				vector<Point2f> location_points(8);
				int points_count = 0;
				vector <vector<int>> points_to_delete;
				for (i = 0; i < 8; i++) {
					if (i != m) {
						Point2f dst_point = mean2f(&location_xy[m][n][0], 4, 0) + vec_from_to[m][i].x * right_vector + vec_from_to[m][i].y * below_vector;
						for (j = 0; j < location_xy[i].size(); j++) {
							if (location_xy[i][j][0].x != -1 &&
								norm(mean2f(&location_xy[i][j][0], 4, 0) - dst_point)  < error_thresh * (pow(double(outer_size) / 220, 0.5))) {
								location_points[i] = mean2f(&location_xy[i][j][0], 4, 0);
								vector<int> point_delete{ i,j };
								points_to_delete.push_back(point_delete);
								points_count += 1;
								break;
							}
						}
					}
				}
				postits->recognized_location_rectangle = points_count + 1;
				if (points_count > 2) {
					//add self
					location_points[m] = mean2f(&location_xy[m][n][0], 4, 0);
					vector < int> point_delete{ m,n };
					points_to_delete.push_back(point_delete);

					//delete already used box
					for (i = 0; i < points_to_delete.size(); i++) {
						vector<int> point_delete = points_to_delete[i];
						location_xy[point_delete[0]][point_delete[1]][0].x = -1;
					}
					//append to all
					//amatihadousuruno余りはどうすｒ
					location_points_all.push_back(location_points);
				}
			}
		}
	}

	//extract postit area
	//homography
	vector<Point2f> dst_points_original
	{ Point2f(horizon_x_buffer + location_width / 2, horizon_y_buffer + location_height / 2),
		Point2f(postit_width / 2, horizon_y_buffer + location_height / 2),
		Point2f(postit_width - horizon_x_buffer - location_width / 2 ,horizon_y_buffer + location_height / 2),
		Point2f(horizon_x_buffer + location_width / 2, postit_height - horizon_y_buffer - location_height / 2),
		Point2f(postit_width / 2,postit_height - horizon_y_buffer - location_height / 2),
		Point2f(postit_width - horizon_x_buffer - location_width / 2,postit_height - horizon_y_buffer - location_height / 2),
		Point2f(horizon_x_buffer + location_height / 2, postit_height / 2),
		Point2f(postit_width - horizon_x_buffer - location_height / 2, postit_height / 2)
	};

	//extract postit area for analyzing
	//homography
	vector<Point2f> dst_points_original_larger(8);
	for (i = 0; i < dst_points_original.size(); i++) {
		dst_points_original_larger[i] = Point2f(larger_buffer, larger_buffer) + dst_points_original[i];
	}




	//points include 8 points(1 postit's each point)
	int count = 0;
	for (m = 0; m < location_points_all.size(); m++) {
		vector<Point2f> points = location_points_all[m];
		vector<Point2f> src_point;
		vector<Point2f> dst_point;
		vector<Point2f> dst_point_larger;
		int i;
		for (i = 0; i < points.size(); i++) {
			if (points[i].x != 0 && points[i].y != 0) {
				src_point.push_back(points[i]);
				dst_point.push_back(dst_points_original[i]);
				dst_point_larger.push_back(dst_points_original_larger[i]);
			}

		}
		cv::Mat M = cv::findHomography(src_point, dst_point_larger, CV_RANSAC, 3);
		Mat postit_result(postit_height + larger_buffer * 2, postit_width + larger_buffer * 2, CV_8UC3);
		//try except suru?
		cv::warpPerspective(frame_original, postit_result, M, postit_result.size());
		//imshow("out", postit_result);
		//imwrite("fusen" + to_string(m) + ".jpg", postit_result);
		//cv::waitKey(1);
		Mat postit_result_gray;
		cv::cvtColor(postit_result, postit_result_gray, CV_RGB2GRAY);
		Mat postit_result_bin;
		Mat postit_result_morphology;
		//try except ? 
		cv::threshold(postit_result_gray, postit_result_bin, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);

		//cv::morphologyEx(postit_result_bin, postit_result_morphology, MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 1);
		//cv::dilate(postit_result_bin, postit_result_morphology, cv::Mat(), cv::Point(-1, -1), 1);

		/*
		cv::namedWindow("husen", CV_WINDOW_AUTOSIZE);
		cv::imshow("husen", postit_result);
		waitKey(1);
		cv::namedWindow("husen_bin", CV_WINDOW_AUTOSIZE);
		cv::imshow("husen_bin", postit_result_bin);
		waitKey(1);

		imwrite("husen_dame.jpg", postit_result_morphology);
		//*/

		Mat M_inv(3, 3, CV_64FC1);
		M_inv = cv::findHomography(dst_point, src_point, CV_RANSAC, 3);
		vector<Mat> postit_area;
		//box_3d = (cv::Mat_<double>(3, 1) << box_a_sorted[m].x, box_a_sorted[m].y, 1);
		vector<vector<double>> data{ { 0, 0, 1 } ,{ double(postit_width), 0, 1 },
		{ double(postit_width), double(postit_height), 1 },{ 0,double(postit_height),1 } };
		vector<Mat> postit_area_before{ Mat(3,1,CV_64FC1,&data[0][0]), Mat(3,1,CV_64FC1,&data[1][0]), Mat(3,1,CV_64FC1,&data[2][0]),Mat(3,1,CV_64FC1,&data[3][0]) };
		vector<Point2f> postit_points_each;

		for (i = 0; i < postit_area_before.size(); i++) {
			Mat postit_area_before_each = postit_area_before[i];
			//print postit_area_before_each
			Mat postit_area_after(3, 1, CV_64FC1);
			postit_area_after = M_inv * postit_area_before_each;
			//miru(postit_area_after);
			postit_points_each.push_back(Point2f(float(postit_area_after.at<double>(0, 0)), float(postit_area_after.at<double>(1, 0))));
		}
		postits->postit_image_analyzing.push_back(postit_result_bin);
		postits->postit_points.push_back(postit_points_each);
		count += 1;
		//miru(postit_area_before[1]);
	}
	//show frame and delete(for save memory)(information for read is drawn)
	show_img = cv::Mat(cv::Size(frame.cols / 5, frame.rows / 5), CV_8UC3);
	cv::resize(frame, show_img, show_img.size());

	if (kakunin) {
		cv::imshow("show", show_img);
		cv::waitKey(1);
	}
	zikken_output << "付箋発見数=" << count << endl;

	Timer(prev_timer, "recognition 付箋領域抽出:");
	return;
}

int read_bit(Mat postit, int first_x, int first_y, bool horizon, int outer_size) {
	float expand_ratio = EXPAND_RATIO;
	//postit parameters
	int postit_width = int(POSTIT_WIDTH * expand_ratio);
	int postit_height = int(POSTIT_HEIGHT * expand_ratio);
	//common parameters for inner
	int space_x = int(SPACE_X * expand_ratio);
	int rect_len = int(RECT_LEN * expand_ratio);
	int x_buffer = int(X_BUFFER * expand_ratio);//from rectangle's edge
	int y_buffer = int(Y_BUFFER * expand_ratio); //from rectangle's edge

	int bit_num = BIT_NUM;
	int bit_width = x_buffer * 2 + space_x * (bit_num - 1) + rect_len;
	int bit_height = int(BIT_HEIGHT * expand_ratio);

	//location parameters for outer
	const int location_dot_num = LOCATION_DOT_NUM;
	int horizon_x_buffer = int(HORIZON_X_BUFFER * expand_ratio); //from postit's edge
	int horizon_y_buffer = int(HORIZON_Y_BUFFER * expand_ratio);//from postit's edge
	int location_width = x_buffer * 2 + space_x * (location_dot_num - 1) + rect_len;
	int location_height = int(LOCATION_HEIGHT * expand_ratio);
	int horizon_space = postit_width / 2 - horizon_x_buffer - location_width / 2;

	//common parameters for outer
	int line_width = int(LINE_WIDTH * expand_ratio);
	int rect_rect_space_horizon = (horizon_space - location_width - bit_width * 2) / 3;
	int rect_rect_space_vertical = (postit_height / 2 - horizon_y_buffer - location_height - location_width / 2 - bit_width * 2) / 3;

	//other parameters
	int error_thresh = ERROR_THRESH;
	int dot_read_thre = DOT_READ_THRE;
	int dot_read_area = int(DOT_READ_AREA * expand_ratio);
	int point_buffer = int(POINT_BUFFER * expand_ratio); //used when searching in larger area and extract larger area of location point area
	int larger_buffer = int(LARGER_BUFFER * expand_ratio); //used for setting larger area in whole postit area for analyzing
	int search_buffer = int(SEARCH_BUFFER * expand_ratio); //area of searching information bit rectangle
	float outer_lower = OUTER_LOWER;
	float outer_upper = OUTER_UPPER;
	bool find_grand_child = FIND_GRAND_CHILD;

	if (kakunin) {
		cv::namedWindow("search_area", CV_WINDOW_AUTOSIZE);
	}
	//return val
	int sum = 0;

	//change parameter
	dot_read_thre = 100;
	float space_x_mod_ratio = 0.90;


	//parameters for this purpose
	float min_ratio = 0.8;
	float max_ratio = 1.7;



	//add buffer
	first_x += larger_buffer;
	first_y += larger_buffer;

	if (horizon == true) {
		//extract search area
		int min_y = max(0, first_y - search_buffer);
		int max_y = min(postit.rows, first_y + bit_height + search_buffer);
		int min_x = max(0, first_x - search_buffer);
		int max_x = min(postit.cols, first_x + bit_width + search_buffer);
		Mat search_area(postit, Rect(min_x, min_y, max_x - min_x, max_y - min_y));
		//find contours
		Mat search_area_for_contours;
		search_area.copyTo(search_area_for_contours);
		std::vector<std::vector<Point>> contours;
		std::vector<Vec4i>hierarchy;
		cv::findContours(search_area_for_contours, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		int box_left_top_idx = -1;
		/*
		cv::imshow("search_area", search_area);
		cv::waitKey(1);
		//*/
		int count;
		RotatedRect rect;
		vector<Point2f> box(4);
		float area;
		for (count = 0; count < contours.size(); count++) {
			//minAreaRect
			rect = cv::minAreaRect(contours[count]);
			rect.points(&box[0]);
			area = rect.size.area();
			int i;
			if (area > outer_size * min_ratio && area < outer_size * max_ratio) {
				//cv::drawContours(search_area, contours, count, Scalar(0, 255, 0), 5);
				//calc left - top
				int i;
				for (i = 0; i < 4; i++) {
					if (box[i].x < rect.center.x && box[i].y < rect.center.y) {
						box_left_top_idx = i;
						break;
					}
				}
				break;
			}
		}
		if (box_left_top_idx == -1) {
			return 0;
		}
		//make box_a_sorted
		vector<Point2f> box_a_sorted(4);
		int i;
		for (i = 0; i < 4; i++) {
			box_a_sorted[i] = box[(i + box_left_top_idx) % 4];
		}
		//set area parameters
		float area_width = norm(box_a_sorted[1] - box_a_sorted[0]);
		float area_height = norm(box_a_sorted[2] - box_a_sorted[1]);
		double area_theta = M_PI / 2;

		if ((box_a_sorted[1] - box_a_sorted[0]).x != 0) {
			double area_tan = double((box_a_sorted[1] - box_a_sorted[0]).y) / (box_a_sorted[1] - box_a_sorted[0]).x;
			area_theta = atan(area_tan);
		}
		Point2f center = mean2f(&box_a_sorted[0], 4, 0);
		float center_x = center.x;
		float center_y = center.y;
		float bit_ratio = float(area_width + area_height) / (bit_width + bit_height + line_width * 2);
		float space_x_mod = space_x * bit_ratio * space_x_mod_ratio;
		float space_x_mod_x = space_x_mod * cos(area_theta);
		float space_x_mod_y = space_x_mod * sin(area_theta);
		float to_center_dst = (bit_width / 2 - (x_buffer + rect_len / 2 + space_x * 4)) * bit_ratio;
		float to_center_dst_x = to_center_dst * cos(area_theta);
		float to_center_dst_y = to_center_dst * sin(area_theta);

		vector<int> dot_point(bit_num, 0);
		//miru(search_area);
		for (i = 0; i < bit_num; i++) {
			int x = int(center_x - to_center_dst_x + space_x_mod_x * (i - 4) - dot_read_area);
			int y = int(center_y - to_center_dst_y + space_x_mod_y * (i - 4) - dot_read_area);
			int w = dot_read_area * 2;
			int h = dot_read_area * 2;
			if (x < 0 || y< 0 || x + w > search_area.cols || y + h > search_area.rows) {
				continue;
			}

			Mat dst_area(search_area,
				Rect(x, y, w, h));
			if (dst_area.rows == 0)
				break;
			//miru(dst_area);
			if (mean(dst_area).val[0] < dot_read_thre) {
				dot_point[i] = 1;
			}
		}
		dot_point[0] = dot_point[bit_num - 1];
		for (i = 0; i < bit_num - 1; i++) {
			if (dot_point[i] == 1) {
				sum += int(pow(2, i));
			}
		}
		return sum;
	}
	else {
		//extract search area
		int min_y = max(0, first_y - search_buffer);
		int max_y = min(postit.rows, first_y + bit_width + search_buffer);
		int min_x = max(0, first_x - search_buffer);
		int max_x = min(postit.cols, first_x + bit_height + search_buffer);

		Mat search_area(postit, Rect(min_x, min_y, max_x - min_x, max_y - min_y));
		Mat search_area_for_contours;
		//imshow("search_area", search_area);
		search_area.copyTo(search_area_for_contours);
		vector<vector<Point>> contours;
		std::vector<Vec4i>hierarchy;
		cv::findContours(search_area_for_contours, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		int box_left_top_idx = -1;
		int count;
		RotatedRect rect;
		vector<Point2f> box(4);
		float area;
		for (count = 0; count < contours.size(); count++) {
			//minAreaRect
			rect = minAreaRect(contours[count]);
			rect.points(&box[0]);
			area = rect.size.area();
			if (area > outer_size * min_ratio && area < outer_size * max_ratio) {
				//print area
				//cv::drawContours(search_area, contours, count, Scalar(0, 255, 0), 5);
				int i;
				for (i = 0; i < 4; i++) {
					if (box[i].x < rect.center.x && box[i].y < rect.center.y) {
						box_left_top_idx = i;
						break;
					}
				}
				break;
			}
		}
		if (box_left_top_idx == -1) {
			return 100;
		}
		//make box_a_sorted
		vector<Point2f> box_a_sorted(4);
		int i;
		for (i = 0; i < 4; i++) {
			box_a_sorted[i] = box[(box_left_top_idx + i) % 4];
		}
		//set area parameters
		float area_width = norm(box_a_sorted[1] - box_a_sorted[0]);
		float area_height = norm(box_a_sorted[2] - box_a_sorted[1]);
		double area_theta = M_PI / 2;
		if ((box_a_sorted[2] - box_a_sorted[1]).x != 0) {
			double area_tan = double((box_a_sorted[2] - box_a_sorted[1]).y) / (box_a_sorted[2] - box_a_sorted[1]).x;
			area_theta = atan(area_tan);
			if (area_theta < 0) {
				area_theta += M_PI;
			}
		}
		float center_x = mean2f(&box_a_sorted[0], 4, 0).x;
		float center_y = mean2f(&box_a_sorted[0], 4, 0).y;
		float bit_ratio = float(area_width + area_height) / (bit_width + bit_height + line_width * 2);
		float space_x_mod = space_x * bit_ratio * space_x_mod_ratio;
		float space_x_mod_x = space_x_mod * cos(area_theta);
		float space_x_mod_y = space_x_mod * sin(area_theta);
		float to_center_dst = (bit_width / 2 - (x_buffer + rect_len / 2 + space_x * 4)) * bit_ratio;
		float to_center_dst_x = to_center_dst * cos(area_theta);
		float to_center_dst_y = to_center_dst * sin(area_theta);
		vector<int>dot_point(bit_num, 0);
		//miru(search_area);
		for (i = 0; i < bit_num; i++) {
			int x = int(center_x - to_center_dst_x + space_x_mod_x * (i - 4) - dot_read_area);
			int y = int(center_y - to_center_dst_y + space_x_mod_y * (i - 4) - dot_read_area);
			int w = dot_read_area * 2;
			int h = dot_read_area * 2;
			if (x < 0 || y< 0 || x + w > search_area.cols || y + h > search_area.rows) {
				continue;
			}

			Mat dst_area(search_area,
				Rect(x, y, w, h));
			if (dst_area.rows == 0) {
				break;
			}
			//miru(dst_area);
			if (mean(dst_area).val[0] < dot_read_thre) {
				dot_point[i] = 1;
			}
		}
		dot_point[0] = dot_point[bit_num - 1];
		for (i = 0; i < bit_num; i++) {
			if (dot_point[i] == 1) {
				sum += int(pow(2, i));
			}
		}
		return sum;
	}
	return 100;
}




vector<int>readDots(Mat postit, int outer_size) {
	float expand_ratio = EXPAND_RATIO;
	//postit parameters
	int postit_width = int(POSTIT_WIDTH * expand_ratio);
	int postit_height = int(POSTIT_HEIGHT * expand_ratio);

	//common parameters for inner
	int space_x = int(SPACE_X * expand_ratio);
	int rect_len = int(RECT_LEN * expand_ratio);
	int x_buffer = int(X_BUFFER * expand_ratio);//from rectangle's edge
	int y_buffer = int(Y_BUFFER * expand_ratio); //from rectangle's edge
	int bit_num = BIT_NUM;
	int bit_width = x_buffer * 2 + space_x * (bit_num - 1) + rect_len;
	int bit_height = int(BIT_HEIGHT * expand_ratio);

	//location parameters for outer
	const int location_dot_num = LOCATION_DOT_NUM;
	int horizon_x_buffer = int(HORIZON_X_BUFFER * expand_ratio); //from postit's edge
	int horizon_y_buffer = int(HORIZON_Y_BUFFER * expand_ratio);//from postit's edge
	int location_width = x_buffer * 2 + space_x * (location_dot_num - 1) + rect_len;
	int location_height = int(LOCATION_HEIGHT * expand_ratio);
	int horizon_space = postit_width / 2 - horizon_x_buffer - location_width / 2;
	//common parameters for outer
	int line_width = int(LINE_WIDTH * expand_ratio);
	int rect_rect_space_horizon = (horizon_space - location_width - bit_width * 2) / 3;
	int rect_rect_space_vertical = (postit_height / 2 - horizon_y_buffer - location_height - location_width / 2 - bit_width * 2) / 3;

	vector<int> bit_array(15, 0);
	outer_size = int(1800 * pow(double(expand_ratio), 2.0));
	//draw upper left and lower left
	int start_left = horizon_x_buffer + location_width + rect_rect_space_horizon;
	bit_array[0] = read_bit(postit, start_left, horizon_y_buffer, true, outer_size);
	bit_array[1] = read_bit(postit, start_left, postit_height - horizon_y_buffer - bit_height, true, outer_size);

	start_left += bit_width + rect_rect_space_horizon;
	bit_array[2] = read_bit(postit, start_left, horizon_y_buffer, true, outer_size);
	bit_array[3] = read_bit(postit, start_left, postit_height - horizon_y_buffer - bit_height, true, outer_size);

	//draw upper right and lower right
	start_left = postit_width / 2 + location_width / 2 + rect_rect_space_horizon;
	bit_array[4] = read_bit(postit, start_left, horizon_y_buffer, true, outer_size);
	bit_array[5] = read_bit(postit, start_left, postit_height - horizon_y_buffer - bit_height, true, outer_size);

	start_left += bit_width + rect_rect_space_horizon;
	bit_array[6] = read_bit(postit, start_left, horizon_y_buffer, true, outer_size);

	//draw left upper and right upper
	int start_upper = horizon_y_buffer + location_height + rect_rect_space_vertical;
	bit_array[7] = read_bit(postit, horizon_x_buffer, start_upper, false, outer_size);
	bit_array[8] = read_bit(postit, postit_width - bit_height - horizon_x_buffer, start_upper, false, outer_size);

	start_upper += bit_width + rect_rect_space_vertical;
	bit_array[9] = read_bit(postit, horizon_x_buffer, start_upper, false, outer_size);
	bit_array[10] = read_bit(postit, postit_width - bit_height - horizon_x_buffer, start_upper, false, outer_size);

	//draw left lower and right lower
	start_upper = postit_height / 2 + location_width / 2 + rect_rect_space_vertical;
	bit_array[11] = read_bit(postit, horizon_x_buffer, start_upper, false, outer_size);
	bit_array[12] = read_bit(postit, postit_width - bit_height - horizon_x_buffer, start_upper, false, outer_size);

	start_upper += bit_width + rect_rect_space_vertical;
	bit_array[13] = read_bit(postit, horizon_x_buffer, start_upper, false, outer_size);
	bit_array[14] = read_bit(postit, postit_width - bit_height - horizon_x_buffer, start_upper, false, outer_size);

	return bit_array;
}


Point2f min2f(Point2f * points, int len, int axis) {
	int i;
	float min_x = points[0].x;
	float min_y = points[0].y;

	if (axis == 0) {
		for (i = 0; i < len; i++) {
			if (points[i].x < min_x) {
				min_x = points[i].x;
			}
			if (points[i].y < min_y) {
				min_y = points[i].y;
			}
		}
	}

	return Point2f(min_x, min_y);
}
Point2f max2f(Point2f * points, int len, int axis) {
	int i;
	float max_x = points[0].x;
	float max_y = points[0].y;

	if (axis == 0) {
		for (i = 0; i < len; i++) {
			if (points[i].x > max_x) {
				max_x = points[i].x;
			}
			if (points[i].y > max_y) {
				max_y = points[i].y;
			}
		}
	}

	return Point2f(max_x, max_y);
}
Point max2i(Point * points, int len, int axis) {
	int i;
	int max_x = points[0].x;
	int max_y = points[0].y;

	if (axis == 0) {
		for (i = 0; i < len; i++) {
			if (points[i].x > max_x) {
				max_x = points[i].x;
			}
			if (points[i].y > max_y) {
				max_y = points[i].y;
			}
		}
	}

	return Point(max_x, max_y);
}

float norm(Point2f p) {
	float dist = sqrt(pow(p.x, 2) + pow(p.y, 2));
	return dist;
}

Point2f mean2f(Point2f* points, int len, int axis) {
	int i;
	float mean_x = 0;
	float mean_y = 0;

	if (axis == 0) {
		for (i = 0; i < len; i++) {
			mean_x += points[i].x;
			mean_y += points[i].y;
		}
	}

	return Point2f(float(mean_x / len), float(mean_y / len));
}
Point mean2i(Point* points, int len, int axis) {
	int i;
	float mean_x = 0;
	float mean_y = 0;

	if (axis == 0) {
		for (i = 0; i < len; i++) {
			mean_x += points[i].x;
			mean_y += points[i].y;
		}
	}

	return Point((int)(mean_x / len), (int)(mean_y / len));
}
Point2f fmean2i(Point* points, int len, int axis) {
	int i;
	float mean_x = 0;
	float mean_y = 0;

	if (axis == 0) {
		for (i = 0; i < len; i++) {
			mean_x += points[i].x;
			mean_y += points[i].y;
		}
	}

	return Point2f(mean_x / len, mean_y / len);
}