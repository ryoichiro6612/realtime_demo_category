#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include "edge.h"

using namespace cv;
using namespace std;

int edgeThresh = 50;
Mat image, gray, edge, cedge;

// define a trackbar callback
static void onTrackbar(int, void*)
{
	blur(gray, edge, Size(3, 3));

	// Run the edge detector on grayscale
	Canny(edge, edge, edgeThresh, edgeThresh * 3, 3);
	cedge = Scalar::all(0);


	image.copyTo(cedge, edge);

	cv::cvtColor(cedge, cedge, CV_RGB2GRAY);
	cedge = ~cedge;
	threshold(cedge, cedge, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("Edge map", cedge);
}

static void help()
{
	printf("\nThis sample demonstrates Canny edge detection\n"
		"Call:\n"
		"    /.edge [image_name -- Default is fruits.jpg]\n\n");
}

const char* keys =
{
	"{1| |fruits.jpg|input image name}"
};

Mat makeedge(Mat inputimage)
{
	help();
	image = inputimage;
	cedge.create(image.size(), image.type());
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// Create a window
	namedWindow("Edge map", 1);

	// create a toolbar
	createTrackbar("Canny threshold", "Edge map", &edgeThresh, 100, onTrackbar);

	// Show the image
	onTrackbar(0, 0);

	// Wait for a key stroke; the same function arranges events processing
	waitKey(1);

	return cedge;
}
