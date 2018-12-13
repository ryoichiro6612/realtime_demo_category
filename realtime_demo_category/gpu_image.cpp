#include "stdafx.h"

#include <iostream>
#include <opencv/cv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv_lib.hpp>
#include <opencv2/cudafilters.hpp>
#include "gpu_image.h"
#include "zikken_state.h"

using namespace std;
LARGE_INTEGER gpu_timer;

int gpu_image()
{
	extern std::ofstream zikken_output;
	zikken_output = ofstream("2値化gpu実験.log");

	int ngpu = cv::cuda::getCudaEnabledDeviceCount();
	cout << ngpu << endl;
	//OpenCV ビルド情報の表示
	cout << cv::getBuildInformation() << endl;

	//GPU デバイス情報の表示
	cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());

	//GPU デバイス情報を個々で表示することも可能
	cv::cuda::DeviceInfo info(0);
	cout <<
		"Name (Device ID): " << info.name() << endl <<
		"MajorVersion: " << info.majorVersion() << endl <<
		"MinorVersion: " << info.minorVersion() << endl <<
		"Clock rate:" << info.clockRate() << endl <<
		"FreeMemory: " << info.freeMemory() << endl <<
		"TotalMemory: " << info.totalMemory() << endl <<
		"isCompatible: " << info.isCompatible() << endl <<
		"Global Memory bus width(bits):" << info.memoryBusWidth() << endl <<
		"supports(FEATURE_SET_COMPUTE_10): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_10) << endl <<
		"supports(FEATURE_SET_COMPUTE_11): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_11) << endl <<
		"supports(FEATURE_SET_COMPUTE_12): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_12) << endl <<
		"supports(FEATURE_SET_COMPUTE_13): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_13) << endl <<
		"supports(FEATURE_SET_COMPUTE_20): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_20) << endl <<
		"supports(FEATURE_SET_COMPUTE_21): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_21) << endl <<
		"supports(FEATURE_SET_COMPUTE_30): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_30) << endl <<
		"supports(FEATURE_SET_COMPUTE_35): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_35) << endl <<
		"supports(FEATURE_SET_COMPUTE_50): " << info.supports(cv::cuda::FEATURE_SET_COMPUTE_50)
		<< endl;
	cv::VideoCapture cap("zikken.wmv");
	//適応的閾値のループ
	int ksize = 25;
	int c = 9;

	extern LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	int count = 0;
	while (1) {
		cv::Mat SrcImage;
		cap >> SrcImage;
		cv::Mat SrcGreyImage, SrcBinImage, GpuBinImage;
		if (SrcImage.rows == 0) break;
		zikken_output << "progress:" << count << std::endl;
		count++;

		//適応的閾値のテスト
		//cpuでの適応的閾値
		QueryPerformanceCounter(&gpu_timer);
		cv::cvtColor(SrcImage, SrcGreyImage, cv::COLOR_BGR2GRAY);
		Timer(gpu_timer, "cpuグレー化");
		cv::adaptiveThreshold(SrcGreyImage, SrcBinImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, ksize, c);
		Timer(gpu_timer, "cpu２値化");
		cv::namedWindow("bin normal", cv::WINDOW_AUTOSIZE);
		cv::imshow("bin normal", SrcBinImage);

		//gpuでの適応的閾値
		QueryPerformanceCounter(&gpu_timer);
		adaptiveThresholdGPU(SrcImage, GpuBinImage, ksize, c);
		cv::imshow("bin gpu", GpuBinImage);

		cv::Mat subtractImage;
		cv::subtract(SrcBinImage, GpuBinImage, subtractImage);
		cv::namedWindow("subtract", cv::WINDOW_AUTOSIZE);
		cv::imshow("subtract", subtractImage);
		cv::waitKey(1);
	}

	return 0;
}

void adaptiveThresholdGPU(cv::Mat &srcImage, cv::Mat &binImage, int ksize, int c) {
	QueryPerformanceCounter(&gpu_timer);
	//GPUを使ってRGBからグレースケールの変換テスト
	cv::cuda::GpuMat GPU_SrcImage(srcImage);
	cv::cuda::GpuMat GPU_GreyImage;
	cv::cuda::GpuMat GPU_CompareImage;
	cv::cuda::GpuMat GPU_BinImage;
	double sigma = 0.3*(ksize / 2 - 1) + 0.8;
	Timer(gpu_timer, "gpu画像をgpuにコピー");

	cv::cuda::cvtColor(GPU_SrcImage, GPU_GreyImage, cv::COLOR_BGR2GRAY);

	Timer(gpu_timer, "gpu画像をグレー化");

	cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(ksize, ksize), sigma, sigma);
	gaussian_filter->apply(GPU_GreyImage, GPU_CompareImage);
	cv::cuda::GpuMat C = cv::cuda::GpuMat(GPU_SrcImage.size(), CV_8UC1, cv::Scalar(c));
	cv::cuda::subtract(GPU_CompareImage, C, GPU_CompareImage);
	cv::cuda::compare(GPU_GreyImage, GPU_CompareImage, GPU_BinImage, cv::CMP_GT);

	Timer(gpu_timer, "gpu画像をgpuに２値化");
	GPU_BinImage.download(binImage);
	Timer(gpu_timer, "gpu画像をcpuにコピー");

}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
