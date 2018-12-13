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
	zikken_output = ofstream("2�l��gpu����.log");

	int ngpu = cv::cuda::getCudaEnabledDeviceCount();
	cout << ngpu << endl;
	//OpenCV �r���h���̕\��
	cout << cv::getBuildInformation() << endl;

	//GPU �f�o�C�X���̕\��
	cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());

	//GPU �f�o�C�X�����X�ŕ\�����邱�Ƃ��\
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
	//�K���I臒l�̃��[�v
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

		//�K���I臒l�̃e�X�g
		//cpu�ł̓K���I臒l
		QueryPerformanceCounter(&gpu_timer);
		cv::cvtColor(SrcImage, SrcGreyImage, cv::COLOR_BGR2GRAY);
		Timer(gpu_timer, "cpu�O���[��");
		cv::adaptiveThreshold(SrcGreyImage, SrcBinImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, ksize, c);
		Timer(gpu_timer, "cpu�Q�l��");
		cv::namedWindow("bin normal", cv::WINDOW_AUTOSIZE);
		cv::imshow("bin normal", SrcBinImage);

		//gpu�ł̓K���I臒l
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
	//GPU���g����RGB����O���[�X�P�[���̕ϊ��e�X�g
	cv::cuda::GpuMat GPU_SrcImage(srcImage);
	cv::cuda::GpuMat GPU_GreyImage;
	cv::cuda::GpuMat GPU_CompareImage;
	cv::cuda::GpuMat GPU_BinImage;
	double sigma = 0.3*(ksize / 2 - 1) + 0.8;
	Timer(gpu_timer, "gpu�摜��gpu�ɃR�s�[");

	cv::cuda::cvtColor(GPU_SrcImage, GPU_GreyImage, cv::COLOR_BGR2GRAY);

	Timer(gpu_timer, "gpu�摜���O���[��");

	cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(ksize, ksize), sigma, sigma);
	gaussian_filter->apply(GPU_GreyImage, GPU_CompareImage);
	cv::cuda::GpuMat C = cv::cuda::GpuMat(GPU_SrcImage.size(), CV_8UC1, cv::Scalar(c));
	cv::cuda::subtract(GPU_CompareImage, C, GPU_CompareImage);
	cv::cuda::compare(GPU_GreyImage, GPU_CompareImage, GPU_BinImage, cv::CMP_GT);

	Timer(gpu_timer, "gpu�摜��gpu�ɂQ�l��");
	GPU_BinImage.download(binImage);
	Timer(gpu_timer, "gpu�摜��cpu�ɃR�s�[");

}

// �v���O�����̎��s: Ctrl + F5 �܂��� [�f�o�b�O] > [�f�o�b�O�Ȃ��ŊJ�n] ���j���[
// �v���O�����̃f�o�b�O: F5 �܂��� [�f�o�b�O] > [�f�o�b�O�̊J�n] ���j���[

// ��Ƃ��J�n���邽�߂̃q���g: 
//    1. �\�����[�V���� �G�N�X�v���[���[ �E�B���h�E���g�p���ăt�@�C����ǉ�/�Ǘ����܂� 
//   2. �`�[�� �G�N�X�v���[���[ �E�B���h�E���g�p���ă\�[�X�Ǘ��ɐڑ����܂�
//   3. �o�̓E�B���h�E���g�p���āA�r���h�o�͂Ƃ��̑��̃��b�Z�[�W��\�����܂�
//   4. �G���[�ꗗ�E�B���h�E���g�p���ăG���[��\�����܂�
//   5. [�v���W�F�N�g] > [�V�������ڂ̒ǉ�] �ƈړ����ĐV�����R�[�h �t�@�C�����쐬���邩�A[�v���W�F�N�g] > [�����̍��ڂ̒ǉ�] �ƈړ����Ċ����̃R�[�h �t�@�C�����v���W�F�N�g�ɒǉ����܂�
//   6. ��قǂ��̃v���W�F�N�g���ĂъJ���ꍇ�A[�t�@�C��] > [�J��] > [�v���W�F�N�g] �ƈړ����� .sln �t�@�C����I�����܂�
