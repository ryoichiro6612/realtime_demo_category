#pragma once

int gpu_image();

void adaptiveThresholdGPU(cv::Mat & srcImage, cv::Mat & binImage, int ksize, int c);
