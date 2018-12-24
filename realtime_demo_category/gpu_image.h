#pragma once

int gpu_image();

void adaptiveThresholdGPU(cv::cuda::GpuMat & GPU_SrcImage, cv::Mat & binImage, int ksize, int c);