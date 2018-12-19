#include "cuda_lib.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_calc.h"
#include <iostream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include <fstream>
char* cuda_malloc(char *h_data) {
	std::ofstream mi("mi.log");
	//tasu
	char *d_data;
	int nByte = 4000 * 3000 * sizeof(char);
	int i;
	LARGE_INTEGER now_timer, prev_timer;
	QueryPerformanceCounter(&prev_timer);
	LARGE_INTEGER timer_freq;
	QueryPerformanceFrequency(&timer_freq);

	//std::cout << label << "," << double(now_timer.QuadPart - prev_timer.QuadPart) * 1000 / timer_freq.QuadPart << std::endl;
	QueryPerformanceCounter(&prev_timer);
	cudaMalloc((void**)&d_data, nByte);
	cudaMemcpy(d_data, h_data, nByte, cudaMemcpyHostToDevice);
	return d_data;
	QueryPerformanceCounter(&now_timer);
	mi << "mallocmemcopy" << "," << double(now_timer.QuadPart - prev_timer.QuadPart) * 1000 / timer_freq.QuadPart << std::endl;
	QueryPerformanceCounter(&prev_timer);
	
	QueryPerformanceCounter(&now_timer);
	mi << "mallocmemcopy" << "," << double(now_timer.QuadPart - prev_timer.QuadPart) * 1000 / timer_freq.QuadPart << std::endl;
}