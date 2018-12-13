#include "stdafx.h"
#include "zikken_state.h"
#include <iostream>
#include <string>

void printZikkenState()
{
	zikken_output
		<< "FLIRカメラの仕様#" << flir_camera << std::endl
		<< "位置マーカの並列化#" << iti_heiretu << std::endl
		<< "idマーカの並列化#" << id_heiretu << std::endl
		<< "gpu並列化#" << use_gpu << std::endl
		<< "以前の位置マーカの利用#" << use_lc_moving << std::endl
		<< "以前のidマーカの利用#" << use_id_moving << std::endl;
}
void Timer(LARGE_INTEGER &prev_timer, std::string label) {
	LARGE_INTEGER now_timer;
	QueryPerformanceCounter(&now_timer);
	LARGE_INTEGER timer_freq;
	QueryPerformanceFrequency(&timer_freq);
	zikken_output << label << "," << double(now_timer.QuadPart - prev_timer.QuadPart) * 1000 / timer_freq.QuadPart << std::endl;
	//std::cout << label << "," << double(now_timer.QuadPart - prev_timer.QuadPart) * 1000 / timer_freq.QuadPart << std::endl;
	QueryPerformanceCounter(&prev_timer);
}
