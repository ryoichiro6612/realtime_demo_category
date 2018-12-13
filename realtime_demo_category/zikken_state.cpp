#include "stdafx.h"
#include "zikken_state.h"
#include <iostream>
#include <string>

void printZikkenState()
{
	zikken_output
		<< "FLIR�J�����̎d�l#" << flir_camera << std::endl
		<< "�ʒu�}�[�J�̕���#" << iti_heiretu << std::endl
		<< "id�}�[�J�̕���#" << id_heiretu << std::endl
		<< "gpu����#" << use_gpu << std::endl
		<< "�ȑO�̈ʒu�}�[�J�̗��p#" << use_lc_moving << std::endl
		<< "�ȑO��id�}�[�J�̗��p#" << use_id_moving << std::endl;
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
