#pragma once
#include "stdafx.h"
#include <fstream>
#include <Windows.h>
#include <string.h>
extern bool iti_heiretu;
extern bool id_heiretu;
extern bool use_gpu;
extern bool camera_heiretu;
extern bool use_lc_moving;
extern bool use_id_moving;
extern bool flir_camera;
extern std::ofstream zikken_output;
extern LARGE_INTEGER freq;
#define NTEST 300
void printZikkenState();
void Timer(LARGE_INTEGER & prev_timer, std::string label, double minus = 0);