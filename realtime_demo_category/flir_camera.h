#include "stdafx.h"
#pragma once
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
using namespace Spinnaker;
int SetCamera(Spinnaker::CameraPtr &, Spinnaker::SystemPtr &);

int GetImage(CameraPtr & pCam, ImagePtr & convertedImage);

int GetSingleImage(CameraPtr & pCam, ImagePtr & pResultImage);

int ImageToMat(ImagePtr & pResultImage, Mat &result);

int FinishCamera(CameraPtr & pCam, SystemPtr & system);

int aqcuisition(int, char **);
