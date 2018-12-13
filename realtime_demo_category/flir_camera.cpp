#include "stdafx.h"
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream> 
#include "spinnaker_lib.hpp"
#include "zikken_state.h"

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv_lib.hpp>
#include <opencv2/cudafilters.hpp>

#include "post_util.h"


using namespace Spinnaker;
//=============================================================================
// Copyright (c) 2001-2018 FLIR Systems, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of FLIR
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You
// shall not disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).
//
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================

/**
 *  @example Acquisition.cpp
 *
 *  @brief Acquisition.cpp shows how to acquire images. It relies on
 *  information provided in the Enumeration example. Also, check out the
 *  ExceptionHandling and NodeMapInfo examples if you haven't already.
 *  ExceptionHandling shows the handling of standard and Spinnaker exceptions
 *  while NodeMapInfo explores retrieving information from various node types.
 *
 *  This example touches on the preparation and cleanup of a camera just before
 *  and just after the acquisition of images. Image retrieval and conversion,
 *  grabbing image data, and saving images are all covered as well.
 *
 *  Once comfortable with Acquisition, we suggest checking out
 *  AcquisitionMultipleCamera, NodeMapCallback, or SaveToAvi.
 *  AcquisitionMultipleCamera demonstrates simultaneously acquiring images from
 *  a number of cameras, NodeMapCallback serves as a good introduction to
 *  programming with callbacks and events, and SaveToAvi exhibits video creation.
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream> 
#include "spinnaker_lib.hpp"
#include "flir_camera.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// Total number of buffers
#define numBuffers 3

// Number of triggers
#define z_numTriggers 6

// Total number of loops
#define k_numLoops 9

#ifdef _DEBUG
// Disables heartbeat on GEV cameras so debugging does not incur timeout errors
int DisableHeartbeat(CameraPtr pCam, INodeMap & nodeMap, INodeMap & nodeMapTLDevice)
{
	cout << "Checking device type to see if we need to disable the camera's heartbeat..." << endl << endl;
	//
	// Write to boolean node controlling the camera's heartbeat
	// 
	// *** NOTES ***
	// This applies only to GEV cameras and only applies when in DEBUG mode.
	// GEV cameras have a heartbeat built in, but when debugging applications the
	// camera may time out due to its heartbeat. Disabling the heartbeat prevents 
	// this timeout from occurring, enabling us to continue with any necessary debugging.
	// This procedure does not affect other types of cameras and will prematurely exit
	// if it determines the device in question is not a GEV camera. 
	//
	// *** LATER ***
	// Since we only disable the heartbeat on GEV cameras during debug mode, it is better
	// to power cycle the camera after debugging. A power cycle will reset the camera 
	// to its default settings. 
	// 

	CEnumerationPtr ptrDeviceType = nodeMapTLDevice.GetNode("DeviceType");
	if (!IsAvailable(ptrDeviceType) && !IsReadable(ptrDeviceType))
	{
		cout << "Error with reading the device's type. Aborting..." << endl << endl;
		return -1;
	}
	else
	{
		if (ptrDeviceType->GetIntValue() == DeviceType_GEV)
		{
			cout << "Working with a GigE camera. Attempting to disable heartbeat before continuing..." << endl << endl;
			CBooleanPtr ptrDeviceHeartbeat = nodeMap.GetNode("GevGVCPHeartbeatDisable");
			if (!IsAvailable(ptrDeviceHeartbeat) || !IsWritable(ptrDeviceHeartbeat))
			{
				cout << "Unable to disable heartbeat on camera. Continuing with execution as this may be non-fatal..." << endl << endl;
			}
			else
			{
				ptrDeviceHeartbeat->SetValue(true);
				cout << "WARNING: Heartbeat on GigE camera disabled for the rest of Debug Mode." << endl;
				cout << "         Power cycle camera when done debugging to re-enable the heartbeat..." << endl << endl;
			}
		}
		else
		{
			cout << "Camera does not use GigE interface. Resuming normal execution..." << endl << endl;
		}
	}
	return 0;
}
#endif

// This function acquires and saves 10 images from a device.  
int AcquireImages(CameraPtr pCam, INodeMap & nodeMap, INodeMap & nodeMapTLDevice)
{
	int result = 0;

	cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

	try
	{
		//
		// Set acquisition mode to continuous
		//
		// *** NOTES ***
		// Because the example acquires and saves 10 images, setting acquisition 
		// mode to continuous lets the example finish. If set to single frame
		// or multiframe (at a lower number of images), the example would just
		// hang. This would happen because the example has been written to
		// acquire 10 images while the camera would have been programmed to 
		// retrieve less than that.
		// 
		// Setting the value of an enumeration node is slightly more complicated
		// than other node types. Two nodes must be retrieved: first, the 
		// enumeration node is retrieved from the nodemap; and second, the entry
		// node is retrieved from the enumeration node. The integer value of the
		// entry node is then set as the new value of the enumeration node.
		//
		// Notice that both the enumeration and the entry nodes are checked for
		// availability and readability/writability. Enumeration nodes are
		// generally readable and writable whereas their entry nodes are only
		// ever readable.
		// 
		// Retrieve enumeration node from nodemap
		CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
		if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
		{
			cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
			return -1;
		}

		// Retrieve entry node from enumeration node
		CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
		if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
		{
			cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
			return -1;
		}

		// Retrieve integer value from entry node
		int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

		// Set integer value from entry node as new value of enumeration node
		ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

		cout << "Acquisition mode set to continuous..." << endl;

#ifdef _DEBUG
		cout << endl << endl << "*** DEBUG ***" << endl << endl;

		// If using a GEV camera and debugging, should disable heartbeat first to prevent further issues
		if (DisableHeartbeat(pCam, nodeMap, nodeMapTLDevice) != 0)
		{
			return -1;
		}

		cout << endl << endl << "*** END OF DEBUG ***" << endl << endl;
#endif

		//
		// Begin acquiring images
		//
		// *** NOTES ***
		// What happens when the camera begins acquiring images depends on the
		// acquisition mode. Single frame captures only a single image, multi 
		// frame catures a set number of images, and continuous captures a 
		// continuous stream of images. Because the example calls for the 
		// retrieval of 10 images, continuous mode has been set.
		// 
		// *** LATER ***
		// Image acquisition must be ended when no more images are needed.
		//
		pCam->BeginAcquisition();

		cout << "Acquiring images..." << endl;

		//
		// Retrieve device serial number for filename
		//
		// *** NOTES ***
		// The device serial number is retrieved in order to keep cameras from 
		// overwriting one another. Grabbing image IDs could also accomplish
		// this.
		//
		gcstring deviceSerialNumber("");
		CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
		if (IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial))
		{
			deviceSerialNumber = ptrStringSerial->GetValue();

			cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
		}
		cout << endl;

		// Retrieve, convert, and save images
		const unsigned int k_numImages = 10;

		for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
		{
			try
			{
				//
				// Retrieve next received image
				//
				// *** NOTES ***
				// Capturing an image houses images on the camera buffer. Trying
				// to capture an image that does not exist will hang the camera.
				//
				// *** LATER ***
				// Once an image from the buffer is saved and/or no longer 
				// needed, the image must be released in order to keep the 
				// buffer from filling up.
				//
				ImagePtr pResultImage = pCam->GetNextImage();

				//
				// Ensure image completion
				//
				// *** NOTES ***
				// Images can easily be checked for completion. This should be
				// done whenever a complete image is expected or required.
				// Further, check image status for a little more insight into
				// why an image is incomplete.
				//
				if (pResultImage->IsIncomplete())
				{
					// Retreive and print the image status description
					cout << "Image incomplete: "
						<< Image::GetImageStatusDescription(pResultImage->GetImageStatus())
						<< "..." << endl << endl;
				}
				else
				{
					//
					// Print image information; height and width recorded in pixels
					//
					// *** NOTES ***
					// Images have quite a bit of available metadata including
					// things such as CRC, image status, and offset values, to
					// name a few.
					//
					size_t width = pResultImage->GetWidth();

					size_t height = pResultImage->GetHeight();

					cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << endl;

					//
					// Convert image to mono 8
					//
					// *** NOTES ***
					// Images can be converted between pixel formats by using 
					// the appropriate enumeration value. Unlike the original 
					// image, the converted one does not need to be released as 
					// it does not affect the camera buffer.
					//
					// When converting images, color processing algorithm is an
					// optional parameter.
					// 
					ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);

					// Create a unique filename
					ostringstream filename;

					filename << "Acquisition-";
					if (deviceSerialNumber != "")
					{
						filename << deviceSerialNumber.c_str() << "-";
					}
					filename << imageCnt << ".jpg";

					//
					// Save image
					// 
					// *** NOTES ***
					// The standard practice of the examples is to use device
					// serial numbers to keep images of one device from 
					// overwriting those of another.
					//
					convertedImage->Save(filename.str().c_str());

					cout << "Image saved at " << filename.str() << endl;
				}

				//
				// Release image
				//
				// *** NOTES ***
				// Images retrieved directly from the camera (i.e. non-converted
				// images) need to be released in order to keep from filling the
				// buffer.
				//
				pResultImage->Release();

				cout << endl;
			}
			catch (Spinnaker::Exception &e)
			{
				cout << "Error: " << e.what() << endl;
				result = -1;
			}
		}

		//
		// End acquisition
		//
		// *** NOTES ***
		// Ending acquisition appropriately helps ensure that devices clean up
		// properly and do not need to be power-cycled to maintain integrity.
		//

		pCam->EndAcquisition();


	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	return result;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int PrintDeviceInfo(INodeMap & nodeMap)
{
	int result = 0;

	cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

	try
	{
		FeatureList_t features;
		CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
		if (IsAvailable(category) && IsReadable(category))
		{
			category->GetFeatures(features);

			FeatureList_t::const_iterator it;
			for (it = features.begin(); it != features.end(); ++it)
			{
				CNodePtr pfeatureNode = *it;
				cout << pfeatureNode->GetName() << " : ";
				CValuePtr pValue = (CValuePtr)pfeatureNode;
				cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
				cout << endl;
			}
		}
		else
		{
			cout << "Device control information not available." << endl;
		}
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	return result;
}

// This function acts as the body of the example; please see NodeMapInfo example 
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam)
{
	int result = 0;

	try
	{
		// Retrieve TL device nodemap and print device information
		INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

		result = PrintDeviceInfo(nodeMapTLDevice);

		// Initialize camera
		pCam->Init();

		// Retrieve GenICam nodemap
		INodeMap & nodeMap = pCam->GetNodeMap();

		// Acquire images
		result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice);

		// Deinitialize camera
		pCam->DeInit();
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}

	return result;
}

int SetCamera(CameraPtr &pCam, SystemPtr &system) {
	int result = 0;

	// Retrieve singleton reference to system object
	system = System::GetInstance();

	// Retrieve list of cameras from the system
	CameraList camList = system->GetCameras();

	unsigned int numCameras = camList.GetSize();

	cout << "Number of cameras detected: " << numCameras << endl << endl;

	// Finish if there are no cameras
	if (numCameras == 0)
	{
		// Clear camera list before releasing system
		camList.Clear();

		// Release system
		system->ReleaseInstance();

		cout << "Not enough cameras!" << endl;
		cout << "Done! Press Enter to exit..." << endl;
		getchar();

		return -1;
	}
	// Select camera
	pCam = camList.GetByIndex(0);

	try
	{
		// Retrieve TL device nodemap and print device information
		INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

		result = PrintDeviceInfo(nodeMapTLDevice);

		// Initialize camera
		pCam->Init();

		// Retrieve GenICam nodemap
		INodeMap & nodeMap = pCam->GetNodeMap();

		try
		{
			CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
			if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
			{
				cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
				return -1;
			}

			// Retrieve entry node from enumeration node
			CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
			if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
			{
				cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
				return -1;
			}

			// Retrieve integer value from entry node
			int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

			// Set integer value from entry node as new value of enumeration node
			ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

			cout << "Acquisition mode set to continuous..." << endl;

			//バッファの設定
			// Retrieve Stream Parameters device nodemap 
			Spinnaker::GenApi::INodeMap & sNodeMap = pCam->GetTLStreamNodeMap();
			// Retrieve Buffer Handling Mode Information
			CEnumerationPtr ptrHandlingMode = sNodeMap.GetNode("StreamBufferHandlingMode");
			if (!IsAvailable(ptrHandlingMode) || !IsWritable(ptrHandlingMode))
			{
				cout << "Unable to set Buffer Handling mode (node retrieval). Aborting..." << endl << endl;
				return -1;
			}

			CEnumEntryPtr ptrHandlingModeEntry = ptrHandlingMode->GetCurrentEntry();
			if (!IsAvailable(ptrHandlingModeEntry) || !IsReadable(ptrHandlingModeEntry))
			{
				cout << "Unable to set Buffer Handling mode (Entry retrieval). Aborting..." << endl << endl;
				return -1;
			}

			// Set stream buffer Count Mode to manual
			CEnumerationPtr ptrStreamBufferCountMode = sNodeMap.GetNode("StreamBufferCountMode");
			if (!IsAvailable(ptrStreamBufferCountMode) || !IsWritable(ptrStreamBufferCountMode))
			{
				cout << "Unable to set Buffer Count Mode (node retrieval). Aborting..." << endl << endl;
				return -1;
			}

			CEnumEntryPtr ptrStreamBufferCountModeManual = ptrStreamBufferCountMode->GetEntryByName("Manual");
			if (!IsAvailable(ptrStreamBufferCountModeManual) || !IsReadable(ptrStreamBufferCountModeManual))
			{
				cout << "Unable to set Buffer Count Mode entry (Entry retrieval). Aborting..." << endl << endl;
				return -1;
			}

			ptrStreamBufferCountMode->SetIntValue(ptrStreamBufferCountModeManual->GetValue());

			cout << "Stream Buffer Count Mode set to manual..." << endl;

			// Retrieve and modify Stream Buffer Count
			CIntegerPtr ptrBufferCount = sNodeMap.GetNode("StreamBufferCountManual");
			if (!IsAvailable(ptrBufferCount) || !IsWritable(ptrBufferCount))
			{
				cout << "Unable to set Buffer Count (Integer node retrieval). Aborting..." << endl << endl;
				return -1;
			}

			// Display Buffer Info
			cout << endl << "Default Buffer Handling Mode: " << ptrHandlingModeEntry->GetDisplayName() << endl;
			cout << "Default Buffer Count: " << ptrBufferCount->GetValue() << endl;
			cout << "Maximum Buffer Count: " << ptrBufferCount->GetMax() << endl;

			ptrBufferCount->SetValue(numBuffers);

			cout << "Buffer count now set to: " << ptrBufferCount->GetValue() << endl;

			cout << endl << "Camera will be triggered " << z_numTriggers << " times in a row before " << k_numLoops - z_numTriggers << " images will be retrieved" << endl;

			ptrHandlingModeEntry = ptrHandlingMode->GetEntryByName("NewestOnly");
			ptrHandlingMode->SetIntValue(ptrHandlingModeEntry->GetValue());
			cout << endl << endl << "Buffer Handling Mode has been set to " << ptrHandlingModeEntry->GetDisplayName() << endl;

#ifdef _DEBUG
			cout << endl << endl << "*** DEBUG ***" << endl << endl;

			// If using a GEV camera and debugging, should disable heartbeat first to prevent further issues
			if (DisableHeartbeat(pCam, nodeMap, nodeMapTLDevice) != 0)
			{
				return -1;
			}

			cout << endl << endl << "*** END OF DEBUG ***" << endl << endl;
#endif


			//
			// Begin acquiring images
			//
			// *** NOTES ***
			// What happens when the camera begins acquiring images depends on the
			// acquisition mode. Single frame captures only a single image, multi 
			// frame catures a set number of images, and continuous captures a 
			// continuous stream of images. Because the example calls for the 
			// retrieval of 10 images, continuous mode has been set.
			// 
			// *** LATER ***
			// Image acquisition must be ended when no more images are needed.
			//
			pCam->BeginAcquisition();

			cout << "Acquiring images..." << endl;

			//
			// Retrieve device serial number for filename
			//
			// *** NOTES ***
			// The device serial number is retrieved in order to keep cameras from 
			// overwriting one another. Grabbing image IDs could also accomplish
			// this.
			//
			gcstring deviceSerialNumber("");
			CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
			if (IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial))
			{
				deviceSerialNumber = ptrStringSerial->GetValue();

				cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
			}
			cout << endl;

		}
		catch (Spinnaker::Exception &e)
		{
			cout << "Error: " << e.what() << endl;
			result = -1;
		}
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		result = -1;
	}
	return 0;
}
int GetImage(CameraPtr &pCam, ImagePtr &convertedImage) {
	LARGE_INTEGER prev_timer;
	QueryPerformanceCounter(&prev_timer);
	ImagePtr pResultImage;
	try
	{

		int captureCount = 0;
		for (captureCount = 0; captureCount < 1; captureCount++) {
			pResultImage = pCam->GetNextImage();
		}
		Timer(prev_timer, "capture");
			
		if (pResultImage->IsIncomplete())
		{
			// Retreive and print the image status description
			cout << "Image incomplete: "
				<< Image::GetImageStatusDescription(pResultImage->GetImageStatus())
				<< "..." << endl << endl;
		}
		else {
			//
		// Print image information; height and width recorded in pixels
		//
		// *** NOTES ***
		// Images have quite a bit of available metadata including
		// things such as CRC, image status, and offset values, to
		// name a few.
		//
			size_t width = pResultImage->GetWidth();

			size_t height = pResultImage->GetHeight();

			cout << "Grabbed image " << ", width = " << width << ", height = " << height << endl;
			//convertedImage = pResultImage->Convert(PixelFormat_BGR8, NEAREST_NEIGHBOR);
			//Mat result(Size(width, height), CV_8UC1, pResultImage->GetData());
			//Mat converted;
			//cuda::GpuMat gpuresult(result);
			//cuda::GpuMat gpuconverted;
			//Ptr<cv::cuda::Filter> filter = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(2, 2));
			//filter->apply(gpuresult, gpuconverted);
			//gpuconverted.download(converted);
			////convertedImage = pResultImage->Convert(PixelFormat_Mono8);
			//Timer(prev_timer, "convert");
			//Mat resized(Size(converted.cols / 5, converted.rows / 5), CV_8UC1);
			//cv::resize(converted, resized, Size(result.cols / 5, result.rows / 5));
			//cv::namedWindow("current Images", CV_WINDOW_AUTOSIZE);
			//cv::imshow("current Images", converted);
			//cv::waitKey(1);
			convertedImage = pResultImage->Convert(PixelFormat_BGR8, NEAREST_NEIGHBOR);
		}
		//
		// Release image
		//
		// *** NOTES ***
		// Images retrieved directly from the camera (i.e. non-converted
		// images) need to be released in order to keep from filling the
		// buffer.
		//
		pResultImage->Release();
		Timer(prev_timer, "release");

		std::cout << endl;
	}
	catch (Spinnaker::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		return -1;
	}
}
int ImageToMat(ImagePtr &convertedImage, Mat &result) {
	LARGE_INTEGER prev_timer;
	QueryPerformanceCounter(&prev_timer);
	unsigned int XPadding = convertedImage->GetXPadding();
	unsigned int YPadding = convertedImage->GetYPadding();
	unsigned int rowsize = convertedImage->GetWidth();
	unsigned int colsize = convertedImage->GetHeight();

	//image data contains padding. When allocating Mat container size, you need to account for the X,Y image data padding. 
	result = cv::Mat(colsize + YPadding, rowsize + XPadding, CV_8UC3, convertedImage->GetData(), convertedImage->GetStride());
	//result = Mat(colsize + YPadding, rowsize + XPadding, CV_8UC1, convertedImage->GetData(), convertedImage->GetStride());
	Timer(prev_timer, "imagetoMat");
	/*
	Mat resized(Size(result.cols / 5, result.rows / 5), CV_8UC1);
	cv::resize(result, resized, Size(result.cols / 5, result.rows / 5));
	cv::namedWindow("current Image", CV_WINDOW_AUTOSIZE);
	cv::imshow("current Image", resized);
	cv::waitKey(1);
	//*/
	return 1;
}
int FinishCamera(CameraPtr &pCam, SystemPtr &system) {
	try {
		pCam->EndAcquisition();
		pCam->DeInit();
	}
	catch (Spinnaker::Exception &e) {
		cout << "Error: " << e.what() << endl;
		return -1;
	}
	pCam = NULL;
	CameraList camList = system->GetCameras();
	// Clear camera list before releasing system
	camList.Clear();

	// Release system
	system->ReleaseInstance();

	cout << endl << "Done! Press Enter to exit..." << endl;
	getchar();
}
// Example entry point; please see Enumeration example for more in-depth 
// comments on preparing and cleaning up the system.
int aqcuisition(int /*argc*/, char** /*argv*/)
{

	int result = 0;

	// Retrieve singleton reference to system object
	SystemPtr system = System::GetInstance();

	// Print out current library version
	const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
	cout << "Spinnaker library version: "
		<< spinnakerLibraryVersion.major << "."
		<< spinnakerLibraryVersion.minor << "."
		<< spinnakerLibraryVersion.type << "."
		<< spinnakerLibraryVersion.build << endl << endl;

	// Retrieve list of cameras from the system
	CameraList camList = system->GetCameras();

	unsigned int numCameras = camList.GetSize();

	cout << "Number of cameras detected: " << numCameras << endl << endl;

	// Finish if there are no cameras
	if (numCameras == 0)
	{
		// Clear camera list before releasing system
		camList.Clear();

		// Release system
		system->ReleaseInstance();

		cout << "Not enough cameras!" << endl;
		cout << "Done! Press Enter to exit..." << endl;
		getchar();

		return -1;
	}

	//
	// Create shared pointer to camera
	//
	// *** NOTES ***
	// The CameraPtr object is a shared pointer, and will generally clean itself
	// up upon exiting its scope. However, if a shared pointer is created in the
	// same scope that a system object is explicitly released (i.e. this scope),
	// the reference to the shared point must be broken manually.
	//
	// *** LATER ***
	// Shared pointers can be terminated manually by assigning them to NULL.
	// This keeps releasing the system from throwing an exception.
	//
	CameraPtr pCam = NULL;

	// Run example on each camera
	for (unsigned int i = 0; i < numCameras; i++)
	{
		// Select camera
		pCam = camList.GetByIndex(i);

		cout << endl << "Running example for camera " << i << "..." << endl;

		// Run example
		result = result | RunSingleCamera(pCam);

		cout << "Camera " << i << " example complete..." << endl << endl;
	}

	//
	// Release reference to the camera
	//
	// *** NOTES ***
	// Had the CameraPtr object been created within the for-loop, it would not
	// be necessary to manually break the reference because the shared pointer
	// would have automatically cleaned itself up upon exiting the loop.
	//
	pCam = NULL;

	// Clear camera list before releasing system
	camList.Clear();

	// Release system
	system->ReleaseInstance();

	cout << endl << "Done! Press Enter to exit..." << endl;
	getchar();

	return result;
}
