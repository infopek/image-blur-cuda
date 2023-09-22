#pragma once

#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "constants.h"

using namespace Constants;

void showImage(unsigned short image[height][width])
{
	cv::Mat img(width, height, CV_8UC1, image);
	//cv::resize(img, img, cv::Size(300, 300), 0, 0, cv::InterpolationFlags::INTER_CUBIC);
	imshow("Image", img);
	cv::waitKey(0);
}