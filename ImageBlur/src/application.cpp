#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../cuda/blur.h"

#include <iostream>
#include <filesystem>
#include <numbers>

namespace fs = std::filesystem;

void show(const cv::Mat& image)
{
	cv::imshow("Image", image);
	cv::waitKey(0);
}

void boxBlur(const cv::Mat& src, cv::Mat& dst, int kernelRadius)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	Blur::boxBlur(channels[2].data, channels[2].data, channels[2].cols, channels[2].rows, kernelRadius);	// B
	Blur::boxBlur(channels[1].data, channels[1].data, channels[1].cols, channels[1].rows, kernelRadius);	// G
	Blur::boxBlur(channels[0].data, channels[0].data, channels[0].cols, channels[0].rows, kernelRadius);	// R

	cv::merge(channels, dst);
}

void gaussBlur(const cv::Mat& src, cv::Mat& dst, int kernelRadius, float sigma)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	Blur::gaussBlur(channels[2].data, channels[2].data, channels[2].cols, channels[2].rows, kernelRadius, sigma);	// B
	Blur::gaussBlur(channels[1].data, channels[1].data, channels[1].cols, channels[1].rows, kernelRadius, sigma);	// G
	Blur::gaussBlur(channels[0].data, channels[0].data, channels[0].cols, channels[0].rows, kernelRadius, sigma);	// R

	cv::merge(channels, dst);
}

int main(int argc, char* argv[])
{
	fs::path dir("res");
	fs::path file("nature.jpg");
	fs::path fullPath = dir / file;

	cv::Mat image = cv::imread(fullPath.string());

	show(image);

	float sigma = 1.5f;
	int kernelRadius = 3;
	gaussBlur(image, image, kernelRadius, sigma);

	show(image);
}
