#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../cuda/box_blur.h"
#include "../cuda/median_filter.h"

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void show(const cv::Mat& image)
{
	cv::imshow("Image", image);
	cv::waitKey(0);
}

void boxBlur(cv::Mat& image)
{
	BoxBlur blurrer(image.cols, image.rows);
	int kernelSize = 6;

	std::vector<cv::Mat> channels;
	cv::split(image, channels);

	blurrer.blur(channels[2].data, channels[2].data, kernelSize);
	blurrer.blur(channels[1].data, channels[1].data, kernelSize);
	blurrer.blur(channels[0].data, channels[0].data, kernelSize);

	cv::merge(channels, image);
}

void medianFilter(cv::Mat& image)
{
	MedianFilter filterer(image.cols, image.rows);

	std::vector<cv::Mat> channels;
	cv::split(image, channels);

	filterer.filter(channels[2].data, channels[2].data);
	filterer.filter(channels[1].data, channels[1].data);
	filterer.filter(channels[0].data, channels[0].data);

	cv::merge(channels, image);
}

int main(int argc, char* argv[])
{
	fs::path dir("res");
	fs::path file("nature.jpg");
	fs::path fullPath = dir / file;

	cv::Mat image = cv::imread(fullPath.string());

	//show(image);
	
	//medianFilter(image);
	boxBlur(image);

	//cv::medianBlur(image, image, 3);
	//cv::blur(image, image, cv::Size(5, 5));

	show(image);
}
