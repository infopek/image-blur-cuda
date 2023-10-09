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

/// <summary>
/// Blurs an image using box blur.
/// </summary>
/// <param name="image">The image you want to blur.</param>
/// <param name="kernelRadius">A kernel with size (2 * kernelRadius + 1) will be used for blurring.</param>
void boxBlur(cv::Mat& image, int kernelRadius)
{
	std::vector<cv::Mat> channels;
	cv::split(image, channels);

	BoxBlur blurrer(image.cols, image.rows);
	blurrer.blur(channels[2].data, channels[2].data, kernelRadius);	// B 
	blurrer.blur(channels[1].data, channels[1].data, kernelRadius);	// G
	blurrer.blur(channels[0].data, channels[0].data, kernelRadius);	// R 

	cv::merge(channels, image);
}

void medianFilter(cv::Mat& image)
{
	std::vector<cv::Mat> channels;
	cv::split(image, channels);

	MedianFilter filterer(image.cols, image.rows);
	filterer.filter(channels[2].data, channels[2].data);	// B
	filterer.filter(channels[1].data, channels[1].data);	// G
	filterer.filter(channels[0].data, channels[0].data);	// R

	cv::merge(channels, image);
}

int main(int argc, char* argv[])
{
	fs::path dir("res");
	fs::path file("nature.jpg");
	fs::path fullPath = dir / file;

	cv::Mat image = cv::imread(fullPath.string());

	show(image);

	boxBlur(image, 3);

	show(image);
}
