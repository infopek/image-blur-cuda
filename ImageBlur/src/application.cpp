#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../cuda/box_blur.h"
#include "../cuda/median_filter.h"
#include "../cuda/gauss_blur.h"

#include <iostream>
#include <filesystem>
#include <numbers>

namespace fs = std::filesystem;

void show(const cv::Mat& image)
{
	cv::imshow("Image", image);
	cv::waitKey(0);
}

/// <summary>
/// Blurs an image using box blur.
/// </summary>
/// <param name="src">The image you want to blur.</param>
/// <param name="dst">The blurred image.</param>
/// <param name="kernelRadius">A kernel with size (2 * kernelRadius + 1) will be used for blurring.</param>
void boxBlur(const cv::Mat& src, cv::Mat& dst, int kernelRadius)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	BoxBlur blurrer(src.cols, src.rows);
	blurrer.blur(channels[2].data, channels[2].data, kernelRadius);	// B 
	blurrer.blur(channels[1].data, channels[1].data, kernelRadius);	// G
	blurrer.blur(channels[0].data, channels[0].data, kernelRadius);	// R 

	cv::merge(channels, dst);
}

void medianFilter(const cv::Mat& src, cv::Mat& dst)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	MedianFilter filterer(src.cols, src.rows);
	filterer.filter(channels[2].data, channels[2].data);	// B
	filterer.filter(channels[1].data, channels[1].data);	// G
	filterer.filter(channels[0].data, channels[0].data);	// R

	cv::merge(channels, dst);
}

void gaussBlur(const cv::Mat& src, cv::Mat& dst, float sigma, int kernelRadius)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	GaussBlur blurrer(src.cols, src.rows);
	blurrer.blur(channels[2].data, channels[2].data, sigma, kernelRadius);	// B
	blurrer.blur(channels[1].data, channels[1].data, sigma, kernelRadius);	// G
	blurrer.blur(channels[0].data, channels[0].data, sigma, kernelRadius);	// R

	cv::merge(channels, dst);
}

int main(int argc, char* argv[])
{
	fs::path dir("res");
	fs::path file("lena.png");
	fs::path fullPath = dir / file;

	cv::Mat image = cv::imread(fullPath.string());

	show(image);

	float sigma = 1.5f;
	int kernelRadius = 5;
	gaussBlur(image, image, sigma, kernelRadius);

	show(image);
}
