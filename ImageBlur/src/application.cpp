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

float gauss(float x, float y, float sigma)
{
	constexpr float inv2Pi = 0.1591549f; // 1 / (2 * pi)
	float invSigmaSqr = 1.0f / (sigma * sigma);

	float denom = inv2Pi * invSigmaSqr;	// 1 / (2 * pi * sigma ^ 2)
	float exponent = -(x * x + y * y) * (0.5f * invSigmaSqr);	// - (x ^ 2 + y ^ 2) / (2 * sigma ^ 2)
	return denom * std::exp(exponent);	// 1 / (2 * pi * sigma^2) * e ^ [-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)]
}

void gb(unsigned char* src, unsigned char* dst, size_t width, size_t height, float sigma, int kernelRadius)
{
	int diameter = 2 * kernelRadius + 1;

	// Setup weights matrix using gauss function
	float* weights = new float[diameter * diameter] {};
	float weightSum = 0.0f;
	for (int y = 0; y < diameter; y++)
	{
		for (int x = 0; x < diameter; x++)
		{
			int offsetY = y - kernelRadius;
			int offsetX = x - kernelRadius;
			weights[y * diameter + x] = gauss(offsetX, offsetY, sigma);
			weightSum += weights[y * diameter + x];
		}
	}

	// Normalize weights matrix (divide by sum of all elements)
	for (int y = 0; y < diameter; y++)
		for (int x = 0; x < diameter; x++)
			weights[y * diameter + x] /= weightSum;

	// Loop through whole image
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			// Indices
			int globalIdx = y * width + x;

			// Setup weighted pixels matrix around (x, y)
			float* weightedPixels = new float[diameter * diameter] {};
			float finalSum = 0.0f;
			for (int w_y = 0; w_y < diameter; w_y++)
			{
				for (int w_x = 0; w_x < diameter; w_x++)
				{
					int offsetY = w_y - kernelRadius;
					int offsetX = w_x - kernelRadius;
					int globalOffsetIdx = globalIdx + offsetY * diameter + offsetX;
					int weightIdx = w_y * diameter + w_x;

					if (globalOffsetIdx >= 0 && globalOffsetIdx < width * height)
					{
						weightedPixels[weightIdx] = weights[weightIdx] * src[globalOffsetIdx];
						finalSum += weightedPixels[weightIdx];
					}
				}
			}

			// Center pixel is the sum of all elements in normalized gauss kernel
			dst[globalIdx] = (unsigned char)finalSum;
		}
	}
}

void gaussBlurCpu(const cv::Mat& src, cv::Mat& dst, float sigma, int kernelRadius)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	gb(channels[2].data, channels[2].data, src.cols, src.rows, sigma, kernelRadius);	// B
	gb(channels[1].data, channels[1].data, src.cols, src.rows, sigma, kernelRadius);	// G
	gb(channels[0].data, channels[0].data, src.cols, src.rows, sigma, kernelRadius);	// R

	cv::merge(channels, dst);
}

void gaussBlur(const cv::Mat& src, cv::Mat& dst, float sigma, int kernelRadius)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	/*gb(channels[2].data, channels[2].data, src.cols, src.rows, sigma, kernelRadius);
	gb(channels[1].data, channels[1].data, src.cols, src.rows, sigma, kernelRadius);
	gb(channels[0].data, channels[0].data, src.cols, src.rows, sigma, kernelRadius);*/
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
	int kernelRadius = 2;
	gaussBlur(image, image, sigma, kernelRadius);

	show(image);
}
