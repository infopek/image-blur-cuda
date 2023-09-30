#include <iostream>

#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../cuda/blur.h"

void show(const cv::Mat& image)
{
	cv::imshow("Image", image);
	cv::waitKey(0);
}

void blurCPU(unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelSize)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int sum{};
			int numPixels{};

			for (int r = -kernelSize / 2; r <= kernelSize / 2; ++r)
			{
				for (int c = -kernelSize / 2; c <= kernelSize / 2; ++c)
				{
					int row = y + r;
					int col = x + c;

					if (row >= 0 && row < height && col >= 0 && col < width)
					{
						sum += src[row * width + col];
						++numPixels;
					}
				}
			}

			dst[y * width + x] = (unsigned char)(sum / (kernelSize * kernelSize));
		}
	}
}

int main()
{
	cv::Mat image = cv::imread("./res/nature.jpg");
	const int width = image.cols;
	const int height = image.rows;

	std::vector<cv::Mat> channels;
	cv::split(image, channels);

	Blur blurrer(width, height);
	int kernelSize = 9;

	blurrer.blur(channels[2].data, channels[2].data, kernelSize);
	blurrer.blur(channels[1].data, channels[1].data, kernelSize);
	blurrer.blur(channels[0].data, channels[0].data, kernelSize);

	/*blurCPU(channels[2].data, channels[2].data, width, height, kernelSize);
	blurCPU(channels[1].data, channels[1].data, width, height, kernelSize);
	blurCPU(channels[0].data, channels[0].data, width, height, kernelSize);*/

	cv::merge(channels, image);

	show(image);
}
