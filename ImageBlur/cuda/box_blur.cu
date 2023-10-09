#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "box_blur.h"
#include "../utils/cuda_utils.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>

__device__ unsigned char* dev_srcBox;
__device__ unsigned char* dev_dstBox;

__global__ void blurImage(unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelSize)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	/*__shared__ unsigned char kernel[9 * 9];
	kernel[blockDim.x * threadIdx.y + threadIdx.x] = src[y * width + x];*/

	//__syncthreads();

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
	/*if (threadIdx.x < blockDim.x && threadIdx.y < blockDim.y)
	{
		for (int r = 0; r < blockDim.y; r++)
		{
			for (int c = 0; c < blockDim.x; c++)
			{
				int row = threadIdx.y + r;
				int col = threadIdx.x + c;
				sum += (int)kernel[row * blockDim.x + col];
			}
		}
	}*/
	//sum /= kernelSize * kernelSize;
	dst[y * width + x] = (unsigned char)(sum / numPixels);
}

BoxBlur::BoxBlur(size_t width, size_t height)
	: m_width{ width }, m_height{ height }
{
	init();
}

BoxBlur::~BoxBlur()
{
	shutdown();
}

void BoxBlur::init()
{
	size_t size = m_width * m_height;
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_srcBox, size));
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_dstBox, size));
}

void BoxBlur::shutdown()
{
	CUDA_CALL(cudaFree(dev_dstBox));
	CUDA_CALL(cudaFree(dev_srcBox));
}

void BoxBlur::blur(unsigned char* src, unsigned char* dst, int kernelSize)
{
	size_t size = m_width * m_height * sizeof(unsigned char);

	CUDA_CALL(cudaMemcpy(dev_srcBox, src, size, cudaMemcpyHostToDevice));

	dim3 blockSize(32, 32);

	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

	blurImage KERNEL_ARGS2(gridSize, blockSize)(dev_srcBox, dev_dstBox, m_width, m_height, kernelSize);

	CUDA_CALL(cudaMemcpy(dst, dev_dstBox, size, cudaMemcpyDeviceToHost));
}
