#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "median_filter.h"
#include "../utils/cuda_utils.h"

__device__ unsigned char* dev_srcMedian;
__device__ unsigned char* dev_dstMedian;

/// <summary>
/// Swaps 'a' and 'b' if 'b' is less than 'a'.
/// </summary>

__device__ void sortMask(unsigned char* mask, int length)
{
	for (int i = 0; i < length; i++)
	{
		for (int j = i + 1; j < length; j++)
		{
			if (mask[j] < mask[i])
			{
				unsigned char tmp = mask[i];
				mask[i] = mask[j];
				mask[j] = tmp;
			}
		}
	}
}

__global__ void filterImage(unsigned char* src, unsigned char* dst, size_t width, size_t height)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int maskSize = 3;
	const int length = maskSize * maskSize;

	unsigned char* mask = new unsigned char[length];

	for (int r = 0; r < maskSize; r++)
	{
		for (int c = 0; c < maskSize; c++)
		{
			int row = y + r - 1;
			int col = x + c - 1;

			if (row >= 0 && col >= 0 && row < height && col < width)
				mask[r * maskSize + c] = src[row * width + col];
		}
	}

	sortMask(mask, length);

	unsigned char median = mask[length / 2];
	dst[y * width + x] = median;

	delete[] mask;
}

MedianFilter::MedianFilter(size_t width, size_t height)
	: m_width{ width }, m_height{ height }
{
	init();
}

MedianFilter::~MedianFilter()
{
	shutdown();
}

void MedianFilter::filter(unsigned char* src, unsigned char* dst)
{
	size_t size = m_width * m_height * sizeof(unsigned char);

	CUDA_CALL(cudaMemcpy(dev_srcMedian, src, size, cudaMemcpyHostToDevice));

	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);
	filterImage KERNEL_ARGS2(gridSize, blockSize)(dev_srcMedian, dev_dstMedian, m_width, m_height);

	CUDA_CALL(cudaMemcpy(dst, dev_dstMedian, size, cudaMemcpyDeviceToHost));
}

void MedianFilter::init()
{
	size_t size = m_width * m_height;
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_srcMedian, size));
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_dstMedian, size));
}

void MedianFilter::shutdown()
{
	CUDA_CALL(cudaFree(dev_srcMedian));
	CUDA_CALL(cudaFree(dev_dstMedian));
}