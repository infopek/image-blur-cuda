#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "median_filter.h"
#include "../utils/cuda_utils.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>

__device__ unsigned char* dev_srcMedian;
__device__ unsigned char* dev_dstMedian;

constexpr int maskDim = 3;
constexpr int maskLength = maskDim * maskDim;
extern __shared__ unsigned char sh_mask[];

__global__ void filterImage(unsigned char* src, unsigned char* dst, size_t width, size_t height)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Fill mask with corresponding image pixels
	sh_mask[threadIdx.y * blockDim.x + threadIdx.x] = src[y * width + x];
	__syncthreads();

	// The first thread sorts the mask using insertion sort
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int i = 1;
		while (i < maskLength)
		{
			int x = sh_mask[i];
			int j = i - 1;
			while (j >= 0 && sh_mask[j] > x)
			{
				sh_mask[j + 1] = sh_mask[j];
				--j;
			}
			sh_mask[j + 1] = x;
			++i;
		}
	}
	__syncthreads();

	dst[y * width + x] = sh_mask[maskLength / 2];	// maskSize / 2 => median of mask
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

	dim3 blockSize(maskDim, maskDim);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);
	size_t sharedMemSize = maskLength * sizeof(unsigned char);
	filterImage KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_srcMedian, dev_dstMedian, m_width, m_height);

	CUDA_CALL(cudaDeviceSynchronize());

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