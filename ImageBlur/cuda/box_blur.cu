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

extern __shared__ unsigned char sh_kernel[];

__global__ void blurImage(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelRadius, int tileSize)
{
	// Get current position of thread in image
	int x = blockIdx.x * tileSize + threadIdx.x - kernelRadius;
	int y = blockIdx.y * tileSize + threadIdx.y - kernelRadius;

	// Clamp position to edge of image
	x = fmaxf(0, x);
	x = fminf(x, width - 1);
	y = fmaxf(y, 0);
	y = fminf(y, height - 1);

	// Calculate global and (block) local index
	unsigned int idx = y * width + x;
	unsigned int blockIdx = blockDim.x * threadIdx.y + threadIdx.x;

	// Each thread in a block copies a pixel to shared block from src image
	sh_kernel[blockIdx] = src[idx];
	__syncthreads();

	if (threadIdx.x >= kernelRadius && threadIdx.y >= kernelRadius && threadIdx.x < (blockDim.x - kernelRadius) && threadIdx.y < (blockDim.y - kernelRadius))
	{
		// Use average of kernel to apply blur effect
		float sum{};
		for (int r = -kernelRadius; r <= kernelRadius; ++r)
			for (int c = -kernelRadius; c <= kernelRadius; ++c)
				sum += (float)sh_kernel[blockIdx + (r * blockDim.x) + c];

		unsigned int diameter = 2 * kernelRadius + 1;
		dst[idx] = sum / (diameter * diameter);
		}
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

void BoxBlur::blur(const unsigned char* src, unsigned char* dst, int kernelRadius)
{
	// Copy source buffer to device buffer
	size_t size = m_width * m_height * sizeof(unsigned char);
	CUDA_CALL(cudaMemcpy(dev_srcBox, src, size, cudaMemcpyHostToDevice));

	// Set up kernel launch dimensions
	dim3 blockSize(32, 32);
	int tileSize = blockSize.x - 2 * kernelRadius;
	dim3 gridSize(m_width / tileSize + 1,
		m_height / tileSize + 1);

	int sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char);

	// Launch kernel
	CUDA_CALL(cudaDeviceSynchronize());
	blurImage KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_srcBox, dev_dstBox, m_width, m_height, kernelRadius, tileSize);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy blurred image to destination
	CUDA_CALL(cudaMemcpy(dst, dev_dstBox, size, cudaMemcpyDeviceToHost));
}
