#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gauss_blur.h"
#include "../utils/cuda_utils.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cmath>
#include <device_functions.h>
#include <numbers>

__device__ unsigned char* dev_srcGauss;
__device__ unsigned char* dev_dstGauss;


__device__ float gaussFunction(float x, float y, float sigma)
{
	float invSigmaSqr = 1.0f / (sigma * sigma);

	float denom = 0.1591549f * invSigmaSqr;	// 1 / (2 * pi * sigma ^ 2)
	float exponent = -(x * x + y * y) * (0.5f * invSigmaSqr);	// - (x ^ 2 + y ^ 2) / (2 * sigma ^ 2)
	return denom * expf(exponent);	// 1 / (2 * pi * sigma^2) * e ^ [-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)]
}

__constant__ float weights[25]{
		1, 4, 7, 4, 1,
		4, 16, 26, 16, 4,
		7, 26, 41, 26, 7,
		4, 16, 26, 16, 4,
		1, 4, 7, 4, 1
};

__global__ void blurImageGauss(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int tileSize, int kernelRadius, float sigma)
{
	extern __shared__ unsigned char sh_kernel[];

	//float* sharedWeights = (float*)&sharedImageData[blockDim.x * blockDim.y];

	int x = blockIdx.x * tileSize + threadIdx.x - kernelRadius;
	int y = blockIdx.y * tileSize + threadIdx.y - kernelRadius;

	// Clamp position to edge of image
	x = fminf(fmaxf(0, x), width - 1);
	y = fminf(fmaxf(0, y), height - 1);

	// Calculate indices
	unsigned int idx = y * width + x;
	unsigned int blockIdx = blockDim.x * threadIdx.y + threadIdx.x;

	// Setup weights matrix using gauss function

	// Normalize weights

	// Each thread in a block copies a pixel to shared block from src image
	sh_kernel[blockIdx] = src[idx];
	__syncthreads();

	// Each pixel val in shared block is weighted

	// Apply gauss blur
	if (threadIdx.x >= kernelRadius && threadIdx.y >= kernelRadius && threadIdx.x < (blockDim.x - kernelRadius) && threadIdx.y < (blockDim.y - kernelRadius))
	{
		int finalSum = 0;
		for (int r = -kernelRadius; r <= kernelRadius; ++r)
			for (int c = -kernelRadius; c <= kernelRadius; ++c)
				finalSum += sh_kernel[blockIdx + (r * blockDim.x) + c] * weights[(r + kernelRadius) * 5 + (c + kernelRadius)];

		dst[idx] = finalSum / 273;
	}
}

GaussBlur::GaussBlur(size_t width, size_t height)
	: m_width{ width }, m_height{ height }
{
	init();
}

GaussBlur::~GaussBlur()
{
	shutdown();
}

void GaussBlur::init()
{
	size_t size = m_width * m_height;
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_srcGauss, size));
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_dstGauss, size));
}

void GaussBlur::shutdown()
{
	CUDA_CALL(cudaFree(dev_dstGauss));
	CUDA_CALL(cudaFree(dev_srcGauss));
}

void GaussBlur::blur(const unsigned char* src, unsigned char* dst, float sigma, int kernelRadius)
{
	// Copy source buffer to device buffer
	size_t size = m_width * m_height * sizeof(unsigned char);
	CUDA_CALL(cudaMemcpy(dev_srcGauss, src, size, cudaMemcpyHostToDevice));

	// Set up kernel launch dimensions
	dim3 blockSize(16, 16);
	int tileSize = blockSize.x - 2 * kernelRadius;
	dim3 gridSize(m_width / tileSize + 1,
		m_height / tileSize + 1);

	int diameter = 2 * kernelRadius + 1;
	size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char);

	// Launch kernel
	blurImageGauss KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_srcGauss, dev_dstGauss, m_width, m_height, tileSize, kernelRadius, sigma);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy blurred image to destination
	CUDA_CALL(cudaMemcpy(dst, dev_dstGauss, size, cudaMemcpyDeviceToHost));
}
