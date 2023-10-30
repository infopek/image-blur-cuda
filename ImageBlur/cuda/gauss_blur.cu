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
__device__ float* dev_weights;

float gaussFunction(float x, float y, float sigma)
{
	float invSigmaSqr = 1.0f / (sigma * sigma);

	float denom = 0.1591549f * invSigmaSqr;	// 1 / (2 * pi * sigma ^ 2)
	float exponent = -(x * x + y * y) * (0.5f * invSigmaSqr);	// - (x ^ 2 + y ^ 2) / (2 * sigma ^ 2)
	return denom * expf(exponent);	// 1 / (2 * pi * sigma^2) * e ^ [-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)]
}

void calculateWeights(float* weights, float sigma, int kernelRadius)
{
	// Fill weights using Gauss function
	int diameter = 2 * kernelRadius + 1;
	float sum = 0.0f;
	for (int y = -kernelRadius; y <= kernelRadius; ++y)
	{
		for (int x = -kernelRadius; x <= kernelRadius; ++x)
		{
			int weightIdx = (y + kernelRadius) * diameter + (x + kernelRadius);
			weights[weightIdx] = gaussFunction(static_cast<float>(x), static_cast<float>(y), sigma);
			sum += weights[weightIdx];
		}
	}

	// Normalize weights
	for (int i = 0; i < diameter; i++)
		for (int j = 0; j < diameter; j++)
			weights[i * diameter + j] /= sum;
}

__global__ void blurImageGauss(const unsigned char* src, unsigned char* dst, float* weights, size_t width, size_t height, int tileSize, int kernelRadius, float sigma)
{
	int diameter = 2 * kernelRadius + 1;

	// Shared data
	extern __shared__ unsigned char sh_kernel[];
	unsigned char* sharedImageData = sh_kernel;
	float* sharedWeights = (float*)&sharedImageData[blockDim.x * blockDim.y];

	// Calculate global indices
	int x = blockIdx.x * tileSize + threadIdx.x - kernelRadius;
	int y = blockIdx.y * tileSize + threadIdx.y - kernelRadius;

	// Clamp position to edge of image
	x = fminf(fmaxf(0, x), width - 1);
	y = fminf(fmaxf(0, y), height - 1);

	// Calculate indices
	unsigned int idx = y * width + x;
	unsigned int blockIdx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int weightIdx = threadIdx.y * diameter + threadIdx.x;

	// Each thread in a block copies a pixel to shared block from src image
	sharedImageData[blockIdx] = src[idx];
	__syncthreads();

	// Some threads also copy weights
	if (weightIdx < diameter * diameter)
		sharedWeights[weightIdx] = weights[weightIdx];

	// Apply gauss blur
	if (threadIdx.x >= kernelRadius && threadIdx.y >= kernelRadius && threadIdx.x < (blockDim.x - kernelRadius) && threadIdx.y < (blockDim.y - kernelRadius))
	{
		float sum = 0.0f;
		for (int r = -kernelRadius; r <= kernelRadius; ++r)
			for (int c = -kernelRadius; c <= kernelRadius; ++c)
				sum += (float)sharedImageData[blockIdx + (r * blockDim.x) + c] * sharedWeights[(r + kernelRadius) * diameter + (c + kernelRadius)];

		dst[idx] = (unsigned char)sum;
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
	CUDA_CALL(cudaMalloc(&dev_srcGauss, size));
	CUDA_CALL(cudaMalloc(&dev_dstGauss, size));
}

void GaussBlur::shutdown()
{
	CUDA_CALL(cudaFree(dev_dstGauss));
	CUDA_CALL(cudaFree(dev_srcGauss));
}

void GaussBlur::blur(const unsigned char* src, unsigned char* dst, float sigma, int kernelRadius)
{
	// Calculate weights
	int diameter = 2 * kernelRadius + 1;
	float* weights = new float[diameter * diameter];
	calculateWeights(weights, sigma, kernelRadius);

	CUDA_CALL(cudaMalloc(&dev_weights, diameter * diameter * sizeof(float)));
	CUDA_CALL(cudaMemcpy(dev_weights, weights, diameter * diameter * sizeof(float), cudaMemcpyHostToDevice));

	// Copy source buffer to device buffer
	size_t size = m_width * m_height * sizeof(unsigned char);
	CUDA_CALL(cudaMemcpy(dev_srcGauss, src, size, cudaMemcpyHostToDevice));

	// Set up kernel launch dimensions
	dim3 blockSize(32, 32);
	int tileSize = blockSize.x - 2 * kernelRadius;
	dim3 gridSize(m_width / tileSize + 1,
		m_height / tileSize + 1);

	size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char) + diameter * diameter * sizeof(float);

	// Launch kernel
	blurImageGauss KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_srcGauss, dev_dstGauss, dev_weights, m_width, m_height, tileSize, kernelRadius, sigma);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy blurred image to destination
	CUDA_CALL(cudaMemcpy(dst, dev_dstGauss, size, cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(dev_weights));
	delete[] weights;
}
