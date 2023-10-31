#include "blur.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../utils/cuda_utils.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>

#include <algorithm>
#include <cmath>
#include <numbers>

__device__ unsigned char* dev_src;
__device__ unsigned char* dev_dst;
__device__ float* dev_weights;

extern __shared__ unsigned char sh_data[];

__global__ void blurImageGauss(const unsigned char* src, unsigned char* dst, float* weights, size_t width, size_t height, int tileSize, int kernelRadius, float sigma)
{
	int diameter = 2 * kernelRadius + 1;

	// Shared data
	unsigned char* sharedImageData = sh_data;
	float* sharedWeights = (float*)&sharedImageData[blockDim.x * blockDim.y];

	// Calculate CUDA indices
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

__global__ void blurImageBox(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelRadius, int tileSize)
{
	// Get current position of thread in image
	int x = blockIdx.x * tileSize + threadIdx.x - kernelRadius;
	int y = blockIdx.y * tileSize + threadIdx.y - kernelRadius;

	// Clamp position to edge of image
	x = fminf(fmaxf(0, x), width - 1);
	y = fminf(fmaxf(0, y), height - 1);

	// Calculate global and (block) local index
	unsigned int idx = y * width + x;
	unsigned int blockIdx = blockDim.x * threadIdx.y + threadIdx.x;

	// Each thread in a block copies a pixel from src to shared memory
	sh_data[blockIdx] = src[idx];
	__syncthreads();

	if (threadIdx.x >= kernelRadius && threadIdx.y >= kernelRadius && threadIdx.x < (blockDim.x - kernelRadius) && threadIdx.y < (blockDim.y - kernelRadius))
	{
		// Use average of kernel to apply blur effect
		float sum = 0.0f;
		for (int r = -kernelRadius; r <= kernelRadius; ++r)
			for (int c = -kernelRadius; c <= kernelRadius; ++c)
				sum += (float)sh_data[blockIdx + (r * blockDim.x) + c];

		int diameter = 2 * kernelRadius + 1;
		dst[idx] = (unsigned char)(sum / (diameter * diameter));
	}
}

void Blur::boxBlur(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelRadius)
{
	// Init CUDA memory
	size_t imageSize = width * height * sizeof(unsigned char);

	CUDA_CALL(cudaMalloc(&dev_src, imageSize));
	CUDA_CALL(cudaMalloc(&dev_dst, imageSize));

	// Copy source buffer to device buffer
	CUDA_CALL(cudaMemcpy(dev_src, src, imageSize, cudaMemcpyHostToDevice));

	// Kernel launch dimensions & parameters
	dim3 blockSize(32, 32);
	int tileSize = blockSize.x - 2 * kernelRadius;

	dim3 gridSize(width / tileSize + 1,
		height / tileSize + 1);

	size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char);

	// Launch kernel
	blurImageBox KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_src, dev_dst, width, height, kernelRadius, tileSize);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy output to dst
	CUDA_CALL(cudaMemcpy(dst, dev_dst, imageSize, cudaMemcpyDeviceToHost));

	// Cleanup
	CUDA_CALL(cudaFree(dev_src));
	CUDA_CALL(cudaFree(dev_dst));
}

void Blur::gaussBlur(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelRadius, float sigma)
{
	// Fill Gauss kernel
	int diameter = 2 * kernelRadius + 1;
	float* weights = new float[diameter * diameter];
	calculateGaussKernel(weights, kernelRadius, sigma);

	// Init CUDA memory
	size_t imageSize = width * height * sizeof(unsigned char);
	size_t kernelSize = diameter * diameter * sizeof(float);

	CUDA_CALL(cudaMalloc(&dev_src, imageSize));
	CUDA_CALL(cudaMalloc(&dev_dst, imageSize));
	CUDA_CALL(cudaMalloc(&dev_weights, kernelSize));

	CUDA_CALL(cudaMemcpy(dev_src, src, imageSize, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_weights, weights, kernelSize, cudaMemcpyHostToDevice));

	// Kernel launch dimensions & parameters
	dim3 blockSize(32, 32);
	int tileSize = blockSize.x - 2 * kernelRadius;

	dim3 gridSize(width / tileSize + 1,
		height / tileSize + 1);

	size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char) + kernelSize;

	// Launch kernel
	blurImageGauss KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_src, dev_dst, dev_weights, width, height, tileSize, kernelRadius, sigma);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy output to dst
	CUDA_CALL(cudaMemcpy(dst, dev_dst, imageSize, cudaMemcpyDeviceToHost));

	// Cleanup
	CUDA_CALL(cudaFree(dev_src));
	CUDA_CALL(cudaFree(dev_dst));
	CUDA_CALL(cudaFree(dev_weights));
	delete[] weights;
}

float Blur::gaussFunction(float x, float y, float sigma)
{
	float halfInvSigmaSqr = 0.5f / (sigma * sigma);

	float denom = 0.3183098f * halfInvSigmaSqr;	// 1 / (2 * pi * sigma ^ 2)
	float exponent = -(x * x + y * y) * (halfInvSigmaSqr);	// - (x ^ 2 + y ^ 2) / (2 * sigma ^ 2)
	return denom * expf(exponent);	// 1 / (2 * pi * sigma^2) * e ^ [-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)]
}

void Blur::calculateGaussKernel(float* gaussKernel, int kernelRadius, float sigma)
{
	int diameter = 2 * kernelRadius + 1;

	float sum = 0.0f;
	for (int y = -kernelRadius; y <= kernelRadius; ++y)
	{
		for (int x = -kernelRadius; x <= kernelRadius; ++x)
		{
			int weightIdx = (y + kernelRadius) * diameter + (x + kernelRadius);
			gaussKernel[weightIdx] = gaussFunction(static_cast<float>(x), static_cast<float>(y), sigma);
			sum += gaussKernel[weightIdx];
		}
	}

	std::for_each(gaussKernel, gaussKernel + diameter * diameter,
		[=](float& f) {
			f /= sum;
		});
}
