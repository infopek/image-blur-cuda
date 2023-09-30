#include "blur.h"

#include <stdio.h>
#include <iostream>

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define CUDA_CALL( call )               \
{                                       \
	cudaError_t result = call;              \
	if ( cudaSuccess != result )            \
		std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

__device__ unsigned char* dev_src;
__device__ unsigned char* dev_dst;

__global__ void blurImage(unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelSize)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

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

	dst[y * width + x] = (unsigned char)(sum / numPixels);
}

Blur::Blur(size_t imgWidth, size_t imgHeight)
	: m_imgWidth{ imgWidth }, m_imgHeight{ imgHeight }
{
	size_t size = imgWidth * imgHeight;
	init(size);
}

Blur::~Blur()
{
	shutdown();
}

void Blur::init(size_t size)
{
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_src, size));
	CUDA_CALL(cudaMalloc<unsigned char>(&dev_dst, size));
}

void Blur::shutdown()
{
	CUDA_CALL(cudaFree(dev_dst));
	CUDA_CALL(cudaFree(dev_src));
}

void Blur::blur(unsigned char* src, unsigned char* dst, int kernelSize)
{
	size_t size = m_imgWidth * m_imgHeight * sizeof(unsigned char);

	CUDA_CALL(cudaMemcpy(dev_src, src, size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_dst, dst, size, cudaMemcpyHostToDevice));

	dim3 blockSize(32, 32);
	dim3 gridSize((m_imgWidth + blockSize.x - 1) / blockSize.x, (m_imgHeight + blockSize.y - 1) / blockSize.y);

	blurImage KERNEL_ARGS2(gridSize, blockSize)(dev_src, dev_dst, m_imgWidth, m_imgHeight, kernelSize);

	CUDA_CALL(cudaDeviceSynchronize());

	CUDA_CALL(cudaMemcpy(dst, dev_dst, size, cudaMemcpyDeviceToHost));
}
