
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>

#include "constants.h"
#include "display.h"
#include "random.h"

#define CUDA_CALL( call )               \
{                                       \
	cudaError_t result = call;              \
	if ( cudaSuccess != result )            \
		std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}

using namespace Constants;

constexpr int screenSize = height * width;

unsigned short screen[height][width]{};
unsigned short blurredScreen[height][width]{};

__device__ unsigned short dev_screen[height][width]{};
__device__ unsigned short dev_blurredScreen[height][width]{};

void initScreen()
{
	// Fill screen with random numbers
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			screen[y][x] = static_cast<unsigned short>(Random::get(0, 255));
}

void initScreenCUDA()
{
	CUDA_CALL(cudaMemcpyToSymbol(dev_screen, screen, sizeof(screen)));
	CUDA_CALL(cudaMemcpyToSymbol(dev_blurredScreen, blurredScreen, sizeof(blurredScreen)));
}

void blurCPU()
{
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			blurredScreen[y][x] = (
				screen[y - 1][x]
				+ screen[y][x - 1]
				+ screen[y][x]
				+ screen[y + 1][x]
				+ screen[y][x + 1]) / 5;
		}
	}
}

__global__ void blurGPUSingle()
{
	for (int y = 1; y < height - 1; y++)
	{
		for (int x = 1; x < width - 1; x++)
		{
			dev_blurredScreen[y][x] = (
				dev_screen[y - 1][x]
				+ dev_screen[y][x - 1]
				+ dev_screen[y][x]
				+ dev_screen[y + 1][x]
				+ dev_screen[y][x + 1]) / 5;
		}
	}
}

__global__ void blurGPUWxH()
{
	if (threadIdx.x > 0 && threadIdx.y > 0 && threadIdx.x < width && threadIdx.y < height)
	{
		dev_blurredScreen[threadIdx.y][threadIdx.x] = (
			dev_screen[threadIdx.y - 1][threadIdx.x]
			+ dev_screen[threadIdx.y][threadIdx.x - 1]
			+ dev_screen[threadIdx.y][threadIdx.x]
			+ dev_screen[threadIdx.y + 1][threadIdx.x]
			+ dev_screen[threadIdx.y][threadIdx.x + 1]) / 5;
	}
}

void callBlurCPU()
{
	blurCPU();
}

void callBlurGPUSingle()
{
	blurGPUSingle << < 1, 1 >> > ();

	CUDA_CALL(cudaMemcpyFromSymbol(blurredScreen, dev_blurredScreen, sizeof(blurredScreen)));
}

void callBlurGPUWxH()
{
	blurGPUWxH << < 1, dim3(width, height) >> > ();

	CUDA_CALL(cudaMemcpyFromSymbol(blurredScreen, dev_blurredScreen, sizeof(blurredScreen)));
}

int main()
{
	initScreen();
	initScreenCUDA();

	callBlurCPU();

	showImage(blurredScreen);
}
