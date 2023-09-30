#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Blur
{
public:
	Blur(size_t imgWidth, size_t imgHeight);
	~Blur();

	void blur(unsigned char* src, unsigned char* dst, int kernelSize);

private:
	void init(size_t size);

	void shutdown();

private:
	size_t m_imgWidth{};
	size_t m_imgHeight{};
};
