#pragma once

/// <summary>
/// Represents a class which can blur an image using gauss blur.
/// </summary>
class GaussBlur
{
public:
	GaussBlur(size_t width, size_t height);
	~GaussBlur();

	void blur(const unsigned char* src, unsigned char* dst, float sigma, int kernelRadius);

private:
	void init();

	void shutdown();

private:
	size_t m_width{};
	size_t m_height{};
};
