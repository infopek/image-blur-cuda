#pragma once

/// <summary>
/// Represents a class which can blur an image using box blur.
/// </summary>
class BoxBlur
{
public:
	BoxBlur(size_t imgWidth, size_t imgHeight);
	~BoxBlur();

	void blur(unsigned char* src, unsigned char* dst, int kernelSize);

private:
	void init();

	void shutdown();

private:
	size_t m_width{};
	size_t m_height{};
};
