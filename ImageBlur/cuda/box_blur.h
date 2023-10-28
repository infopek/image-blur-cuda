#pragma once

/// <summary>
/// Represents a class which can blur an image using box blur.
/// </summary>
class BoxBlur
{
public:
	BoxBlur(size_t width, size_t height);
	~BoxBlur();

	void blur(const unsigned char* src, unsigned char* dst, int kernelRadius);

private:
	void init();

	void shutdown();

private:
	size_t m_width{};
	size_t m_height{};
};