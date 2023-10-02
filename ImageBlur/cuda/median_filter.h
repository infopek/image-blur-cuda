#pragma once

/// <summary>
/// Represents a class which can blur an image using median blur.
/// </summary>
class MedianFilter
{
public:
	MedianFilter(size_t width, size_t height);
	~MedianFilter();

	void filter(unsigned char* src, unsigned char* dst);

private:
	void init();

	void shutdown();

private:
	size_t m_width{};
	size_t m_height{};
};