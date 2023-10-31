#pragma once

/// <summary>
/// Static class containing different kinds of blurs.
/// </summary>
class Blur
{
public:
	Blur() = delete;
	Blur(const Blur&) = delete;
	Blur(Blur&&) noexcept = delete;
	~Blur() = delete;

	/// <summary>
	/// Blurs src using box blur with a kernel size of (2 * kernelRadius + 1), saves output to dst.
	/// </summary>
	/// <param name="width">Image width.</param>
	/// <param name="height">Image height.</param>
	static void boxBlur(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelRadius);

	/// <summary>
	/// Blurs src using Gauss blur with a kernel size of (2 * kernelRadius + 1) and sigma_x = sigma_y = sigma, saves output to dst.
	/// </summary>
	/// <param name="width">Image width.</param>
	/// <param name="height">Image height.</param>
	static void gaussBlur(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelRadius, float sigma);

private:
	/// <summary>
	/// Calculates the Gaussian function value at (x, y) with sigma_x = sigma_y = sigma.
	/// </summary>
	/// <returns>
	///		1 / (2 * pi * sigma ^ 2) * exp(-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)).
	/// </returns>
	static float gaussFunction(float x, float y, float sigma);

	/// <summary>
	/// Fills gaussKernel with values calculated via Gaussian function.
	/// </summary>
	static void calculateGaussKernel(float* gaussKernel, int kernelRadius, float sigma);
};
