#pragma once

#include <chrono>
#include <random>

namespace Random
{
	inline std::mt19937 init()
	{
		std::random_device rd;

		// Create seed_seq with high-res clock and 7 random numbers from std::random_device
		std::seed_seq ss{
			static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count()),
			rd(), rd(), rd(), rd(), rd(), rd(), rd() };

		return std::mt19937{ ss };
	}

	// PRNG object
	std::mt19937 mt{ init() };

	// Generate a random number between [min, max] (inclusive)
	inline int get(int min, int max)
	{
		std::uniform_int_distribution<int> dist{ min, max };
		return dist(mt);
	}
}