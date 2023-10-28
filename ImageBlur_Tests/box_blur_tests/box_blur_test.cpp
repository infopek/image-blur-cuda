#include "pch.h"
#include "../../ImageBlur/cuda/box_blur.h"

constexpr int width = 4;
constexpr int height = 4;

struct BoxBlurTest : testing::Test
{
	BoxBlurTest()
	{
		blurrer = new BoxBlur(width, height);
	}

	~BoxBlurTest()
	{
		delete blurrer;
	}

	BoxBlur* blurrer;
};

TEST_F(BoxBlurTest, BoxBlurBlursImage)
{
	unsigned char* expected = new unsigned char[width * height] {
		12, 17, 26, 18,
			16, 26, 43, 31,
			18, 28, 45, 31,
			17, 25, 35, 21
		};

	unsigned char* actual = new unsigned char[width * height] {
			30, 70, 30, 50,
			10, 00, 20, 70,
			00, 40, 40, 70,
			30, 90, 30, 50,
		};
	int kernelRadius = 1;

	blurrer->blur(actual, actual, kernelRadius);

	EXPECT_EQ(expected, actual);
}