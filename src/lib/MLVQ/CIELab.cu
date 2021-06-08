/*!**************************************************************************
\file    CIELab.cpp
\author  Jiri Filip
\date    13/11/2006
\version 0.00

  The header file for the:
	Colour spaces conversions (RGB,XYZ,CIE Lab1976,Lab2000)

******************************************************************************/

#include <cstdlib>
#include <cassert>
#include <iostream>

// This is the color model proposed in the technical report
// Backward Compatible High Dynamic Range MPEG Compression
// MPI-I-2005, December 2005, by Rafal Mantiuk, Alexander Efremov
// and Karol Myszkowski.
int Rgb2LogLuv(const float rgb[], float luv[])
{
	/*! \brief Conversion from RGB colour space into
	  LogLu'v' (not L*u*v*!) according to [Ward98]
	  for Observer. = 2degree, Illuminant = D65 */

	float rgbCopy[3];

	// Let us linearize the correction by gamma before processing
	// - we know that values are roughly in visible range <0-255>
	// as they were multiplied by user selected coefficient
	rgbCopy[0] = rgb[0];
	rgbCopy[1] = rgb[1];
	rgbCopy[2] = rgb[2];

	if (rgbCopy[0] > 0.04045f) rgbCopy[0] = powf((rgbCopy[0] + 0.055f) / 1.055f, 2.4f);
	else              rgbCopy[0] /= 12.92f;
	if (rgbCopy[1] > 0.04045f) rgbCopy[1] = powf((rgbCopy[1] + 0.055f) / 1.055f, 2.4f);
	else              rgbCopy[1] /= 12.92f;
	if (rgbCopy[2] > 0.04045f) rgbCopy[2] = powf((rgbCopy[2] + 0.055f) / 1.055f, 2.4f);
	else              rgbCopy[2] /= 12.92f;

	//Observer. = 2degree, Illuminant = D65
	// RGB -> XYZ
	float X = rgbCopy[0] * 0.4124f + rgbCopy[1] * 0.3576f + rgbCopy[2] * 0.1805f;
	float Y = rgbCopy[0] * 0.2126f + rgbCopy[1] * 0.7152f + rgbCopy[2] * 0.0722f; // luminance
	float Z = rgbCopy[0] * 0.0193f + rgbCopy[1] * 0.1192f + rgbCopy[2] * 0.9505f;

	// For LDR images
	// luma (weighted sum of the non-linear RGB components after gamma
	// correction has been applied) is:
	// float luma = RGB[0] * 0.2126f + RGB[1] * 0.7152f + RGB[2] * 0.0722f;

	// HDR value
	// For HDR images, using 11-12 bits of visible range of luminance
	// we encode to Luma this way:
	// XYZ -> LogLuv, according to the technical report by Rafal Mantiuk, 2006
	if (Y < 5.6046f)
		luv[0] = 17.554f * Y;
	else
		if (Y < 10469.f)
			luv[0] = 826.81f * powf(Y, 0.10013f) - 884.17f;
		else
			luv[0] = 209.f * logf(Y) - 731.28f;

	// only for debugging values
#if 0
	static float Lmax = 0.f;
	if (Luv[0] > Lmax) {
		Lmax = Luv[0];
		printf("Lmax = %3.3f ", Lmax);
	}
#endif

	float den = (X + 15.f * Y + 3.f * Z);
	if (den > 1e-30f) {
		luv[1] = 410.f * (4.f * X / den); // this can be encoded in 8 bits
		luv[2] = 410.f * (9.f * Y / den); // this can be encoded in 8 bits
	}
	else {
		// Basically black colour, i.e. rgb[0]=rgb[1]=rgb[2]=epsilon
		luv[1] = 410.f * 4.f * 0.9502f / 19.2172f;
		luv[2] = 410.f * 9.f / 19.2172f;
	}

	return 1;
}//--- RGB2LogLuv -------------------------------------------

int LogLuv2Rgb(const float luv[], float rgb[])
{
	/*! \brief Conversion from LogLu'v' (not L*u*v*)  colour space according to [Ward98] into
	  RGB colourspace for Observer. = 2degree, Illuminant = D65 */

	  // LogLuv -> XYZ

	float Y;

	// HDR
	// This is according to the technical report of Rafal Mantiuk, 2006
	if (luv[0] < 98.381f)
		Y = 0.056968f * luv[0];
	else
		if (luv[0] < 1204.7f)
			Y = 7.3014e-30f * powf(luv[0] + 884.17f, 9.9872f);
		else
			Y = 32.994f * expf(0.0047811f * luv[0]);

	float X, Z;
	if (luv[2] > 0.f) {
		X = 9.0f / 4.0f * luv[1] / luv[2] * Y;
		Z = Y * (3.0f * 410.f / luv[2] - 5.0f) - X / 3.0f;
	}
	else {
		assert(luv[1] == 0.f);
		X = Z = 0.f;
	}

	// XYZ -> RGB
	rgb[0] = X * 3.2406f + Y * -1.5372f + Z * -0.4986f;
	rgb[1] = X * -0.9689f + Y * 1.8758f + Z * 0.0415f;
	rgb[2] = X * 0.0557f + Y * -0.2040f + Z * 1.0570f;

	if (rgb[0] < 0.f) rgb[0] = 0.f;
	if (rgb[1] < 0.f) rgb[1] = 0.f;
	if (rgb[2] < 0.f) rgb[2] = 0.f;

	// 11/01/2009 - in fact, this should not be here - but it seems
	// that all UBO HDR images are gamma corrected  - Vlastimil
	if (rgb[0] > 0.0031308f) rgb[0] = 1.055f * (float)powf((double)rgb[0], 1.f / 2.4f) - 0.055f;
	else               rgb[0] = 12.92f * rgb[0];
	if (rgb[1] > 0.0031308f) rgb[1] = 1.055f * (float)powf((double)rgb[1], 1.f / 2.4f) - 0.055f;
	else               rgb[1] = 12.92f * rgb[1];
	if (rgb[2] > 0.0031308f) rgb[2] = 1.055f * (float)powf((double)rgb[2], 1.f / 2.4f) - 0.055f;
	else               rgb[2] = 12.92f * rgb[2];

	return 1;
}//--- LogLuv2RGB -------------------------------------------


int LogLuv2RgbNormalized(const float luv[], float rgb[])
{
	/*! \brief Conversion from LogLu'v' (not L*u*v*)  colour space according to [Ward98] into
	  RGB colourspace for Observer. = 2degree, Illuminant = D65 */

	  // LogLuv -> XYZ

	float Y;

	// HDR
	// This is according to the technical report of Rafal Mantiuk, 2006
	if (luv[0] < 98.381f)
		Y = 0.056968f * luv[0];
	else
		if (luv[0] < 1204.7f)
			Y = 7.3014e-30f * powf(luv[0] + 884.17f, 9.9872f);
		else
			Y = 32.994f * expf(0.0047811f * luv[0]);

	float X, Z;
	if (luv[2] > 0.f) {
		X = 9.0f / 4.0f * luv[1] / luv[2] * Y;
		Z = Y * (3.0f * 410.f / luv[2] - 5.0f) - X / 3.0f;
	}
	else {
		assert(luv[1] == 0.f);
		X = Z = 0.f;
	}

	X /= 256.0f;
	Y /= 256.0f;
	Z /= 256.0f;

	// XYZ -> RGB
	rgb[0] = X * 3.2406f + Y * -1.5372f + Z * -0.4986f;
	rgb[1] = X * -0.9689f + Y * 1.8758f + Z * 0.0415f;
	rgb[2] = X * 0.0557f + Y * -0.2040f + Z * 1.0570f;

	if (rgb[0] < 0.f) rgb[0] = 0.f;
	if (rgb[1] < 0.f) rgb[1] = 0.f;
	if (rgb[2] < 0.f) rgb[2] = 0.f;

	// 11/01/2009 - in fact, this should not be here - but it seems
	// that all UBO HDR images are gamma corrected  - Vlastimil
	if (rgb[0] > 0.0031308f) rgb[0] = 1.055f * (float)powf((double)rgb[0], 1.f / 2.4f) - 0.055f;
	else               rgb[0] = 12.92f * rgb[0];
	if (rgb[1] > 0.0031308f) rgb[1] = 1.055f * (float)powf((double)rgb[1], 1.f / 2.4f) - 0.055f;
	else               rgb[1] = 12.92f * rgb[1];
	if (rgb[2] > 0.0031308f) rgb[2] = 1.055f * (float)powf((double)rgb[2], 1.f / 2.4f) - 0.055f;
	else               rgb[2] = 12.92f * rgb[2];

	return 1;
}//--- LogLuv2RGB -------------------------------------------

/*! RGB to YCbCr space */
void RgBtoYCbCr(const float rgb[], float yCbCr[])
{
	assert(rgb[0] >= 0.f);
	assert(rgb[1] >= 0.f);
	assert(rgb[2] >= 0.f);

	// This is for Kb=0.114 and Kr=0.299, when the data
	// are given with 8-bit digital precision. It is according to ITU-R BT.601

	// Y
	yCbCr[0] = 16.f + rgb[0] * 0.25690625f + rgb[1] * 0.50412891f +
		rgb[2] * 0.09790625f;
	// Cb
	yCbCr[1] = 128.f - rgb[0] * 0.14822266f - rgb[1] * 0.29099219f +
		rgb[2] * 0.43921484f;
	// Cr
	yCbCr[2] = 128.f + rgb[0] * 0.43921484f - rgb[1] * 0.36778906f -
		rgb[2] * 0.07142578f;
	if (yCbCr[0] < 0.f) yCbCr[0] = 0.f;
	if (yCbCr[1] < 0.f) yCbCr[1] = 0.f;
	if (yCbCr[2] < 0.f) yCbCr[2] = 0.f;

	return;
}

/*! YCbCr to RGB space */
void YCbCrToRgb(const float yCbCr[], float rgb[])
{
	assert(yCbCr[0] >= 0.f);
	assert(yCbCr[1] >= 0.f);
	assert(yCbCr[2] >= 0.f);

	// R
	rgb[0] = yCbCr[0] * 1.1643828f + yCbCr[2] * 1.5960273f - 222.921f;
	// G
	rgb[1] = yCbCr[0] * 1.1643828f - yCbCr[1] * 0.39176172f -
		yCbCr[2] * 0.81296875f + 135.576f;
	// B
	rgb[2] = yCbCr[0] * 1.1643828f + yCbCr[1] * 2.0172344f - 276.836f;


	if (rgb[0] < 0.f) rgb[0] = 0.f;
	if (rgb[1] < 0.f) rgb[1] = 0.f;
	if (rgb[2] < 0.f) rgb[2] = 0.f;

	return;

}

/*! YCbCr to RGB space */
void YCbCrToRgbNormalized(const float yCbCr[], float rgb[])
{
	assert(yCbCr[0] >= 0.f);
	assert(yCbCr[1] >= 0.f);
	assert(yCbCr[2] >= 0.f);

	// R
	rgb[0] = yCbCr[0] * 1.1643828f + yCbCr[2] * 1.5960273f - 222.921f / 256.0f;
	// G
	rgb[1] = yCbCr[0] * 1.1643828f - yCbCr[1] * 0.39176172f -
		yCbCr[2] * 0.81296875f + 135.576f / 256.0f;
	// B
	rgb[2] = yCbCr[0] * 1.1643828f + yCbCr[1] * 2.0172344f - 276.836f / 256.0f;

	if (rgb[0] < 0.f) rgb[0] = 0.f;
	if (rgb[1] < 0.f) rgb[1] = 0.f;
	if (rgb[2] < 0.f) rgb[2] = 0.f;

	return;
}
