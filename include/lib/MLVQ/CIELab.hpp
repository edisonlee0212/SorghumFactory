/*!**************************************************************************
\file    CIELab.h
\author  Jiri Filip and Vlastimil Havran
\date    13/11/2006
\version 0.00

  The header file for the:
	Colour spaces conversions (RGB,XYZ,CIE Lab1976,Lab2000)

******************************************************************************/

#ifndef CIELAB_c
#define CIELAB_c

#include <SharCoors.hpp>

//extern float GammaVal;
//extern int HDRflag;

// This is the color model proposed in the technical report
// Backward Compatible High Dynamic Range MPEG Compression
// MPI-I-2005, December 2005, by Rafal Mantiuk, Alexander Efremov
// and Karol Myszkowski
int Rgb2LogLuv(const float rgb[], float luv[]);
int LogLuv2Rgb(const float luv[], float rgb[]);
int LogLuv2RgbNormalized(const float luv[], float rgb[]);

/*! RGB to YCbCr space */
void RgBtoYCbCr(const float rgb[], float yCbCr[]);

/*! YCbCr to RGB space */
void YCbCrToRgb(const float yCbCr[], float rgb[]);

void YCbCrToRgbNormalized(const float yCbCr[], float rgb[]);

// Let use only one color space for both HDR/LDR BTF data, which is YCrCb
//#define ONLY_ONE_COLOR_SPACE

// This is user defined model for BTF Compression
inline void UserCmToRgb(const float userColorModelData[],
	float rgb[],
	const TSharedCoordinates& tc) {
#ifndef ONLY_ONE_COLOR_SPACE
	if (tc.m_hdrFlag) {
		if (tc.m_codeBtfFlag)
			LogLuv2RgbNormalized(userColorModelData, rgb);
		else
			LogLuv2Rgb(userColorModelData, rgb);
	}
	else
#endif
		if (tc.m_codeBtfFlag) {
			YCbCrToRgbNormalized(userColorModelData, rgb);
		}
		else {
			YCbCrToRgb(userColorModelData, rgb);
		}
}

inline void RgbToUserCm(const float rgb[], float userColorModelData[],
	const bool hdrFlag)
{
#ifndef ONLY_ONE_COLOR_SPACE
	if (hdrFlag)
		Rgb2LogLuv(rgb, userColorModelData);
	else
#endif
		RgBtoYCbCr(rgb, userColorModelData);

	return;
}

inline void UserCmToRgb(const float userColorModelData[], float rgb[],
                        const float multi, const TSharedCoordinates& tc)
{
	// #ifndef ONLY_ONE_COLOR_SPACE
	//   if (HDRflag)
	//     LogLuv2RGB(UserColorModelData, RGB);
	//   else
	// #endif    
	//     YCbCrtoRGB(UserColorModelData, RGB);
	UserCmToRgb(userColorModelData, rgb, tc);

	const float multi2 = 1.0f / multi;
	for (int i = 0; i < 3; i++) rgb[i] *= multi2;
}

inline void RgbToUserCm(const float rgb[], float userColorModelData[],
                        const float multi, const bool HdrFlag)
{
	float rgb2[3];
	for (int i = 0; i < 3; i++) rgb2[i] = rgb[i] * multi;
#ifndef ONLY_ONE_COLOR_SPACE
	if (HdrFlag)
		Rgb2LogLuv(rgb2, userColorModelData);
	else
#endif
		RgBtoYCbCr(rgb2, userColorModelData);
}

#endif // CIELAB_c

