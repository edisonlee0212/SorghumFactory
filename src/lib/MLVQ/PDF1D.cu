/*!**************************************************************************
\file    PDF1D.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  1D PDF database
******************************************************************************/

#include <cassert>
#include <cmath>
#include <cstdio>
#include <PDF1D.hpp>
#include <TAlloc.hpp>
#include <CIELab.hpp>
#include <AuxFuncs.hpp>

//#########################################################################
//######## CPDF1D #####################################################JF##
//#########################################################################

CPDF1D::CPDF1D(int maxPDF1D, int LengthOfSlice, int metric, int maxShift)
{
	assert(maxPDF1D > 0);
	assert(LengthOfSlice > 0);

	this->m_maxPdf1D = maxPDF1D;
	this->m_lengthOfSlice = LengthOfSlice;
	this->m_numOfPdf1D = 0;
	m_pdf1DBasis = Allocation2(0, maxPDF1D - 1, 0, LengthOfSlice - 1);
	assert(m_pdf1DBasis);
}//--- CPDF1D ----------------------------------------------------------

CPDF1D::~CPDF1D()
{
	if (m_pdf1DBasis != NULL)
		Free2(m_pdf1DBasis, 0, m_maxPdf1D - 1, 0, m_lengthOfSlice - 1);
}//--- ~CPDF1D ---------------------------------------------------------

void
CPDF1D::DeleteData()
{
	if (m_pdf1DBasis != NULL)
		Free2(m_pdf1DBasis, 0, m_maxPdf1D - 1, 0, m_lengthOfSlice - 1);

	m_maxPdf1D = 1;
	m_pdf1DBasis = Allocation2(0, m_maxPdf1D - 1, 0, m_lengthOfSlice - 1);
	assert(m_pdf1DBasis);
	m_numOfPdf1D = 0;
} // --------------------------------------------------------------------

// check & reallocation of database if needed
void
CPDF1D::Reallocate()
{
	if (m_numOfPdf1D >= m_maxPdf1D) {
		int newMaxPDF1D = Max(m_maxPdf1D + 10000, m_numOfPdf1D);
		m_pdf1DBasis = Reallocation2Rows(m_pdf1DBasis, 0,
			m_maxPdf1D - 1, newMaxPDF1D - 1,
			0, m_lengthOfSlice - 1);
		m_maxPdf1D = newMaxPDF1D;
	}
}//--- realloc -----------------------------------------

float
CPDF1D::NormalPdf1D(const float* const lums)
{
	float normPar = 0.f;

	// just sum of values
	for (int t = 0; t < m_lengthOfSlice; t++)
		normPar += lums[t];

	// This makes normal value to be non-zero !
	if (normPar < 1e-6)
		normPar = 1e-6;

	return normPar;
}//--- normalPDF1D ----------------------------------------------------

float
CPDF1D::GetVal(int sliceIndex, float beta, TSharedCoordinates& tc) const
{
	assert((sliceIndex >= 0) && (sliceIndex < m_numOfPdf1D));
	assert((beta >= -90.f) && (beta <= 90.f));

	int i;
	float w;
	tc.ComputeIndexForAngleBetaDeg(beta, i, w);
	//  printf("%d %f %f\n",i,beta/stepBeta);
	assert((i >= 0) && (i <= m_lengthOfSlice - 1));
	assert((w >= 0.f) && (w <= 1.f));

	// Define one of the three possible methods for univariate interpolation
	//#define LINEAR_INTERPOLANT
#define HERMITE_INTERPOLANT
//#define AKIMA_INTERPOLANT

#ifdef LINEAR_INTERPOLANT
  // This implements simple linear interpolation between two values
	return (1.0f - w) * PDF1Dbasis[sliceIndex][i] + w * PDF1Dbasis[sliceIndex][i + 1];
#endif
#ifdef HERMITE_INTERPOLANT
	// This implements Fergusson cubic interpolation based on Cubic Hermite Splines
	float p0 = m_pdf1DBasis[sliceIndex][i];
	float p1 = m_pdf1DBasis[sliceIndex][i + 1];
	float m0h, m1h;
	if (i == 0)
		m0h = p1 - p0; // end point
	else
		// standard way
		m0h = 0.5f * (p1 - m_pdf1DBasis[sliceIndex][i - 1]);

	if (i == m_lengthOfSlice - 2)
		m1h = p1 - p0; // end point
	else
		// standard way
		m1h = 0.5f * (m_pdf1DBasis[sliceIndex][i + 1] - p0);
	float t2 = w * w;
	float t3 = t2 * w;
	float h01 = -2.0f * t3 + 3.0f * t2;
	float h00 = 1.0f - h01;
	float h11 = t3 - t2;
	float h10 = h11 - t2 + w;

	// This implements the whole formula
	float res = h00 * p0 + h10 * m0h + h01 * p1 + h11 * m1h;
	return res;
#endif // HERMITE_INTERPOLANT
#ifdef  AKIMA_INTERPOLANT
	// This implements Akima interpolation method published at:
	// H. Akima, ``A New Method of Interpolation and Smooth Curve Fitting Based
	// on Local Procedures,'' J.ACM, vol. 17, no. 4, pp. 589-602, 1970.
	float a0 = PDF1Dbasis[sliceIndex][i];
	float y1 = PDF1Dbasis[sliceIndex][i + 1];
	float t10 = (tc.betaAngles[i + 1] - tc.betaAngles[i]);
	float x = w;

	float d0h, d1h;
	if (tc.iBeta == 0)
		d0h = (y1 - a0); // end point
	else
		// standard way
		d0h = 0.5f * (y1 - PDF1Dbasis[sliceIndex][i - 1]);

	assert(i < LengthOfSlice - 1);
	if (i == LengthOfSlice - 2)
		d1h = (y1 - a0); // end point
	else
		// standard way
		d1h = 0.5f * (PDF1Dbasis[sliceIndex][i + 1] - a0);

	float mi = (y1 - a0);
	float a2 = -(2.0f * d0h - 3.0f * mi + d1h);
	float a3 = (d0h + d1h - mi - mi);
	// This implements the whole formula of Akima univariate interpolation
	float res = a0 + x * (d0h + x * (a2 + x * a3));
	return res;
#endif // AKIMA_INTERPOLANT

}//--- getVal --------------------------------------------------------

float
CPDF1D::GetVal(int sliceIndex, TSharedCoordinates& tc) const
{
	assert((sliceIndex >= 0) && (sliceIndex < m_numOfPdf1D));
#ifdef LINEAR_INTERPOLANT
	// This implements simple linear interpolation between two values
	return (1.f - tc.wBeta) * PDF1Dbasis[sliceIndex][tc.iBeta] +
		tc.wBeta * PDF1Dbasis[sliceIndex][tc.iBeta + 1];
#endif

#ifdef HERMITE_INTERPOLANT
	// This implements Fergusson cubic interpolation based on Cubic Hermite Splines
	float w = tc.m_wBeta;
	float p0 = m_pdf1DBasis[sliceIndex][tc.m_iBeta];
	float p1 = m_pdf1DBasis[sliceIndex][tc.m_iBeta + 1];
	float m0h, m1h;
	if (tc.m_iBeta == 0)
		m0h = p1 - p0; // end point
	else
		// standard way
		m0h = 0.5f * (p1 - m_pdf1DBasis[sliceIndex][tc.m_iBeta - 1]);

	assert(tc.m_iBeta < m_lengthOfSlice - 1);
	if (tc.m_iBeta == m_lengthOfSlice - 2)
		m1h = p1 - p0; // end point
	else
		// standard way
		m1h = 0.5f * (m_pdf1DBasis[sliceIndex][tc.m_iBeta + 1] - p0);
	float t2 = w * w;
	float t3 = t2 * w;
	float h01 = -2.0f * t3 + 3.0f * t2;
	float h00 = 1.0f - h01;
	float h11 = t3 - t2;
	float h10 = h11 - t2 + w;

	// This implements the whole formula
	float res = h00 * p0 + h10 * m0h + h01 * p1 + h11 * m1h;
	return res;
#endif // HERMITE_INTERPOLANT
#ifdef AKIMA_INTERPOLANT
	// This implements Akima interpolation method published at:
	// H. Akima, ``A New Method of Interpolation and Smooth Curve Fitting Based
	// on Local Procedures,'' J.ACM, vol. 17, no. 4, pp. 589-602, 1970.
	float a0 = PDF1Dbasis[sliceIndex][tc.iBeta];
	float y1 = PDF1Dbasis[sliceIndex][tc.iBeta + 1];
	float t10 = (tc.betaAngles[tc.iBeta + 1] - tc.betaAngles[tc.iBeta]);
	float x = tc.wBeta;

	float d0h, d1h;
	if (tc.iBeta == 0)
		d0h = (y1 - a0); // end point
	else
		// standard way
		d0h = 0.5f * (y1 - PDF1Dbasis[sliceIndex][tc.iBeta - 1]);

	assert(tc.iBeta < LengthOfSlice - 1);
	if (tc.iBeta == LengthOfSlice - 2)
		d1h = (y1 - a0); // end point
	else
		// standard way
		d1h = 0.5f * (PDF1Dbasis[sliceIndex][tc.iBeta + 1] - a0);

	float mi = (y1 - a0);
	float a2 = -(2.0f * d0h - 3.0f * mi + d1h);
	float a3 = (d0h + d1h - mi - mi);
	// This implements the whole formula of Akima univariate interpolation
	float res = a0 + x * (d0h + x * (a2 + x * a3));
	return res;
#endif // AKIMA_INTERPOLANT

}//--- getVal --------------------------------------------------------

void
CPDF1D::GetValShepard(int sliceIndex, float scale, float sumDist2,
	float userCmData[], TSharedCoordinates& tc) const
{
	int iMaxDist = (int)(sqrt(tc.m_maxDist2 - sumDist2));
	const float w = tc.m_wBeta;
	const int i = tc.m_iBeta;
	float RGB[3];
	for (int ii = i - iMaxDist; (ii <= i); ii++) {
		if (ii >= 0) {
			float distPart2 = Square(w + (float)ii - (float)i);
			if (distPart2 + sumDist2 < tc.m_maxDist2) {
				userCmData[0] = scale * m_pdf1DBasis[sliceIndex][i];
				// Convert from user color model to RGB
				UserCmToRgb(userCmData, RGB, tc);

				float dist2 = distPart2 + sumDist2;
				float weight = Square(tc.m_maxDist - sqrt(dist2)) / (dist2 * tc.m_maxDist2);
				tc.m_rgb[0] += RGB[0] * weight;
				tc.m_rgb[1] += RGB[1] * weight;
				tc.m_rgb[2] += RGB[2] * weight;
				tc.m_sumWeight += weight;
				tc.m_countWeight++;
			} // if
		} // if ii
	} // for

	for (int ii = i + 1; (ii < m_lengthOfSlice) && (ii <= i + 1 + iMaxDist); ii++) {
		// The distance along 
		float distPart2 = Square((float)ii - (float)i - w);
		if (distPart2 + sumDist2 < tc.m_maxDist2) {
			// there is a chance of having grid point value in the distance smaller than specified

			userCmData[0] = scale * m_pdf1DBasis[sliceIndex][i];
			// Convert to RGB
			UserCmToRgb(userCmData, RGB, tc);

			float dist2 = distPart2 + sumDist2;
			float weight = Square(tc.m_maxDist - sqrt(dist2)) / (dist2 * tc.m_maxDist2);
			tc.m_rgb[0] += RGB[0] * weight;
			tc.m_rgb[1] += RGB[1] * weight;
			tc.m_rgb[2] += RGB[2] * weight;
			tc.m_sumWeight += weight;
			tc.m_countWeight++;
		} // if
	} // for

	return;
}

int
CPDF1D::GetNumOfPdf1D() const
{
	return m_numOfPdf1D;
}//--- getNoOfPDF1D ---------------------------------------------------

int
CPDF1D::GetMemory() const
{
	return m_numOfPdf1D * m_lengthOfSlice * sizeof(float); //+NoOfPDF1D*sizeof(float *);
}//--- getMemory ---------------------------------------------------

int
CPDF1D::GetMemoryQ() const
{
	// The second array is not necessary for efficient implemention
	// Clearly, this could be quantized according to minima and maxima
	// of PDF values
	return m_numOfPdf1D * m_lengthOfSlice * sizeof(float);
}//--- getMemory ---------------------------------------------------


int
CPDF1D::GetSliceLength() const
{
	return m_lengthOfSlice;
}//--- getSliceLength ------------------------------------------------

int
CPDF1D::Load(char* prefix, int mlf, int algPut)
{
	assert(prefix);
	assert((mlf == 0) || (mlf == 1));
	char fileName[1000];

	// loading data from TXT file
	sprintf(fileName, "%s_PDF1Dslice.txt", prefix);
	int nr, nc;
	float minV, maxV;
	ReadTxtHeader(fileName, &nr, &nc, &minV, &maxV);
	assert(nc == m_lengthOfSlice);
	m_numOfPdf1D = nr;
	Reallocate();

	float** tmpArr = Allocation2(0, nr - 1, 0, nc - 1);
	ReadTxt(tmpArr, fileName, &nr, &nc);

	for (int irow = 0; irow < m_numOfPdf1D; irow++) {
		for (int jcol = 0; jcol < m_lengthOfSlice; jcol++)
			m_pdf1DBasis[irow][jcol] = tmpArr[irow][jcol];
	} // ------------
	Free2(tmpArr, 0, nr - 1, 0, nc - 1);

	return 0; // ok
}//--- load ----------------------------------------------------------

