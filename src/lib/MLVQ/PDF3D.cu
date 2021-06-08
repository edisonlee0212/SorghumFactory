/*!**************************************************************************
\file    PDF3D.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  3D PDF indices database
******************************************************************************/

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using namespace std;

#include <AuxFuncs.hpp>
#include <TAlloc.hpp>
#include <PDF1D.hpp>
#include <IndexAB.hpp>
#include <PDF2D.hpp>
#include <PDF3D.hpp>

//#########################################################################
//######## CPDF3D #####################################################JF##
//#########################################################################

CPDF3D::CPDF3D(int maxPDF3D, int SlicesPerThetaValue, CPDF1D* PDF1,
	CIndexAB* IAB, CPDF2D* PDF2, int metric)
{
	assert(maxPDF3D > 0);
	assert(PDF1);
	assert(IAB);
	assert(PDF2);
	assert(metric >= 0);

	this->m_maxPdf3D = maxPDF3D;
	this->m_pdf1 = PDF1;
	this->m_iab = IAB;
	this->m_pdf2 = PDF2;

	// By slices we mean the number of discrete values in which we sample BTF
	this->m_slicesPerTheta = SlicesPerThetaValue;
	this->m_stepTheta = 90.f / (float)(m_slicesPerTheta - 1);
	this->m_numOfPdf3D = 0;
	int size2D = PDF2->GetSlicesPerHemisphere() * PDF1->GetSliceLength();
	this->m_size3D = m_slicesPerTheta * size2D;

	m_pdf3Dslices = IAllocation2(0, maxPDF3D - 1, 0, m_slicesPerTheta - 1);
	assert(m_pdf3Dslices);
	m_pdf3Dscale = Allocation2(0, maxPDF3D - 1, 0, m_slicesPerTheta - 1);
	assert(m_pdf3Dscale);
}//--- CPDF3D ---------------------------------------------------------

CPDF3D::~CPDF3D()
{
	if (m_pdf3Dslices != NULL)
		IFree2(m_pdf3Dslices, 0, m_maxPdf3D - 1, 0, m_slicesPerTheta - 1);
	if (m_pdf3Dscale != NULL)
		Free2(m_pdf3Dscale, 0, m_maxPdf3D - 1, 0, m_slicesPerTheta - 1);
}//--- CPDF3D ---------------------------------------------------------

void
CPDF3D::DeleteData()
{
	if (m_pdf3Dslices != NULL)
		IFree2(m_pdf3Dslices, 0, m_maxPdf3D - 1, 0, m_slicesPerTheta - 1);
	if (m_pdf3Dscale != NULL)
		Free2(m_pdf3Dscale, 0, m_maxPdf3D - 1, 0, m_slicesPerTheta - 1);

	this->m_maxPdf3D = 1;
	this->m_numOfPdf3D = 0;

	m_pdf3Dslices = IAllocation2(0, m_maxPdf3D - 1, 0, m_slicesPerTheta - 1);
	assert(m_pdf3Dslices);
	m_pdf3Dscale = Allocation2(0, m_maxPdf3D - 1, 0, m_slicesPerTheta - 1);
	assert(m_pdf3Dscale);

	return;
}//--- CPDF3D ---------------------------------------------------------

void
CPDF3D::Reallocate()
{
	// reallocation of database if needed
	if (m_numOfPdf3D >= m_maxPdf3D)
	{
		int newMaxPDF3D = Max(m_maxPdf3D + 1000, m_numOfPdf3D);
		m_pdf3Dslices = IReallocation2Rows(m_pdf3Dslices, 0,
			m_maxPdf3D - 1, newMaxPDF3D - 1,
			0, m_slicesPerTheta - 1);
		m_pdf3Dscale = Reallocation2Rows(m_pdf3Dscale, 0,
			m_maxPdf3D - 1, newMaxPDF3D - 1,
			0, m_slicesPerTheta - 1);
		m_maxPdf3D = newMaxPDF3D;
	}
}//--- realloc -------------------------------------------------------

void
CPDF3D::GetVal(int pdf3DIndex, float viewTheta, float alpha, float beta, float rgb[],
	TSharedCoordinates& tc) const
{
	assert((pdf3DIndex >= 0) && (pdf3DIndex < m_numOfPdf3D));
	assert((beta >= -90.f) && (beta <= 90.f));
	assert((alpha >= -90.f) && (alpha <= 90.f));
	assert((viewTheta >= 0.f) && (viewTheta <= 90.f));

	int i = (int)floor(viewTheta / m_stepTheta);
	if (i == m_slicesPerTheta)
		i = m_slicesPerTheta - 1;
	assert((i >= 0) && (i < m_slicesPerTheta));

	if (i < (m_slicesPerTheta - 1)) {
		float w = (viewTheta - i * m_stepTheta) / m_stepTheta;
		assert((w >= 0.f) && (w <= 1.f));
		// interpolation between two values retrieved from PDF2D
		float RGB2[3];
		m_pdf2->GetVal(m_pdf3Dslices[pdf3DIndex][i], alpha, beta, rgb, tc);
		m_pdf2->GetVal(m_pdf3Dslices[pdf3DIndex][i + 1], alpha, beta, RGB2, tc);
		float s1 = m_pdf3Dscale[pdf3DIndex][i] * (1.f - w);
		float s2 = m_pdf3Dscale[pdf3DIndex][i + 1] * w;
		rgb[0] = rgb[0] * s1 + RGB2[0] * s2;
		rgb[1] = rgb[1] * s1 + RGB2[1] * s2;
		rgb[2] = rgb[2] * s1 + RGB2[2] * s2;
	}
	else {
		m_pdf2->GetVal(m_pdf3Dslices[pdf3DIndex][i], alpha, beta, rgb, tc);
		float s = m_pdf3Dscale[pdf3DIndex][i];
		rgb[0] *= s;
		rgb[1] *= s;
		rgb[2] *= s;
	}

	return;
}//--- getVal --------------------------------------------------------

void
CPDF3D::GetVal(const int pdf3DIndex, float rgb[], TSharedCoordinates& tc) const
{
	assert((pdf3DIndex >= 0) && (pdf3DIndex < m_numOfPdf3D));
	const int i = tc.m_iTheta;
	if (i < m_slicesPerTheta - 1) {
		const float w = tc.m_wTheta;
		// interpolation between two values retrieved from PDF2D
		float rgb2[3];
		m_pdf2->GetVal(m_pdf3Dslices[pdf3DIndex][i], rgb, tc);
		m_pdf2->GetVal(m_pdf3Dslices[pdf3DIndex][i + 1], rgb2, tc);
		const float s1 = m_pdf3Dscale[pdf3DIndex][i] * (1.f - w);
		const float s2 = m_pdf3Dscale[pdf3DIndex][i + 1] * w;
		rgb[0] = rgb[0] * s1 + rgb2[0] * s2;
		rgb[1] = rgb[1] * s1 + rgb2[1] * s2;
		rgb[2] = rgb[2] * s1 + rgb2[2] * s2;
	}
	else {
		m_pdf2->GetVal(m_pdf3Dslices[pdf3DIndex][i], rgb, tc);
		const float s = m_pdf3Dscale[pdf3DIndex][i];
		rgb[0] *= s;
		rgb[1] *= s;
		rgb[2] *= s;
	}

	return;
}//--- getVal --------------------------------------------------------

void
CPDF3D::GetValShepard(int pdf3DIndex, float scale, float sumDist2, TSharedCoordinates& tc) const
{
	float pd2 = sumDist2 + tc.m_wMinAlpha2 + tc.m_wMinBeta2;
	int iMaxDist = (int)(sqrt(tc.m_maxDist2 - sumDist2));
	const float w = tc.m_wTheta;
	const int i = tc.m_iTheta;
	for (int ii = i - iMaxDist; (ii <= i); ii++) {
		if (ii >= 0) {
			float minDist2Bound = Square(w + (float)ii - (float)i);
			if (minDist2Bound + pd2 < tc.m_maxDist2) {
				// there is a chance of having grid point value in the distance smaller than specified
				m_pdf2->GetValShepard(m_pdf3Dslices[pdf3DIndex][ii], scale * m_pdf3Dscale[pdf3DIndex][ii],
					minDist2Bound + sumDist2, tc);
			}
		}
	}
	for (int ii = i + 1; (ii < m_slicesPerTheta) && (ii <= i + 1 + iMaxDist); ii++) {
		// The distance along 
		float minDist2Bound = Square((float)ii - (float)i - w);
		if (minDist2Bound + pd2 < tc.m_maxDist2) {
			// there is a chance of having grid point value in the distance smaller than specified
			m_pdf2->GetValShepard(m_pdf3Dslices[pdf3DIndex][ii], scale * m_pdf3Dscale[pdf3DIndex][ii],
				minDist2Bound + sumDist2, tc);
		}
	}

	return;
}//--- getVal --------------------------------------------------------

int
CPDF3D::GetMemory() const
{
	// This is considered without PDF2Dnorm, as this is not required for applications  
	return m_numOfPdf3D * m_slicesPerTheta * (sizeof(int) + sizeof(float)) + m_numOfPdf3D * 2 * sizeof(int*);
}//--- getMemory ---------------------------------------------------

int
CPDF3D::GetMemoryQ() const
{
	int bitsForIndex = (int)(ceilf(log2(m_pdf2->GetNoOfPdf2D() + 1)));
	// index to 2D
	int size = (m_numOfPdf3D * m_slicesPerTheta * bitsForIndex) / 8 + 1; // in Bytes
	// scale
	size += m_numOfPdf3D * m_slicesPerTheta * sizeof(float);

	return size;
}//--- getMemory ---------------------------------------------------

int
CPDF3D::Load(char* prefix, int MLF, int maxPDF2, int algPut)
{
	assert(prefix);
	char fileName[1000];

	// loading data from TXT file
	int nr, nc, minI, maxI;
	sprintf(fileName, "%s_PDF3Dslices.txt", prefix);
	IReadTxtHeader(fileName, &nr, &nc, &minI, &maxI);
	int** tmpiArr = IAllocation2(0, nr - 1, 0, nc - 1);
	assert(nc == m_slicesPerTheta);
	m_numOfPdf3D = nr;
	Reallocate();
	if (maxI >= maxPDF2) {
		printf("ERROR in the BTFBASE for CPDF3D - indexing corrupt\n");
		AbortRun(20000);
	}

	IReadTxt(tmpiArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfPdf3D; irow++)
		for (int jcol = 0; jcol < m_slicesPerTheta; jcol++)
			m_pdf3Dslices[irow][jcol] = tmpiArr[irow][jcol];
	IFree2(tmpiArr, 0, nr - 1, 0, nc - 1);

	float** tmpArr = Allocation2(0, nr - 1, 0, nc - 1);
	sprintf(fileName, "%s_PDF3Dscale.txt", prefix);
	float minF, maxF;
	ReadTxtHeader(fileName, &nr, &nc, &minF, &maxF);
	assert(nc == m_slicesPerTheta);
	m_numOfPdf3D = nr;

	ReadTxt(tmpArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfPdf3D; irow++)
		for (int jcol = 0; jcol < m_slicesPerTheta; jcol++) {
			m_pdf3Dscale[irow][jcol] = tmpArr[irow][jcol];
			if (m_pdf3Dscale[irow][jcol] < 1e-6)
				m_pdf3Dscale[irow][jcol] = 1e-6;
		}

	Free2(tmpArr, 0, nr - 1, 0, nc - 1);
	this->m_stepTheta = 90.f / (float)(m_slicesPerTheta - 1 + 1);

	return 0; // loaded, ok
}//--- load ------------------------------------------------------------------

int
CPDF3D::ImportanceSamplingDeg(int PDF3Dindex1, float w1, int PDF3Dindex2, float w2,
	float q0, float q1, float& theta_i, float& phi_i,
	TSharedCoordinates& tc)
{
	//   // For PDF3D
	//   tc.iTheta = (int)floor(theta_v/tc.stepTheta);
	//   if (tc.iTheta == tc.SlicesPerTheta)
	//     tc.iTheta = tc.SlicesPerTheta-1;
	//   assert((tc.iTheta>=0)&&(tc.iTheta<=tc.SlicesPerTheta-1));
	//   tc.wTheta = (theta_v-tc.iTheta*tc.stepTheta)/tc.stepTheta;
	//   assert((tc.wTheta>=-1e-5)&&(tc.wTheta<=1.f));
	//   if (tc.wTheta < 0.f) tc.wTheta = 0.f;

	int i = tc.m_iTheta;
	if (i == m_slicesPerTheta)
		i = m_slicesPerTheta - 1;
	assert((i >= 0) && (i < m_slicesPerTheta));
	int i2 = i + 1;
	if (i2 == m_slicesPerTheta)
		i2 = i;

	float w11 = w1 * (1.0f - tc.m_wTheta);
	float w12 = w1 * tc.m_wTheta;
	float w21 = w2 * (1.0f - tc.m_wTheta);
	float w22 = w2 * tc.m_wTheta;

	// Now we have to apply the scales in this function
	w11 *= m_pdf3Dscale[PDF3Dindex1][i];
	float sum = w11;
	w12 *= m_pdf3Dscale[PDF3Dindex1][i2];
	sum += w12;
	w21 *= m_pdf3Dscale[PDF3Dindex2][i];
	sum += w21;
	w22 *= m_pdf3Dscale[PDF3Dindex2][i2];
	sum += w22;
	assert(sum > 0.f);
	// This applies to the renormalization according to the weighting
	// coefficients
	float norm = 1.0f / sum;
	w11 *= norm;
	w12 *= norm;
	w21 *= norm;
	w22 *= norm;

	if (i < (m_slicesPerTheta - 1)) {
		// interpolation between four PDF2D
		int PDF2Dindex11 = m_pdf3Dslices[PDF3Dindex1][i];
		int PDF2Dindex12 = m_pdf3Dslices[PDF3Dindex1][i2]; // increasing phi
		int PDF2Dindex21 = m_pdf3Dslices[PDF3Dindex2][i]; // increasing theta
		int PDF2Dindex22 = m_pdf3Dslices[PDF3Dindex2][i2]; // incresing theta and phi

		// Now we can make importance sampling
		return m_pdf2->ImportanceSampling(PDF2Dindex11, w11,
			PDF2Dindex12, w12,
			PDF2Dindex21, w21,
			PDF2Dindex22, w22,
			q0, q1, theta_i, phi_i, tc);
		//    return 0;
	}

	int PDF2Dindex11 = m_pdf3Dslices[PDF3Dindex1][i];
	int PDF2Dindex12 = m_pdf3Dslices[PDF3Dindex1][i2]; // increasing phi
	int PDF2Dindex21 = PDF2Dindex11;
	int PDF2Dindex22 = PDF2Dindex12;

	// Now we can make importance sampling
	return m_pdf2->ImportanceSampling(PDF2Dindex11, w11,
		PDF2Dindex12, w12,
		PDF2Dindex21, w21,
		PDF2Dindex22, w22,
		q0, q1, theta_i, phi_i, tc);
}//--- importanceSamplingDeg -------------------------------------------------------------


int
CPDF3D::ImportanceSamplingDeg(const int pdf3DIndex1, const float w1, const int pdf3DIndex2, const float w2,
                              const int cntRays, float q0Q1[], float illuminationThetaPhi[],
                              TSharedCoordinates& tc) const
{
	//   // For PDF3D
	//   tc.iTheta = (int)floor(theta_v/tc.stepTheta);
	//   if (tc.iTheta == tc.SlicesPerTheta)
	//     tc.iTheta = tc.SlicesPerTheta-1;
	//   assert((tc.iTheta>=0)&&(tc.iTheta<=tc.SlicesPerTheta-1));
	//   tc.wTheta = (theta_v-tc.iTheta*tc.stepTheta)/tc.stepTheta;
	//   assert((tc.wTheta>=-1e-5)&&(tc.wTheta<=1.f));
	//   if (tc.wTheta < 0.f) tc.wTheta = 0.f;

	auto i = tc.m_iTheta;
	if (i == m_slicesPerTheta)
		i = m_slicesPerTheta - 1;
	assert((i >= 0) && (i < m_slicesPerTheta));
	auto i2 = i + 1;
	if (i2 == m_slicesPerTheta)
		i2 = i;

	float w11 = w1 * (1.0f - tc.m_wTheta);
	float w12 = w1 * tc.m_wTheta;
	float w21 = w2 * (1.0f - tc.m_wTheta);
	float w22 = w2 * tc.m_wTheta;

	// Now we have to apply the scales in this function
	w11 *= m_pdf3Dscale[pdf3DIndex1][i];
	float sum = w11;
	w12 *= m_pdf3Dscale[pdf3DIndex1][i2];
	sum += w12;
	w21 *= m_pdf3Dscale[pdf3DIndex2][i];
	sum += w21;
	w22 *= m_pdf3Dscale[pdf3DIndex2][i2];
	sum += w22;
	assert(sum > 0.f);
	// This applies to the renormalization according to the weighting
	// coefficients
	const float norm = 1.0f / sum;
	w11 *= norm;
	w12 *= norm;
	w21 *= norm;
	w22 *= norm;

	if (i < (m_slicesPerTheta - 1)) {
		// interpolation between four PDF2D
		const int pdf2DIndex11 = m_pdf3Dslices[pdf3DIndex1][i];
		const int pdf2DIndex12 = m_pdf3Dslices[pdf3DIndex1][i2]; // increasing phi
		const int pdf2DIndex21 = m_pdf3Dslices[pdf3DIndex2][i]; // increasing theta
		const int pdf2DIndex22 = m_pdf3Dslices[pdf3DIndex2][i2]; // incresing theta and phi

		// Now we can make importance sampling
		return m_pdf2->ImportanceSampling(pdf2DIndex11, w11,
			pdf2DIndex12, w12,
			pdf2DIndex21, w21,
			pdf2DIndex22, w22,
			cntRays, q0Q1, illuminationThetaPhi, tc);
	}

	const int pdf2DIndex11 = m_pdf3Dslices[pdf3DIndex1][i];
	const int pdf2DIndex12 = m_pdf3Dslices[pdf3DIndex1][i2]; // increasing phi
	const int pdf2DIndex21 = pdf2DIndex11;
	const int pdf2DIndex22 = pdf2DIndex12;

	// Now we can make importance sampling
	return m_pdf2->ImportanceSampling(pdf2DIndex11, w11,
		pdf2DIndex12, w12,
		pdf2DIndex21, w21,
		pdf2DIndex22, w22,
		cntRays, q0Q1, illuminationThetaPhi, tc);
}//--- importanceSamplingDeg -------------------------------------------------------------

void
CPDF3D::GetViewerAlbedoDeg(int PDF3Dindex, float theta_v, float RGB[], float& normParV,
	TSharedCoordinates& tc)
{
	assert((PDF3Dindex >= 0) && (PDF3Dindex < m_numOfPdf3D));
	assert((theta_v >= 0.f) && (theta_v <= 90.f));

	int i = (int)floor(theta_v / m_stepTheta);
	if (i == m_slicesPerTheta)
		i = m_slicesPerTheta - 1;
	assert((i >= 0) && (i < m_slicesPerTheta));

	if (i < (m_slicesPerTheta - 1)) {
		float RGB2[3];
		// interpolation between two values retrieved from PDF2D
		float normPar1, normPar2; // the integral values
		m_pdf2->GetViewerAlbedoDeg(m_pdf3Dslices[PDF3Dindex][i], RGB, normPar1, tc);
		m_pdf2->GetViewerAlbedoDeg(m_pdf3Dslices[PDF3Dindex][i + 1], RGB2, normPar2, tc);
		float wn = (theta_v - i * m_stepTheta) / m_stepTheta;
		float value1 = wn * m_pdf3Dscale[PDF3Dindex][i];
		float value2 = (1.0f - wn) * m_pdf3Dscale[PDF3Dindex][i + 1];
		float w = value1 / (value1 + value2);
		assert((w >= 0.f) && (w <= 1.f));

		float w2 = 1.0f - w;
		RGB[0] = RGB[0] * w + RGB2[0] * w2;
		RGB[1] = RGB[1] * w + RGB2[1] * w2;
		RGB[2] = RGB[2] * w + RGB2[2] * w2;
	}
	else {
		float normPar1;
		m_pdf2->GetViewerAlbedoDeg(m_pdf3Dslices[PDF3Dindex][i], RGB, normPar1, tc);
	}

#if 0
	// This is original
	normParV = PDF3Dnorm[PDF3Dindex];
#else  
	normParV = 1.0f;
#endif  
	return;
}//--- getVal --------------------------------------------------------

