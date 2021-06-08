/*!**************************************************************************
\file    PDF4D.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  4D PDF indices database
******************************************************************************/

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace std;

#include <AuxFuncs.hpp>
#include <TAlloc.hpp>
#include <PDF1D.hpp>
#include <IndexAB.hpp>
#include <PDF2D.hpp>
#include <PDF3D.hpp>
#include <PDF4D.hpp>

//#########################################################################
//######## CPDF4D #####################################################JF##
//#########################################################################

CPDF4D::CPDF4D(int maxPDF4D, int SlicesPerPhi, CPDF1D* PDF1, CIndexAB* IAB,
	CPDF2D* PDF2, CPDF3D* PDF3, int metric)
{
	assert(maxPDF4D > 0);
	assert(PDF1);
	assert(IAB);
	assert(PDF2);
	assert(PDF3);
	assert(metric >= 0);

	this->m_maxPdf4D = maxPDF4D;
	this->m_pdf1 = PDF1;
	this->m_iab = IAB;
	this->m_pdf2 = PDF2;
	this->m_pdf3 = PDF3;
	this->m_slicesPerPhi = SlicesPerPhi;
	this->m_stepPhi = 360.f / (float)(SlicesPerPhi);
	this->m_numOfPdf4D = 0;

	int size2D = PDF2->GetSlicesPerHemisphere() * PDF1->GetSliceLength();
	int size3D = PDF3->GetSlicesPerTheta() * size2D;
	this->m_size4D = SlicesPerPhi * size3D;

	m_pdf4DSlices = IAllocation2(0, maxPDF4D - 1, 0, SlicesPerPhi - 1);
	assert(m_pdf4DSlices);
	m_pdf4DScale = Allocation2(0, maxPDF4D - 1, 0, SlicesPerPhi - 1);
	assert(m_pdf4DScale);

}//--- CPDF4D ----------------------------------------------------------------

CPDF4D::~CPDF4D()
{
	if (m_pdf4DSlices != NULL)
		IFree2(m_pdf4DSlices, 0, m_maxPdf4D - 1, 0, m_slicesPerPhi - 1);
	if (m_pdf4DScale != NULL)
		Free2(m_pdf4DScale, 0, m_maxPdf4D - 1, 0, m_slicesPerPhi - 1);
}//--- ~CPDF4D ---------------------------------------------------------------

void
CPDF4D::DeleteData()
{
	if (m_pdf4DSlices != NULL)
		IFree2(m_pdf4DSlices, 0, m_maxPdf4D - 1, 0, m_slicesPerPhi - 1);
	if (m_pdf4DScale != NULL)
		Free2(m_pdf4DScale, 0, m_maxPdf4D - 1, 0, m_slicesPerPhi - 1);

	this->m_maxPdf4D = 1;
	this->m_numOfPdf4D = 0;

	m_pdf4DSlices = IAllocation2(0, m_maxPdf4D - 1, 0, m_slicesPerPhi - 1);
	assert(m_pdf4DSlices);
	m_pdf4DScale = Allocation2(0, m_maxPdf4D - 1, 0, m_slicesPerPhi - 1);
	assert(m_pdf4DScale);

	return;
}//--- DeleteData() ------------------------------------------------------------

void
CPDF4D::Reallocate()
{
	// reallocation of database if needed
	if (m_numOfPdf4D >= m_maxPdf4D) {
		int newMaxPDF4D = Max(m_maxPdf4D + 10000, m_numOfPdf4D);
		m_pdf4DSlices = IReallocation2Rows(m_pdf4DSlices, 0,
			m_maxPdf4D - 1, newMaxPDF4D - 1,
			0, m_slicesPerPhi - 1);
		m_pdf4DScale = Reallocation2Rows(m_pdf4DScale, 0,
			m_maxPdf4D - 1, newMaxPDF4D - 1,
			0, m_slicesPerPhi - 1);
		m_maxPdf4D = newMaxPDF4D;
	}
}//--- realloc --------------------------------------------------------


void
CPDF4D::GetVal(int PDF4Dindex, float phi_v, float theta_v,
	float alpha, float beta, float RGB[],
	TSharedCoordinates& tc) const
{
	assert((PDF4Dindex >= 0) && (PDF4Dindex < m_numOfPdf4D));
	assert((beta >= -90.f) && (beta <= 90.f));
	assert((alpha >= -90.f) && (alpha <= 90.f));
	assert((theta_v >= 0.f) && (theta_v <= 90.f));
	assert((phi_v >= 0.f) && (phi_v <= 360.f));

	int i = (int)floor(phi_v / m_stepPhi);
	assert((i >= 0) && (i <= m_slicesPerPhi));
	float w = (phi_v - i * m_stepPhi) / m_stepPhi;
	assert((w >= 0.f) && (w <= 1.f));
	// for interpolation in phi this is needed
	int i2 = i + 1;
	if (i2 == m_slicesPerPhi)
		i2 = 0;

	// interpolation between two values retrieved from PDF2D
	float RGB2[3];
	m_pdf3->GetVal(m_pdf4DSlices[PDF4Dindex][i], theta_v, alpha, beta, RGB, tc);
	m_pdf3->GetVal(m_pdf4DSlices[PDF4Dindex][i2], theta_v, alpha, beta, RGB2, tc);
	float s1 = m_pdf4DScale[PDF4Dindex][i] * (1.f - w);
	float s2 = m_pdf4DScale[PDF4Dindex][i2] * w;
	RGB[0] = s1 * RGB[0] + s2 * RGB2[0];
	RGB[1] = s1 * RGB[1] + s2 * RGB2[1];
	RGB[2] = s1 * RGB[2] + s2 * RGB2[2];
}//--- getVal --------------------------------------------------------

void
CPDF4D::GetVal(int pdf4DIndex, float rgb[], TSharedCoordinates& tc) const
{
	const int i = tc.m_iPhi;
	int i2 = i + 1;
	if (i2 == m_slicesPerPhi)
		i2 = 0;
	const float w = tc.m_wPhi;

	// interpolation between two values retrieved from PDF2D
	float rgb2[3];
	m_pdf3->GetVal(m_pdf4DSlices[pdf4DIndex][i], rgb, tc);
	m_pdf3->GetVal(m_pdf4DSlices[pdf4DIndex][i2], rgb2, tc);
	float s1 = m_pdf4DScale[pdf4DIndex][i] * (1 - w);
	float s2 = m_pdf4DScale[pdf4DIndex][i2] * w;
	rgb[0] = s1 * rgb[0] + s2 * rgb2[0];
	rgb[1] = s1 * rgb[1] + s2 * rgb2[1];
	rgb[2] = s1 * rgb[2] + s2 * rgb2[2];
}//--- getVal --------------------------------------------------------

void
CPDF4D::GetValShepard(int PDF4Dindex, TSharedCoordinates& tc) const
{
	const int i = tc.m_iPhi;
	int i2 = i + 1;
	if (i2 == m_slicesPerPhi)
		i2 = 0;
	const float w = tc.m_wPhi;

	// interpolation between two values retrieved from PDF2D
	// float RGB2[3];
	float pd2 = tc.m_wMinTheta2 + tc.m_wMinAlpha2 + tc.m_wMinBeta2;
	int intMaxDist = (int)floorf(tc.m_maxDist);

	for (int ii = i - intMaxDist; (ii <= i); ii++) {
		if (ii >= 0) {
			float minDist2Bound = Square(w + (float)ii - (float)i);
			if (minDist2Bound + pd2 < tc.m_maxDist2) {
				// there is a chance of having grid point value in
				// the distance smaller than specified
				m_pdf3->GetValShepard(m_pdf4DSlices[PDF4Dindex][ii],
					m_pdf4DScale[PDF4Dindex][ii],
					minDist2Bound, tc);
			}
		}
	} // for ii

	for (int ii = i + 1; (ii < m_slicesPerPhi) && (ii <= i + 1 + intMaxDist); ii++) {
		float minDist2Bound = Square((float)ii - (float)i - w);
		if (minDist2Bound + pd2 < tc.m_maxDist2) {
			// there is a chance of having grid point value in the
			// distance smaller than specified
			m_pdf3->GetValShepard(m_pdf4DSlices[PDF4Dindex][ii],
				m_pdf4DScale[PDF4Dindex][ii],
				minDist2Bound, tc);
		}
	} // for ii

	return;
}//--- getValShepard --------------------------------------------------------

// S_6 .. the number of 4D PDF functions in the 4D database
int
CPDF4D::GetNoOfPdf4D() const
{
	return m_numOfPdf4D;
}//--- getNoOfPDF4D ----------------------------------------------------------

int
CPDF4D::GetMemory() const
{
	// This is considered without PDF2Dnorm, as this is not required for applications  
	return m_numOfPdf4D * m_slicesPerPhi * (sizeof(int) + sizeof(float)) + m_numOfPdf4D * 2 * sizeof(int*);
}//--- getMemory ---------------------------------------------------

int
CPDF4D::GetMemoryQ() const
{
	int bitsForIndex = (int)(ceilf(log2(m_pdf3->GetNoOfPdf3D() + 1)));
	// index to 3D
	int size = (m_numOfPdf4D * m_slicesPerPhi * bitsForIndex) / 8 + 1; // in Bytes
	// scaling factor
	size += m_numOfPdf4D * m_slicesPerPhi * (sizeof(float));
	return size;
}//--- getMemoryQ ---------------------------------------------------


int
CPDF4D::GetSlicesPerPhi() const
{
	return m_slicesPerPhi;
}//--- getSlicesPerPhi -----------------------------------------------------

int
CPDF4D::Load(char* prefix, int MLF, int maxPDF3D, int algPut)
{
	assert(prefix);
	assert((MLF == 0) || (MLF == 1));

	char fileName[1000];

	// loading data from TXT file
	int nr, nc, minI, maxI;
	sprintf(fileName, "%s_PDF4Dslices.txt", prefix);
	IReadTxtHeader(fileName, &nr, &nc, &minI, &maxI);
	if (maxI >= maxPDF3D) {
		printf("ERROR in the BTFBASE for CPDF4D - indexing corrupt\n");
		AbortRun(20000);
	}

	int** tmpiArr = IAllocation2(0, nr - 1, 0, nc - 1);
	assert(nc == m_slicesPerPhi);
	m_numOfPdf4D = nr;
	Reallocate();

	IReadTxt(tmpiArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfPdf4D; irow++)
		for (int jcol = 0; jcol < m_slicesPerPhi; jcol++)
			m_pdf4DSlices[irow][jcol] = tmpiArr[irow][jcol];
	IFree2(tmpiArr, 0, nr - 1, 0, nc - 1);

	float** tmpArr = Allocation2(0, nr - 1, 0, nc - 1);
	sprintf(fileName, "%s_PDF4Dscale.txt", prefix);
	float minF, maxF;
	ReadTxtHeader(fileName, &nr, &nc, &minF, &maxF);
	assert(nc == m_slicesPerPhi);
	m_numOfPdf4D = nr;

	ReadTxt(tmpArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfPdf4D; irow++)
		for (int jcol = 0; jcol < m_slicesPerPhi; jcol++)
			m_pdf4DScale[irow][jcol] = tmpArr[irow][jcol];

	Free2(tmpArr, 0, nr - 1, 0, nc - 1);

	this->m_stepPhi = 360.f / (float)(m_slicesPerPhi - 1);

	return 0; // ok, loaded
}//--- load ------------------------------------------------------------------

int
CPDF4D::ImportanceSamplingDeg(int PDF4Dindex, float q0, float q1,
	float& theta_i, float& phi_i,
	TSharedCoordinates& tc)
{
	// For PDF4D
	//tc.iPhi = (int)floor(phi_v/tc.stepPhi);
	//assert((tc.iPhi>=0)&&(tc.iPhi<=tc.SlicesPerPhi-1));
	//tc.wPhi = (phi_v-tc.iPhi*tc.stepPhi)/tc.stepPhi;
	//assert((tc.wPhi>=0.f)&&(tc.wPhi<=1.f));

	// for interpolation in phi this is needed
	int i = tc.m_iPhi;
	int i2 = tc.m_iPhi + 1;
	if (i2 == m_slicesPerPhi)
		i2 = 0;

	float w1 = (1.0f - tc.m_wPhi);
	float w2 = tc.m_wPhi;
	// We have to apply the scaling factor that are for theta range
	// between two discretization steps
	w1 *= m_pdf4DScale[PDF4Dindex][i];
	w2 *= m_pdf4DScale[PDF4Dindex][i2];
	// here we make renormalization of weights
	float sum = w1 + w2;
	assert(sum > 0.f);
	w1 /= sum;
	w2 /= sum;

	int PDF3Dindex1 = m_pdf4DSlices[PDF4Dindex][i];
	int PDF3Dindex2 = m_pdf4DSlices[PDF4Dindex][i2];

	// Now we can make importance sampling - 0 .. OK
	return m_pdf3->ImportanceSamplingDeg(PDF3Dindex1, w1, PDF3Dindex2, w2,
		q0, q1, theta_i, phi_i, tc);
}//--- importanceSamplingDeg -------------------------------------------------------------

int
CPDF4D::ImportanceSamplingDeg(const int pdf4Index, const int cntRays,
                              float q0Q1[], float illuminationThetaPhi[],
                              TSharedCoordinates& tc) const
{
	// For PDF4D
	//tc.iPhi = (int)floor(phi_v/tc.stepPhi);
	//assert((tc.iPhi>=0)&&(tc.iPhi<=tc.SlicesPerPhi-1));
	//tc.wPhi = (phi_v-tc.iPhi*tc.stepPhi)/tc.stepPhi;
	//assert((tc.wPhi>=0.f)&&(tc.wPhi<=1.f));

	// for interpolation in phi this is needed
	int i = tc.m_iPhi;
	int i2 = tc.m_iPhi + 1;
	if (i2 == m_slicesPerPhi)
		i2 = 0;

	float w1 = (1.0f - tc.m_wPhi);
	float w2 = tc.m_wPhi;
	// We have to apply the scaling factor that are for theta range
	// between two discretization steps
	w1 *= m_pdf4DScale[pdf4Index][i];
	w2 *= m_pdf4DScale[pdf4Index][i2];
	// here we make renormalization of weights
	float sum = w1 + w2;
	assert(sum > 0.f);
	w1 /= sum;
	w2 /= sum;

	const int pdf3DIndex1 = m_pdf4DSlices[pdf4Index][i];
	const int pdf3DIndex2 = m_pdf4DSlices[pdf4Index][i2];

	// Now we can make importance sampling
	return m_pdf3->ImportanceSamplingDeg(pdf3DIndex1, w1, pdf3DIndex2, w2,
		cntRays, q0Q1, illuminationThetaPhi, tc);
}//--- importanceSamplingDeg -------------------------------------------------------------

//! \brief computes albedo for fixed viewer direction
void
CPDF4D::GetViewerAlbedoDeg(int PDF4Dindex, float theta_v, float phi_v, float RGB[],
	TSharedCoordinates& tc)
{
	assert((PDF4Dindex >= 0) && (PDF4Dindex < m_numOfPdf4D));
	assert((theta_v >= 0.f) && (theta_v <= 90.f));
	assert((phi_v >= 0.f) && (phi_v <= 360.f));

	int i = (int)floor(phi_v / m_stepPhi);
	assert((i >= 0) && (i <= m_slicesPerPhi - 1));
	float w = (phi_v - i * m_stepPhi) / m_stepPhi;
	assert((w >= 0.f) && (w <= 1.f));
	// for interpolation in phi this is needed
	int i2 = i + 1;
	if (i2 == m_slicesPerPhi)
		i2 = 0;

	// interpolation between two values retrieved from PDF2D
	float normPar1, normPar2; // the integral values
	float RGB2[3];
	m_pdf3->GetViewerAlbedoDeg(m_pdf4DSlices[PDF4Dindex][i], theta_v, RGB, normPar1, tc);
	m_pdf3->GetViewerAlbedoDeg(m_pdf4DSlices[PDF4Dindex][i2], theta_v, RGB2, normPar2, tc);

	float s1 = m_pdf4DScale[PDF4Dindex][i] * (1 - w);
	float s2 = m_pdf4DScale[PDF4Dindex][i2] * w;
	float wd = s1 / (s1 + s2);
	assert((wd >= 0.f) && (wd <= 1.0001f));
	float wd2 = 1.0f - wd;
	RGB[0] = RGB[0] * wd + RGB2[0] * wd2;
	RGB[1] = RGB[1] * wd + RGB2[1] * wd2;
	RGB[2] = RGB[2] * wd + RGB2[2] * wd2;

	return;
}
