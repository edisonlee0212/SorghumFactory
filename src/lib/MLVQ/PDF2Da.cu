/* **************************************************************************
\file    PDF2D.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  2D PDF database
******************************************************************************/
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#include <AuxFuncs.hpp>
#include <TAlloc.hpp>
#include <CIELab.hpp>
#include <PDF1D.hpp>
#include <IndexAB.hpp>
#include <PDF2Da.hpp>

// ===========================================================================
// Here is the class representing only colors in 2D (alpha-beta parametrization)
CPDF2DSeparate::CPDF2DColor::CPDF2DColor(int maxPdf2D, int SlicesPerHemi,
	int LengthOfSlice, CIndexAB* iab,
	float** restoredValuesOpt)
{
	this->m_maxPdf2D = maxPdf2D;
	this->m_numOfPdf2D = 0;
	this->m_slicesPerHemisphere = SlicesPerHemi;
	this->m_lengthOfSlice = LengthOfSlice;
	this->m_size2D = SlicesPerHemi * LengthOfSlice;
	this->m_iab = iab;
	m_pdf2DColors = IAllocation2(0, maxPdf2D - 1, 0, SlicesPerHemi - 1);
	assert(m_pdf2DColors);
} // ------- CPDF2Dseparate::CPDF2Dcolor::CPDF2Dcolor -------------------------

CPDF2DSeparate::CPDF2DColor::~CPDF2DColor()
{
	if (m_pdf2DColors != NULL)
		IFree2(m_pdf2DColors, 0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);

} // -------- CPDF2Dseparate::CPDF2Dcolor::~CPDF2Dcolor ---------------------------

void
CPDF2DSeparate::CPDF2DColor::DeleteData()
{
	if (m_pdf2DColors != NULL)
		IFree2(m_pdf2DColors, 0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);

	m_maxPdf2D = 1;
	m_numOfPdf2D = 0;

	m_pdf2DColors = IAllocation2(0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);
	assert(m_pdf2DColors);

	return;
}

int CPDF2DSeparate::CPDF2DColor::GetNumOfPdf2DColor() const
{
	return m_numOfPdf2D;
}

// check & reallocation of database if needed
void
CPDF2DSeparate::CPDF2DColor::Reallocate()
{
	// reallocation of individual databases if needed
	if (m_numOfPdf2D >= m_maxPdf2D) {
		int newMaxPDF2D = Max(m_maxPdf2D + 5000, m_numOfPdf2D);
		m_pdf2DColors = IReallocation2Rows(m_pdf2DColors, 0,
			m_maxPdf2D - 1, newMaxPDF2D - 1,
			0, m_slicesPerHemisphere - 1);
		m_maxPdf2D = newMaxPDF2D;
	}
} // ---------------CPDF2Dseparate::CPDF2Dcolor::realloc ------------------------

// get a single value of A and B
void
CPDF2DSeparate::CPDF2DColor::GetVal(int pdf2DIndex, float alpha, float beta,
	float lab[], TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((beta >= -90.f) && (beta <= 90.f));
	assert((alpha >= -90.f) && (alpha <= 90.f));

	int i = (int)floor((90.f + alpha) / tc.m_stepAlpha);
	assert((i >= 0) && (i <= m_slicesPerHemisphere - 1));
	float w = (alpha + 90.f - i * tc.m_stepAlpha) / tc.m_stepAlpha;
	assert((w >= -1e-5) && (w <= 1.f));
	if (w < 0.f)
		w = 0.f;

	float valAB1[3], valAB2[3];
	// colours
	m_iab->GetVal(m_pdf2DColors[pdf2DIndex][i], beta, valAB1, tc);
	m_iab->GetVal(m_pdf2DColors[pdf2DIndex][i + 1], beta, valAB2, tc);
	lab[1] = valAB1[0] * (1.f - w) + valAB2[0] * w;
	lab[2] = valAB1[1] * (1.f - w) + valAB2[1] * w;

	return;
} // ------------------- CPDF2Dcolor::getVal ---------------------------

// get a single value of A and B in LAB[3] variable
void
CPDF2DSeparate::CPDF2DColor::GetVal(int PDF2Dindex, float lab[], TSharedCoordinates& tc) const
{
	assert((PDF2Dindex >= 0) && (PDF2Dindex < m_numOfPdf2D));

	const int i = tc.m_iAlpha;
	const float w = tc.m_wAlpha;
	float valAB1[3], valAB2[3];
	// colours
	m_iab->GetVal(m_pdf2DColors[PDF2Dindex][i], valAB1, tc);
	m_iab->GetVal(m_pdf2DColors[PDF2Dindex][i + 1], valAB2, tc);
	lab[1] = valAB1[0] * (1.f - w) + valAB2[0] * w;
	lab[2] = valAB1[1] * (1.f - w) + valAB2[1] * w;

	return;
} // ------------------ CPDF2Dcolor::getVal ---------------------------

// get a single value of A and B in LAB[3] variable
bool
CPDF2DSeparate::CPDF2DColor::GetValShepard(int pdf2DIndex, int ii, float sumDist2,
	float lab[], TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));

	// colors - here is the approximation, we do not take care
	// about position of beta coordinate! Index for beta coordinate
	// would be needed, which complicates the computation
	m_iab->GetVal(m_pdf2DColors[pdf2DIndex][ii], &(lab[1]), tc);

	return true;
} // ------------------ CPDF2Dcolor::getVal ---------------------------

//return memory in bytes required by the data representation
int
CPDF2DSeparate::CPDF2DColor::GetMemory() const
{
	// This is considered without PDF2Dnorm, as this is not required for applications  
	return m_numOfPdf2D * m_slicesPerHemisphere * (sizeof(int)) + m_numOfPdf2D * sizeof(int*);
} // --------------------- CPDF2Dcolor::getMemory ----------------------------

int
CPDF2DSeparate::CPDF2DColor::GetMemoryQ() const
{
	// Number of bits to reprezent the information without loss
	int bitsForIndex = (int)(ceilf(log2(m_iab->GetNoOfIndexSlices() + 1)));
	return (m_numOfPdf2D * m_slicesPerHemisphere * bitsForIndex) / 8 + 1;
} // --------------------- CPDF2Dcolor::getMemoryQ ----------------------------


int
CPDF2DSeparate::CPDF2DColor::Load(char* prefix, int mlf, int maxIndexColor, int algPut)
{
	assert(prefix);
	assert((mlf == 0) || (mlf == 1));

	char fileName[1000];
	if (mlf) {
		sprintf(fileName, "%s_PDF2Dcolours", prefix);
	}

	sprintf(fileName, "%s_PDF2Dcolours.txt", prefix);
	int nr, nc, minI, maxI;
	IReadTxtHeader(fileName, &nr, &nc, &minI, &maxI);
	if (maxI >= maxIndexColor) {
		printf("ERROR in the BTFBASE for CPDF2Dseparate:color - indexing corrupt\n");
		AbortRun(20000);
	}

	int** tmpiArr = IAllocation2(0, nr - 1, 0, nc - 1);
	assert(nc == m_slicesPerHemisphere);
	m_numOfPdf2D = nr;
	Reallocate();

	IReadTxt(tmpiArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfPdf2D; irow++) {
		for (int jcol = 0; jcol < m_slicesPerHemisphere; jcol++)
			m_pdf2DColors[irow][jcol] = tmpiArr[irow][jcol];
	}

	IFree2(tmpiArr, 0, nr - 1, 0, nc - 1);
	return 0; // ok, loaded
} // ---------------- CPDF2Dcolor::load ------------------------------------

// =======================================================================
// =======================================================================
// ===========================================================================

// Here is the class representing only luminances in 2D (alpha-beta parametrization)
CPDF2DSeparate::CPDF2DLuminance::CPDF2DLuminance(int maxPdf2D, int slicesPerHemisphere,
	CPDF1D* pdf1, int metric,
	float** restoredValuesOpt)
{
	this->m_maxPdf2D = maxPdf2D;
	this->m_numOfPdf2D = 0;
	this->m_slicesPerHemisphere = slicesPerHemisphere;
	this->m_pdf1 = pdf1;
	this->m_lengthOfSlice = pdf1->GetSliceLength();
	this->m_size2D = slicesPerHemisphere * slicesPerHemisphere;
	m_pdf2DSlices = IAllocation2(0, maxPdf2D - 1, 0, slicesPerHemisphere - 1);
	assert(m_pdf2DSlices);
	m_pdf2DScale = Allocation2(0, maxPdf2D - 1, 0, slicesPerHemisphere - 1);
	assert(m_pdf2DScale);
} // ---------------------- CPDF2Dlum::CPDF2Dlum --------------------------------

CPDF2DSeparate::CPDF2DLuminance::~CPDF2DLuminance()
{
	if (m_pdf2DSlices != NULL)
		IFree2(m_pdf2DSlices, 0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);
	if (m_pdf2DScale != NULL)
		Free2(m_pdf2DScale, 0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);
} // ---------------------- CPDF2Dlum::~CPDF2Dlum --------------------------------

void
CPDF2DSeparate::CPDF2DLuminance::DeleteData()
{
	if (m_pdf2DSlices != NULL)
		IFree2(m_pdf2DSlices, 0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);
	if (m_pdf2DScale != NULL)
		Free2(m_pdf2DScale, 0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);

	m_maxPdf2D = 1;
	m_numOfPdf2D = 0;

	m_pdf2DSlices = IAllocation2(0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);
	assert(m_pdf2DSlices);
	m_pdf2DScale = Allocation2(0, m_maxPdf2D - 1, 0, m_slicesPerHemisphere - 1);
	assert(m_pdf2DScale);
	return;
}

int CPDF2DSeparate::CPDF2DLuminance::GetNumOfPdf2DLuminance() const
{
	return m_numOfPdf2D;
}

// check & reallocation of database if needed
void
CPDF2DSeparate::CPDF2DLuminance::Reallocation()
{
	// reallocation of individual databases if needed
	if (m_numOfPdf2D >= m_maxPdf2D) {
		int newMaxPDF2D = Max(m_maxPdf2D + 5000, m_numOfPdf2D);
		m_pdf2DSlices = IReallocation2Rows(m_pdf2DSlices, 0,
			m_maxPdf2D - 1, newMaxPDF2D - 1,
			0, m_slicesPerHemisphere - 1);
		m_pdf2DScale = Reallocation2Rows(m_pdf2DScale, 0,
			m_maxPdf2D - 1, newMaxPDF2D - 1,
			0, m_slicesPerHemisphere - 1);
		m_maxPdf2D = newMaxPDF2D;
	}
} // ---------------------- CPDF2Dlum::realloc --------------------------------

// Restore the data assuming correct subrange set in the CDataEntry by startIndex
// and endIndex + dim. Restores the slice index by variable 'sliceIndex'.
int
CPDF2DSeparate::CPDF2DLuminance::RestoreLum(float** restoredValues, int pdf2DIndex, TSharedCoordinates& tc)
{
	assert(restoredValues);
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));

	for (int a = 0; a < m_slicesPerHemisphere; a++) {
		for (int b = 0; b < m_lengthOfSlice; b++) {
			restoredValues[0][a * m_lengthOfSlice + b] = Get(pdf2DIndex, a, b, tc);
			/*
#ifdef _DEBUG
			if ((restoredValues[0][a * m_lengthOfSlice + b] > 1e30) ||
				(!isfiniteBTF(restoredValues[0][a * m_lengthOfSlice + b]))) {

				printf("CPDF2Dseparate: Problem with restoreAll a = %d b = %d isp = %d PDF2Dindex = %d"
					, a, b, 0, PDF2Dindex);
				restoredValues[0][a * m_lengthOfSlice + b] = Get(PDF2Dindex, a, b, tc);
				abort();
			}
#endif
*/
		} // for b
	} // for a

	return 1;
} // ---------------------- CPDF2Dlum::restoreLum --------------------------------

// get a single value of A and B
void
CPDF2DSeparate::CPDF2DLuminance::GetVal(int pdf2DIndex, float alpha, float beta,
	float lab[], TSharedCoordinates& tc) const
{
	assert((pdf2DIndex >= 0) && (pdf2DIndex < m_numOfPdf2D));
	assert((beta >= -90.f) && (beta <= 90.f));
	assert((alpha >= -90.f) && (alpha <= 90.f));

	int i = (int)floor((90.f + alpha) / tc.m_stepAlpha);
	assert((i >= 0) && (i <= m_slicesPerHemisphere - 1));
	float w = (alpha + 90.f - i * tc.m_stepAlpha) / tc.m_stepAlpha;
	assert((w >= -1e-5) && (w <= 1.f));
	if (w < 0.f)
		w = 0.f;

	// This is different to compact representation ! we interpolate in luminances
	float L1 = m_pdf2DScale[pdf2DIndex][i] * m_pdf1->GetVal(m_pdf2DSlices[pdf2DIndex][i], beta, tc);
	float L2 = m_pdf2DScale[pdf2DIndex][i + 1] * m_pdf1->GetVal(m_pdf2DSlices[pdf2DIndex][i + 1], beta, tc);
	lab[0] = (1.f - w) * L1 + w * L2;
	return;
} // ---------------------- CPDF2Dlum::getVal --------------------------------


// get a single value of A and B
void
CPDF2DSeparate::CPDF2DLuminance::GetVal(int PDF2Dindex, float lab[], TSharedCoordinates& tc) const
{
	assert((PDF2Dindex >= 0) && (PDF2Dindex < m_numOfPdf2D));

	const int i = tc.m_iAlpha;
	const float w = tc.m_wAlpha;

	// This is different to compact representation ! we interpolate in luminances
	const float l1 = m_pdf2DScale[PDF2Dindex][i] * m_pdf1->GetVal(m_pdf2DSlices[PDF2Dindex][i], tc);
	const float l2 = m_pdf2DScale[PDF2Dindex][i + 1] * m_pdf1->GetVal(m_pdf2DSlices[PDF2Dindex][i + 1], tc);
	lab[0] = (1.f - w) * l1 + w * l2;

	return;
} // ---------------------- CPDF2Dlum::getVal --------------------------------

// get a single value of A and B
void
CPDF2DSeparate::CPDF2DLuminance::GetValShepard(int PDF2Dindex, int ii,
	float scale, float maxDist2, float lab[],
	TSharedCoordinates& tc) const
{
	assert((PDF2Dindex >= 0) && (PDF2Dindex < m_numOfPdf2D));

	// This is different to compact representation ! we interpolate in luminances
	m_pdf1->GetValShepard(m_pdf2DSlices[PDF2Dindex][ii], scale * m_pdf2DScale[PDF2Dindex][ii],
		maxDist2, lab, tc);

	return;
} // ---------------------- CPDF2Dlum::getVal --------------------------------


//return memory in bytes required by the data representation
int
CPDF2DSeparate::CPDF2DLuminance::GetMemory() const
{
	// This is considered without PDF2Dnorm, as this is not required for
	// the use of BTF database, only for compression.
	return m_numOfPdf2D * m_slicesPerHemisphere * (sizeof(int) + sizeof(float)) + m_numOfPdf2D * 2 * sizeof(int*);
} // ---------------------- CPDF2Dlum::getMemory --------------------------------

int
CPDF2DSeparate::CPDF2DLuminance::GetMemoryQ() const
{
	// the indices
	int bitsForIndex = (int)(ceilf(log2(m_pdf1->GetNumOfPdf1D() + 1)));
	int size = (m_numOfPdf2D * m_slicesPerHemisphere * bitsForIndex) / 8 + 1;
	// the scales
	size += m_numOfPdf2D * m_slicesPerHemisphere * sizeof(float);

	return size;
} // --------------------- CPDF2Dlum::getMemoryQ ----------------------------

int
CPDF2DSeparate::CPDF2DLuminance::Load(char* prefix, int mlf, int maxPdf1D, int algPut)
{
	assert(prefix);
	assert((mlf == 0) || (mlf == 1));

	char fileName[1000];
	if (mlf) {
		sprintf(fileName, "%s_PDF2Dslices", prefix);
		sprintf(fileName, "%s_PDF2Dscale", prefix);
	}

	// loading data from TXT files
	// TBD
	int nr, nc, minI, maxI;
	sprintf(fileName, "%s_PDF2Dslices.txt", prefix);
	IReadTxtHeader(fileName, &nr, &nc, &minI, &maxI);
	if (maxI >= maxPdf1D) {
		printf("ERROR in the BTFBASE for CPDF2Dseparate:lum - indexing corrupt\n");
		AbortRun(20000);
	}

	int** tmpiArr = IAllocation2(0, nr - 1, 0, nc - 1);
	assert(nc == m_slicesPerHemisphere);
	m_numOfPdf2D = nr;
	Reallocation();

	IReadTxt(tmpiArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfPdf2D; irow++)
		for (int jcol = 0; jcol < m_slicesPerHemisphere; jcol++)
			m_pdf2DSlices[irow][jcol] = tmpiArr[irow][jcol];

	IFree2(tmpiArr, 0, nr - 1, 0, nc - 1);

	float** tmpArr = Allocation2(0, nr - 1, 0, nc - 1);
	sprintf(fileName, "%s_PDF2Dscale.txt", prefix);
	float minF, maxF;
	ReadTxtHeader(fileName, &nr, &nc, &minF, &maxF);
	assert(nc == m_slicesPerHemisphere);
	m_numOfPdf2D = nr;

	ReadTxt(tmpArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_numOfPdf2D; irow++)
		for (int jcol = 0; jcol < m_slicesPerHemisphere; jcol++) {
			m_pdf2DScale[irow][jcol] = tmpArr[irow][jcol];
			if (m_pdf2DScale[irow][jcol] < 1e-12)
				m_pdf2DScale[irow][jcol] = 1e-12;
		}
	Free2(tmpArr, 0, nr - 1, 0, nc - 1);

	return 0; // ok, loaded
}// ---------------------- CPDF2Dlum::load --------------------------------


