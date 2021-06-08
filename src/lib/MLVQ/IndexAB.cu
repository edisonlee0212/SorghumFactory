/*!**************************************************************************
\file    IndexAB.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  CIE ab colours indices database
******************************************************************************/

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>

#include <IndexAB.hpp>
#include <TAlloc.hpp>
#include <GlobalDefs.hpp>
#include <AuxFuncs.hpp>

//#########################################################################
//######## CIndexAB ###################################################JF##
//#########################################################################

CIndexAB::CIndexAB(int maxIndexSlices, int LengthOfSlice, CVectorColor* AB)
{
	assert(maxIndexSlices > 0);
	assert(LengthOfSlice > 0);
	assert(AB != NULL);

	this->m_maxIndexSlices = maxIndexSlices;
	this->m_lengthOfSlice = LengthOfSlice;
	this->m_ab = AB;
	this->m_noOfIndexSlices = 0;

	m_indexAbBasis = IAllocation2(0, maxIndexSlices - 1, 0, LengthOfSlice - 1);
	assert(m_indexAbBasis);

}//--- TDBF2 ---------------------------------------------------------

CIndexAB::~CIndexAB()
{
	if (m_indexAbBasis != NULL)
		IFree2(m_indexAbBasis, 0, m_maxIndexSlices - 1, 0, m_lengthOfSlice - 1);
}//--- TDBF2 ---------------------------------------------------------

void
CIndexAB::DeleteData()
{
	if (m_indexAbBasis != NULL)
		IFree2(m_indexAbBasis, 0, m_maxIndexSlices - 1, 0, m_lengthOfSlice - 1);

	m_maxIndexSlices = 1;
	m_noOfIndexSlices = 0;
	m_indexAbBasis = IAllocation2(0, m_maxIndexSlices - 1, 0, m_lengthOfSlice - 1);
	assert(m_indexAbBasis);

	return;
}//--- TDBF2 ---------------------------------------------------------

void
CIndexAB::Reallocate()
{
	// reallocation of database if needed
	if (m_noOfIndexSlices >= m_maxIndexSlices) {
		int newMaxIndexSlices = Max(m_maxIndexSlices + 10000, m_noOfIndexSlices);
		m_indexAbBasis = IReallocation2Rows(m_indexAbBasis, 0,
			m_maxIndexSlices - 1, newMaxIndexSlices - 1,
			0, m_lengthOfSlice - 1);
		m_maxIndexSlices = newMaxIndexSlices;
	}
}//--- realloc --------------------------------------------------------

void
CIndexAB::GetVal(int sliceIndex, float beta, float valAB[], TSharedCoordinates& tc) const
{
	assert((sliceIndex >= 0) && (sliceIndex < m_noOfIndexSlices));
	assert((beta >= -90.f) && (beta <= 90.f));

	// Here we compute the right index to beta
	int i;
	float w;
	tc.ComputeIndexForAngleBetaDeg(beta, i, w);

	valAB[0] = (1 - w) * Get(sliceIndex, i, 0, tc) + w * Get(sliceIndex, i + 1, 0, tc);
	valAB[1] = (1 - w) * Get(sliceIndex, i, 1, tc) + w * Get(sliceIndex, i + 1, 1, tc);
}//--- getVal --------------------------------------------------------

void
CIndexAB::GetVal(int sliceIndex, float valAb[], TSharedCoordinates& tc) const
{
	assert((sliceIndex >= 0) && (sliceIndex < m_noOfIndexSlices));
	assert((tc.m_iBeta >= 0) && (tc.m_iBeta < m_lengthOfSlice - 1));
	valAb[0] = (1.f - tc.m_wBeta) * Get(sliceIndex, tc.m_iBeta, 0, tc) +
		tc.m_wBeta * Get(sliceIndex, tc.m_iBeta + 1, 0, tc);
	valAb[1] = (1.f - tc.m_wBeta) * Get(sliceIndex, tc.m_iBeta, 1, tc) +
		tc.m_wBeta * Get(sliceIndex, tc.m_iBeta + 1, 1, tc);
}//--- getVal --------------------------------------------------------

bool
CIndexAB::GetValShepard(int sliceIndex, float maxDist2, float valAb[], TSharedCoordinates& tc) const
{
	valAb[0] = (1.f - tc.m_wBeta) * Get(sliceIndex, tc.m_iBeta, 0, tc) +
		tc.m_wBeta * Get(sliceIndex, tc.m_iBeta + 1, 0, tc);
	valAb[1] = (1.f - tc.m_wBeta) * Get(sliceIndex, tc.m_iBeta, 1, tc) +
		tc.m_wBeta * Get(sliceIndex, tc.m_iBeta + 1, 1, tc);
	return true;
}

int
CIndexAB::GetNoOfIndexSlices() const
{
	return m_noOfIndexSlices;
}//--- getNoOfIndexSlices --------------------------------------------

int
CIndexAB::GetMemory() const
{
	return m_noOfIndexSlices * m_lengthOfSlice * sizeof(int) + m_noOfIndexSlices * sizeof(int*);
}//--- getMemory ---------------------------------------------------

int
CIndexAB::GetMemoryQ() const
{
	// Number of bits to reprezent the information without loss
	int bitsForIndex = (int)(ceilf(log2(m_ab->GetNoOfColors() + 1)));

	return (m_noOfIndexSlices * m_lengthOfSlice * bitsForIndex) / 8 + 1; // in Bytes
}//--- getMemory ---------------------------------------------------


int
CIndexAB::GetSliceLength() const
{
	return m_lengthOfSlice;
}//--- getSliceLength ------------------------------------------------

int
CIndexAB::Load(char* prefix, int MLF, int algPut)
{
	assert(prefix);
	assert((MLF == 0) || (MLF == 1));

	char fileName[1000];

	// loading data from TXT file
	sprintf(fileName, "%s_indexAB.txt", prefix);
	int nr, nc; float minV, maxV;
	ReadTxtHeader(fileName, &nr, &nc, &minV, &maxV);
	assert(nc == m_lengthOfSlice);
	m_noOfIndexSlices = nr;
	Reallocate();

	int** tmpArr = IAllocation2(0, nr - 1, 0, nc - 1);
	IReadTxt(tmpArr, fileName, &nr, &nc);

	for (int irow = 0; irow < m_noOfIndexSlices; irow++) {
		for (int jcol = 0; jcol < m_lengthOfSlice; jcol++) {
			m_indexAbBasis[irow][jcol] = tmpArr[irow][jcol];
		}
	}

	IFree2(tmpArr, 0, nr - 1, 0, nc - 1);

	return 0; // ok, loaded
}//--- load ----------------------------------------------------------

