/*!**************************************************************************
\file    VectorColor.cpp
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The main file for the:  CIE ab colours database
******************************************************************************/
#include <cstdio>
#include <cmath>
#include <cassert>

#include <AuxFuncs.hpp>
#include <TAlloc.hpp>
#include <TBTFbase.hpp>
#include <VectorColor.hpp>


//#########################################################################
//######## CVectorColor ###############################################JF##
//#########################################################################

CVectorColor::CVectorColor(int maxVectorColor)
{
	assert(maxVectorColor > 0);

	this->m_startIndex = 0;
	this->m_maxVectorColor = maxVectorColor;
	this->m_noOfColors = 0;
	this->m_numOfChannels = 2;
	m_vectorColorBasis = Allocation2(0, maxVectorColor - 1, 0, m_numOfChannels - 1);
	assert(m_vectorColorBasis);

}//--- CVectorColor --------------------------------------------------

CVectorColor::~CVectorColor()
{
	if (m_vectorColorBasis != NULL)
		Free2(m_vectorColorBasis, 0, m_maxVectorColor - 1, 0, m_numOfChannels - 1);
}//--- ~CVectorColor -------------------------------------------------

void
CVectorColor::DeleteData()
{
	if (m_vectorColorBasis != NULL)
		Free2(m_vectorColorBasis, 0, m_maxVectorColor - 1, 0, m_numOfChannels - 1);

	m_maxVectorColor = 1;
	m_vectorColorBasis = Allocation2(0, m_maxVectorColor - 1, 0, m_numOfChannels - 1);
	assert(m_vectorColorBasis);
	this->m_noOfColors = 0;
}

void
CVectorColor::Reallocate()
{
	if (m_noOfColors >= m_maxVectorColor) {
		int newMaxVectorColor = Max(m_maxVectorColor + 10000, m_noOfColors);
		m_vectorColorBasis = Reallocation2Rows(m_vectorColorBasis, 0,
			m_maxVectorColor - 1, newMaxVectorColor - 1,
			0, m_numOfChannels - 1);
		m_maxVectorColor = newMaxVectorColor;
	}
}//--- realloc ------------------------------------------------------

int
CVectorColor::GetNoOfColors() const
{
	return m_noOfColors;
}//--- getNoOf -------------------------------------------------------

int
CVectorColor::GetMemory() const
{
	return m_noOfColors * m_numOfChannels * sizeof(float) + m_noOfColors * sizeof(float*);
}//--- getMemory ---------------------------------------------------

int
CVectorColor::GetMemoryQ() const
{
	// How much space we need for one index, when properly quantized in AB space
	float* minV = new float[m_numOfChannels + 1];
	float* maxV = new float[m_numOfChannels + 1];
	for (int j = 0; j < m_numOfChannels; j++) {
		minV[j] = 1e30;
		maxV[j] = -1e30;
	}

	for (int i = 0; i < m_noOfColors; i++) {
		for (int j = 0; j < m_numOfChannels; j++) {
			minV[j] = Min(m_vectorColorBasis[i][j], minV[j]);
			maxV[j] = Max(m_vectorColorBasis[i][j], maxV[j]);
		} // for j
	} // i

	int bitsForIndex = 0;
	for (int j = 0; j < m_numOfChannels; j++) {
		int range = (int)(maxV[j] - minV[j]) + 1;
		assert(range >= 0);
		// Lab space is considered now
		// For sure in AB space we take twice as much could be perceieved
		bitsForIndex += (int)(ceilf(log2((float)(range * 2 + 1))));
	} // for j

	// Returns total number of bits needed to represent color
	return (m_noOfColors * bitsForIndex) / 8 + 1; // In bytes
}//--- getMemory ---------------------------------------------------

int
CVectorColor::Load(char* prefix, int MLF, int algPut)
{
	assert(prefix);
	assert((MLF == 0) || (MLF == 1));

	char fileName[1000];

	// loading data from TXT file
	sprintf(fileName, "%s_colors.txt", prefix);
	int nr, nc; float minV, maxV;
	ReadTxtHeader(fileName, &nr, &nc, &minV, &maxV);
	assert(nc == m_numOfChannels);
	m_noOfColors = nr;
	Reallocate();

	float** tmpArr = Allocation2(0, nr - 1, 0, nc - 1);
	ReadTxt(tmpArr, fileName, &nr, &nc);
	for (int irow = 0; irow < m_noOfColors; irow++)
		for (int jcol = 0; jcol < m_numOfChannels; jcol++)
			m_vectorColorBasis[irow][jcol] = tmpArr[irow][jcol];
	Free2(tmpArr, 0, nr - 1, 0, nc - 1);

	return 0; // ok, loaded
}//--- load ----------------------------------------------------------

