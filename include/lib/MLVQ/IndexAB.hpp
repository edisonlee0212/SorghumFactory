/*!**************************************************************************
\file    IndexAB.h
\author  J. Filip, V. Havran
\date    15/12/2006
\version 0.00

BTFbase project

The header file for the:  CIE ab colours indices database
******************************************************************************/

#ifndef INDEXAB_h
#define INDEXAB_h

#include <SharCoors.hpp>
#include <VectorColor.hpp>

//#################################################################################
//! \brief IndexAB - database of 1D slices of indices to colours a-b
//#################################################################################

class CIndexAB {
private:
	// the number of allocated 1D index slices
	int m_maxIndexSlices;
	// the data array of 1D colour index slices
	int** m_indexAbBasis;
	// current number of stored 1D index slices
	int m_noOfIndexSlices;
	// length of index slice
	int m_lengthOfSlice;

	// the database of CIE a-b colours
	CVectorColor* m_ab;
	// The shared coordinates to be used for interpolation
	// when retrieving the data from the database
public:
	// constructor, allocates maximum number of 1D index slices, the size of 1D slice,
	// the database of a-b colors
	CIndexAB(int maxIndexSlices, int LengthOfSlice, CVectorColor* AB);
	~CIndexAB();

	// delete the data from the database, but not from the databases below
	void DeleteData();

	// check & reallocation of database if needed
	void Reallocate();

	// get a single colour value specified by sliceindex, slice position and posAB (0,1)
	inline float Get(int sliceIndex, int posBeta, int posAB, TSharedCoordinates& tc) const;
	inline void GetAll(int sliceIndex, int posBeta, float lab[], TSharedCoordinates& tc) const;
	inline void GetAll(int sliceIndex, TSharedCoordinates& tc) const;

	// get single value for arbitrary angle beta and slice index
	void GetVal(int sliceIndex, float beta, float valAB[], TSharedCoordinates& tc) const;
	// Here beta is specified by 'tc'
	void GetVal(int sliceIndex, float valAb[], TSharedCoordinates& tc) const;
	bool GetValShepard(int sliceIndex, float maxDist2, float valAb[],
		TSharedCoordinates& tc) const;

	// S_3 .. the number of 1D color index slices in the 1D database
	int GetNoOfIndexSlices() const;

	//return memory in bytes required by the data representation
	int GetMemory() const;
	int GetMemoryQ() const;
	// returns the length of slice
	int GetSliceLength() const;

	int Load(char* prefix, int MLF, int algPut);
};//--- CIndexAB ---------------------------------------------------------

inline float
CIndexAB::Get(const int sliceIndex, const int posBeta, const int posAB, TSharedCoordinates& tc) const
{
	assert(sliceIndex >= 0);
	assert(posBeta >= 0 && posBeta < m_lengthOfSlice);
	assert(posAB >= 0 && posAB <= 1);

	return m_ab->Get(m_indexAbBasis[sliceIndex][posBeta], posAB, tc);
}//--- get -----------------------------------------------------------

inline void
CIndexAB::GetAll(const int sliceIndex, const int posBeta, float lab[], TSharedCoordinates& tc) const
{
	assert(sliceIndex >= 0);
	assert(posBeta >= 0 && (posBeta < m_lengthOfSlice));

	m_ab->GetAll(m_indexAbBasis[sliceIndex][posBeta], lab, tc);
}//--- get -----------------------------------------------------------

inline void
CIndexAB::GetAll(const int sliceIndex, TSharedCoordinates& tc) const
{
	assert(sliceIndex >= 0);
	assert(tc.m_indexBeta >= 0 && tc.m_indexBeta < m_lengthOfSlice);

	m_ab->GetAll(m_indexAbBasis[sliceIndex][tc.m_indexBeta], tc);
}//--- get -----------------------------------------------------------

#endif // INDEXAB_h

